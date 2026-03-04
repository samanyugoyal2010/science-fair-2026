import argparse
import math
import os

import pandas as pd


def ci95(std: float, n: int) -> float:
    if n <= 1:
        return float("nan")
    return 1.96 * std / math.sqrt(n)


def direction_from_delta(delta: float, better_when: str, atol: float = 1e-12) -> str:
    if pd.isna(delta):
        return "unclear"
    if abs(delta) <= atol:
        return "unclear"
    if better_when == "lower":
        return "hybrid better" if delta < 0 else "hybrid worse"
    if better_when == "higher":
        return "hybrid better" if delta > 0 else "hybrid worse"
    raise ValueError(f"Unsupported better_when: {better_when}")


def evidence_strength(n_pairs: int) -> str:
    if n_pairs >= 3:
        return "replicated"
    if n_pairs >= 2:
        return "underpowered"
    return "pilot"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="results/raw/runs.csv")
    parser.add_argument("--ablations", default="results/raw/ablations.csv")
    parser.add_argument("--out", default="results/processed/summary.csv")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    runs = pd.read_csv(args.runs)
    if runs.empty:
        raise ValueError(f"Runs file has no rows: {args.runs}")
    if runs["run_id"].duplicated().any():
        dupes = runs.loc[runs["run_id"].duplicated(), "run_id"].tolist()
        raise ValueError(f"Duplicate run_id values found: {dupes[:5]}")

    required_cols = {"run_id", "model_type", "seed", "context_len", "val_ppl", "peak_mem_mb", "toks_per_sec"}
    missing_cols = sorted(required_cols - set(runs.columns))
    if missing_cols:
        raise ValueError(f"Runs file missing required columns: {missing_cols}")

    grp = runs.groupby(["model_type", "context_len"], as_index=False).agg(
        n=("val_ppl", "count"),
        val_ppl_mean=("val_ppl", "mean"),
        val_ppl_std=("val_ppl", "std"),
        peak_mem_mb_mean=("peak_mem_mb", "mean"),
        peak_mem_mb_std=("peak_mem_mb", "std"),
        toks_per_sec_mean=("toks_per_sec", "mean"),
        toks_per_sec_std=("toks_per_sec", "std"),
    )
    grp["val_ppl_ci95"] = grp.apply(lambda r: ci95(r.val_ppl_std, int(r.n)), axis=1)
    grp["peak_mem_ci95"] = grp.apply(lambda r: ci95(r.peak_mem_mb_std, int(r.n)), axis=1)
    grp["claim_guard"] = grp["n"].apply(lambda n: "ok" if int(n) >= 3 else "n_lt_3")
    grp.to_csv(args.out, index=False)

    pair = runs[runs["model_type"].isin(["baseline", "hybrid"])].copy()
    seed_counts = pair.groupby(["context_len", "seed"])["model_type"].nunique().reset_index(name="n_models")
    if (seed_counts["n_models"] != 2).any():
        bad = seed_counts[seed_counts["n_models"] != 2]
        bad_pairs = bad[["context_len", "seed"]].to_dict(orient="records")
        raise ValueError(f"Inconsistent paired runs (need baseline+hybrid per seed): {bad_pairs}")

    pivot = (
        pair.pivot_table(
            index=["context_len", "seed"],
            columns="model_type",
            values=["val_ppl", "peak_mem_mb", "toks_per_sec"],
            aggfunc="first",
        )
        .sort_index(axis=1)
        .reset_index()
    )
    pivot.columns = [
        "_".join([str(part) for part in c if part]).strip("_") if isinstance(c, tuple) else str(c)
        for c in pivot.columns
    ]

    pivot["delta_vs_paired_baseline_val_ppl"] = pivot["val_ppl_hybrid"] - pivot["val_ppl_baseline"]
    pivot["delta_vs_paired_baseline_peak_mem_mb"] = pivot["peak_mem_mb_hybrid"] - pivot["peak_mem_mb_baseline"]
    pivot["delta_vs_paired_baseline_toks_per_sec"] = pivot["toks_per_sec_hybrid"] - pivot["toks_per_sec_baseline"]
    paired_out = os.path.join(out_dir, "paired_deltas.csv")
    pivot.to_csv(paired_out, index=False)

    paired_summary = pivot.groupby("context_len", as_index=False).agg(
        n_pairs=("seed", "count"),
        delta_val_ppl_mean=("delta_vs_paired_baseline_val_ppl", "mean"),
        delta_val_ppl_std=("delta_vs_paired_baseline_val_ppl", "std"),
        delta_peak_mem_mb_mean=("delta_vs_paired_baseline_peak_mem_mb", "mean"),
        delta_peak_mem_mb_std=("delta_vs_paired_baseline_peak_mem_mb", "std"),
        delta_toks_per_sec_mean=("delta_vs_paired_baseline_toks_per_sec", "mean"),
        delta_toks_per_sec_std=("delta_vs_paired_baseline_toks_per_sec", "std"),
    )
    paired_summary["delta_val_ppl_ci95"] = paired_summary.apply(lambda r: ci95(r.delta_val_ppl_std, int(r.n_pairs)), axis=1)
    paired_summary["delta_peak_mem_mb_ci95"] = paired_summary.apply(
        lambda r: ci95(r.delta_peak_mem_mb_std, int(r.n_pairs)), axis=1
    )
    paired_summary["delta_toks_per_sec_ci95"] = paired_summary.apply(
        lambda r: ci95(r.delta_toks_per_sec_std, int(r.n_pairs)), axis=1
    )
    paired_summary["quality_direction"] = paired_summary["delta_val_ppl_mean"].apply(
        lambda d: direction_from_delta(d, better_when="lower")
    )
    paired_summary["memory_direction"] = paired_summary["delta_peak_mem_mb_mean"].apply(
        lambda d: direction_from_delta(d, better_when="lower")
    )
    paired_summary["speed_direction"] = paired_summary["delta_toks_per_sec_mean"].apply(
        lambda d: direction_from_delta(d, better_when="higher")
    )
    paired_summary["evidence_strength"] = paired_summary["n_pairs"].apply(lambda n: evidence_strength(int(n)))
    paired_summary["claim_guard"] = paired_summary["n_pairs"].apply(lambda n: "ok" if int(n) >= 3 else "n_lt_3")
    paired_summary_out = os.path.join(out_dir, "paired_summary.csv")
    paired_summary.to_csv(paired_summary_out, index=False)

    if args.ablations:
        abls = pd.read_csv(args.ablations)
        ablation_out = os.path.join(out_dir, "ablation_summary.csv")
        abls.groupby(["ablation_name", "context_len"], as_index=False).agg(
            n=("val_ppl", "count"),
            val_ppl_mean=("val_ppl", "mean"),
            peak_mem_mb_mean=("peak_mem_mb", "mean"),
        ).to_csv(ablation_out, index=False)


if __name__ == "__main__":
    main()
