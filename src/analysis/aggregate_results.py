import argparse
import math
import os

import pandas as pd


def ci95(std: float, n: int) -> float:
    if n <= 1:
        return float("nan")
    return 1.96 * std / math.sqrt(n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="results/raw/runs.csv")
    parser.add_argument("--ablations", default="results/raw/ablations.csv")
    parser.add_argument("--out", default="results/processed/summary.csv")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    runs = pd.read_csv(args.runs)
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
    grp.to_csv(args.out, index=False)

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
