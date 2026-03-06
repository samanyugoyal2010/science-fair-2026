import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"baseline": "#4C72B0", "hybrid": "#DD8452"}


def plot_metric_vs_context(runs, metric, ylabel, outpath, higher_is_better=False):
    grp = runs.groupby(["model_type", "context_len"], as_index=False).agg(
        mean=(metric, "mean"), std=(metric, "std"), n=(metric, "count")
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    for mtype in ["baseline", "hybrid"]:
        sub = grp[grp.model_type == mtype].sort_values("context_len")
        ax.errorbar(
            sub.context_len,
            sub["mean"],
            yerr=sub["std"].fillna(0),
            marker="o",
            capsize=4,
            label=mtype,
            color=COLORS.get(mtype),
        )
    ax.set_xlabel("Context length")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_efficiency_frontier(runs, outpath):
    grp = runs.groupby("model_type", as_index=False).agg(
        mem_mean=("peak_mem_mb", "mean"),
        ppl_mean=("val_ppl", "mean"),
        mem_std=("peak_mem_mb", "std"),
        ppl_std=("val_ppl", "std"),
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    for _, r in grp.iterrows():
        ax.errorbar(
            r.mem_mean,
            r.ppl_mean,
            xerr=r.mem_std if pd.notna(r.mem_std) else 0,
            yerr=r.ppl_std if pd.notna(r.ppl_std) else 0,
            marker="o",
            capsize=4,
            color=COLORS.get(r.model_type, "#333"),
        )
        ax.annotate(
            r.model_type,
            (r.mem_mean, r.ppl_mean),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
        )
    ax.set_xlabel("Peak memory (MB)")
    ax.set_ylabel("Validation perplexity")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_ablation_bars(ablations, outpath):
    abg = ablations.groupby("ablation_name", as_index=False).agg(
        mean=("val_ppl", "mean"), std=("val_ppl", "std")
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        abg.ablation_name,
        abg["mean"],
        yerr=abg["std"].fillna(0),
        capsize=4,
        color="#7A9E9F",
    )
    ax.set_ylabel("Validation perplexity")
    ax.set_xticklabels(abg.ablation_name, rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_paired_deltas(runs, outpath):
    pair = runs[runs["model_type"].isin(["baseline", "hybrid"])].copy()
    pivot = pair.pivot_table(
        index=["context_len", "seed"],
        columns="model_type",
        values="val_ppl",
        aggfunc="first",
    ).reset_index()
    pivot["delta"] = pivot["hybrid"] - pivot["baseline"]
    summary = pivot.groupby("context_len", as_index=False).agg(
        mean=("delta", "mean"), std=("delta", "std"), n=("delta", "count")
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        summary.context_len.astype(str),
        summary["mean"],
        yerr=summary["std"].fillna(0),
        capsize=4,
        color="#DD8452",
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Context length")
    ax.set_ylabel("PPL delta (hybrid \u2212 baseline)")
    for bar, n in zip(bars, summary["n"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def plot_training_curves(loss_dir, outpath):
    """Plot loss-vs-step from per-run CSV logs produced by train.py."""
    csvs = sorted(glob.glob(os.path.join(loss_dir, "loss_*.csv")))
    if not csvs:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for path in csvs:
        name = os.path.basename(path).replace("loss_", "").replace(".csv", "")
        df = pd.read_csv(path)
        if df.empty:
            continue
        mtype = "hybrid" if "hybrid" in name else "baseline"
        alpha = 0.5 if len(csvs) > 4 else 0.8
        ax.plot(df["step"], df["loss"].astype(float), label=name, alpha=alpha, color=COLORS.get(mtype))
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="results/raw/runs.csv")
    parser.add_argument("--ablations", default="results/raw/ablations.csv")
    parser.add_argument("--outdir", default="results/figures")
    parser.add_argument("--loss-dir", default=None, help="Directory with loss_*.csv files from training")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    runs = pd.read_csv(args.runs)

    plot_metric_vs_context(runs, "peak_mem_mb", "Peak memory (MB)", f"{args.outdir}/memory_vs_context.png")
    plot_metric_vs_context(runs, "val_ppl", "Validation perplexity", f"{args.outdir}/ppl_vs_context.png")
    plot_efficiency_frontier(runs, f"{args.outdir}/efficiency_frontier.png")
    plot_paired_deltas(runs, f"{args.outdir}/paired_ppl_delta.png")

    if "toks_per_sec" in runs.columns:
        plot_metric_vs_context(runs, "toks_per_sec", "Tokens / sec", f"{args.outdir}/throughput_vs_context.png", higher_is_better=True)

    if os.path.exists(args.ablations):
        ab = pd.read_csv(args.ablations)
        if not ab.empty:
            plot_ablation_bars(ab, f"{args.outdir}/ablation_bars.png")

    if args.loss_dir:
        plot_training_curves(args.loss_dir, f"{args.outdir}/training_curves.png")


if __name__ == "__main__":
    main()
