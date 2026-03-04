import os

import matplotlib.pyplot as plt
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_PATH = os.path.join(ROOT, "results", "raw", "runs_submission.csv")
ABLATIONS_S33_PATH = os.path.join(ROOT, "results", "raw", "ablations_s33.csv")
CHECKPOINT_DIR = os.path.join(ROOT, "results", "checkpoints", "science-fair")
OUTDIR = os.path.join(ROOT, "stats")
COMBINED_TRAIN_PATH = os.path.join(ROOT, "combined_science_fair_data.csv")


def ensure_outdir():
    os.makedirs(OUTDIR, exist_ok=True)


def plot_ppl_vs_context():
    runs = pd.read_csv(RUNS_PATH)
    grp = (
        runs.groupby(["model_type", "context_len"], as_index=False)["val_ppl"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    for mtype, color in [("baseline", "#4C72B0"), ("hybrid", "#DD8452")]:
        sub = grp[grp["model_type"] == mtype].sort_values("context_len")
        ax.errorbar(
            sub["context_len"],
            sub["mean"],
            yerr=sub["std"].fillna(0),
            label=mtype,
            marker="o",
            capsize=4,
            color=color,
        )
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Validation perplexity (PPL)")
    ax.set_title("Perplexity vs Context Length (mean ± std, 3 seeds)")
    ax.legend(title="Model")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "ppl_vs_context.png"), dpi=200)
    plt.close(fig)


def plot_paired_delta():
    runs = pd.read_csv(RUNS_PATH)
    # pivot per (context_len, seed)
    pivot = (
        runs.pivot_table(
            index=["context_len", "seed"],
            columns="model_type",
            values="val_ppl",
            aggfunc="first",
        )
        .reset_index()
        .dropna()
    )
    pivot["delta"] = pivot["hybrid"] - pivot["baseline"]
    summary = (
        pivot.groupby("context_len", as_index=False)["delta"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    x_labels = summary["context_len"].astype(str)
    bars = ax.bar(
        x_labels,
        summary["mean"],
        yerr=summary["std"].fillna(0),
        capsize=4,
        color="#DD8452",
        alpha=0.8,
        width=0.6,
    )
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("ΔPPL (hybrid − baseline)")
    ax.set_title("Paired PPL Delta vs Context (hybrid − baseline)")
    for bar, n in zip(bars, summary["count"]):
        # place n label just above the top of the bar (toward zero)
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.1 if bar.get_height() < 0 else bar.get_height() - 0.1,
            f"n={int(n)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "paired_ppl_delta.png"), dpi=200)
    plt.close(fig)


def plot_ablations():
    if not os.path.exists(ABLATIONS_S33_PATH):
        return
    ab = pd.read_csv(ABLATIONS_S33_PATH)
    grp = (
        ab.groupby("ablation_name", as_index=False)["val_ppl"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(grp["ablation_name"], grp["mean"], yerr=grp["std"].fillna(0), capsize=4, color="#7A9E9F")
    ax.set_ylabel("Validation perplexity (PPL)")
    ax.set_title("Ablation Study (single-seed, ctx=1024)")
    ax.set_xticklabels(grp["ablation_name"], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "ablation_ppl.png"), dpi=200)
    plt.close(fig)


def plot_training_curves_single(context_len: int, seed: int = 33):
    """Baseline vs hybrid loss curves at a given context length and seed."""
    bl_path = os.path.join(CHECKPOINT_DIR, f"loss_baseline_ctx{context_len}_s{seed}.csv")
    hy_path = os.path.join(CHECKPOINT_DIR, f"loss_hybrid_ctx{context_len}_s{seed}.csv")
    if not (os.path.exists(bl_path) and os.path.exists(hy_path)):
        return
    bl = pd.read_csv(bl_path)
    hy = pd.read_csv(hy_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bl["step"], bl["loss"], label="baseline", color="#4C72B0", alpha=0.9)
    if "loss" in hy.columns and len(hy) > 1:
        ax.plot(hy["step"], hy["loss"], label="hybrid", color="#DD8452", alpha=0.9)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    ax.set_title(f"Training Loss vs Step (ctx={context_len}, seed={seed})")
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"training_curves_ctx{context_len}_s{seed}.png"), dpi=300)
    plt.close(fig)


def plot_training_curves_single_zoom(context_len: int, seed: int = 33, tail_steps: int = 200):
    """Zoomed-in view on the last tail_steps steps to show convergence region in detail."""
    bl_path = os.path.join(CHECKPOINT_DIR, f"loss_baseline_ctx{context_len}_s{seed}.csv")
    hy_path = os.path.join(CHECKPOINT_DIR, f"loss_hybrid_ctx{context_len}_s{seed}.csv")
    if not (os.path.exists(bl_path) and os.path.exists(hy_path)):
        return
    bl = pd.read_csv(bl_path)
    hy = pd.read_csv(hy_path)
    # focus on last tail_steps points
    bl_tail = bl.tail(tail_steps)
    hy_tail = hy.tail(tail_steps) if len(hy) > 1 else hy
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bl_tail["step"], bl_tail["loss"], label="baseline", color="#4C72B0", alpha=0.9)
    if "loss" in hy_tail.columns and len(hy_tail) > 1:
        ax.plot(hy_tail["step"], hy_tail["loss"], label="hybrid", color="#DD8452", alpha=0.9)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    ax.set_title(f"Training Loss (last {tail_steps} steps, ctx={context_len}, seed={seed})")
    # tighten y-limits around the minimum loss region for better resolution
    min_loss = min(bl_tail["loss"].min(), hy_tail["loss"].min())
    ax.set_ylim(bottom=max(0.0, min_loss - 1.0), top=min_loss + 4.0)
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"training_curves_zoom_ctx{context_len}_s{seed}.png"), dpi=300)
    plt.close(fig)


def plot_speed_vs_context():
    """Throughput vs context length for baseline and hybrid."""
    runs = pd.read_csv(RUNS_PATH)
    grp = (
        runs.groupby(["model_type", "context_len"], as_index=False)["toks_per_sec"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    for mtype, color in [("baseline", "#4C72B0"), ("hybrid", "#DD8452")]:
        sub = grp[grp["model_type"] == mtype].sort_values("context_len")
        ax.errorbar(
            sub["context_len"],
            sub["mean"],
            yerr=sub["std"].fillna(0),
            marker="o",
            capsize=4,
            label=mtype,
            color=color,
        )
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Throughput (tokens / second)")
    ax.set_title("Training Throughput vs Context Length (mean ± std)")
    ax.legend(title="Model")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "throughput_vs_context.png"), dpi=200)
    plt.close(fig)


def plot_avg_training_curves_combined():
    """Average loss vs step across seeds, from combined_science_fair_data.csv."""
    if not os.path.exists(COMBINED_TRAIN_PATH):
        return
    df = pd.read_csv(COMBINED_TRAIN_PATH)
    # sanity check columns
    required = {"step", "loss", "type", "context_length", "seed"}
    if not required.issubset(df.columns):
        return
    # average loss per (type, context_length, step)
    grp = (
        df.groupby(["type", "context_length", "step"], as_index=False)["loss"]
        .mean()
        .rename(columns={"loss": "loss_mean"})
    )
    contexts = sorted(df["context_length"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for ax, mtype, title_prefix in zip(
        axes, ["baseline", "hybrid"], ["Baseline", "Hybrid"]
    ):
        sub = grp[grp["type"] == mtype]
        for ctx, color in zip(contexts, ["#999999", "#4C72B0", "#DD8452"]):
            ctx_sub = sub[sub["context_length"] == ctx]
            ax.plot(
                ctx_sub["step"],
                ctx_sub["loss_mean"],
                label=f"ctx={ctx}",
                color=color,
                alpha=0.9,
            )
        ax.set_ylabel("Avg loss")
        ax.set_title(f"{title_prefix}: Average Training Loss vs Step (3 seeds)")
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
        ax.legend()
    axes[-1].set_xlabel("Training step")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "avg_training_curves_by_context.png"), dpi=200)
    plt.close(fig)


def main():
    ensure_outdir()
    plot_ppl_vs_context()
    plot_paired_delta()
    plot_ablations()
    plot_training_curves_single(256, seed=33)
    plot_training_curves_single(512, seed=33)
    plot_training_curves_single(1024, seed=33)
    plot_training_curves_single_zoom(256, seed=33)
    plot_training_curves_single_zoom(512, seed=33)
    plot_training_curves_single_zoom(1024, seed=33)
    plot_speed_vs_context()
    plot_avg_training_curves_combined()
    print(f"Saved plots to {OUTDIR}")


if __name__ == "__main__":
    main()

