import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="results/raw/runs.csv")
    parser.add_argument("--ablations", default="results/raw/ablations.csv")
    parser.add_argument("--outdir", default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    runs = pd.read_csv(args.runs)

    mem = runs.groupby(["model_type", "context_len"], as_index=False)["peak_mem_mb"].mean()
    for mtype in mem.model_type.unique():
        sub = mem[mem.model_type == mtype]
        plt.plot(sub.context_len, sub.peak_mem_mb, marker="o", label=mtype)
    plt.xlabel("Context length")
    plt.ylabel("Peak memory (MB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/memory_vs_context.png", dpi=160)
    plt.clf()

    ppl = runs.groupby(["model_type", "context_len"], as_index=False)["val_ppl"].mean()
    for mtype in ppl.model_type.unique():
        sub = ppl[ppl.model_type == mtype]
        plt.plot(sub.context_len, sub.val_ppl, marker="o", label=mtype)
    plt.xlabel("Context length")
    plt.ylabel("Validation perplexity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/ppl_vs_context.png", dpi=160)
    plt.clf()

    tr = runs.groupby("model_type", as_index=False).agg({"peak_mem_mb": "mean", "val_ppl": "mean"})
    plt.scatter(tr.peak_mem_mb, tr.val_ppl)
    for _, r in tr.iterrows():
        plt.text(r.peak_mem_mb, r.val_ppl, r.model_type)
    plt.xlabel("Peak memory (MB)")
    plt.ylabel("Validation perplexity")
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/efficiency_frontier.png", dpi=160)
    plt.clf()

    if os.path.exists(args.ablations):
        ab = pd.read_csv(args.ablations)
        abg = ab.groupby("ablation_name", as_index=False)["val_ppl"].mean()
        plt.bar(abg.ablation_name, abg.val_ppl)
        plt.ylabel("Validation perplexity")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(f"{args.outdir}/ablation_bars.png", dpi=160)


if __name__ == "__main__":
    main()
