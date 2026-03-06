import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = os.path.dirname(os.path.abspath(__file__))


def _c(r, g, b, a=1.0):
    return (r / 255, g / 255, b / 255, a)


DARK = _c(44, 62, 80)
BLUE = _c(66, 103, 172)
ORANGE = _c(221, 132, 82)
GREEN = _c(39, 174, 96)
RED = _c(192, 57, 43)
GRAY = _c(120, 120, 120)
LIGHT_BG = _c(245, 247, 250)


def make_exec_summary():
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 11)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # --- Header bar ---
    header = plt.Rectangle((0, 10.0), 8.5, 1.0, facecolor=DARK, edgecolor="none")
    ax.add_patch(header)
    ax.text(4.25, 10.65, "Executive Summary", ha="center", va="center",
            fontsize=22, fontweight="bold", color="white", fontfamily="sans-serif")
    ax.text(4.25, 10.25, "Split-Stream Hybrid Language Model Architecture",
            ha="center", va="center", fontsize=12, color=_c(180, 200, 220),
            fontfamily="sans-serif")

    # --- Author line ---
    ax.text(4.25, 9.7, "Samanyu Goyal  |  California High School  |  Math/Computer Science  |  CCCSEF 2026",
            ha="center", va="center", fontsize=9, color=GRAY, fontfamily="sans-serif")

    # --- Divider ---
    ax.plot([0.6, 7.9], [9.45, 9.45], color=_c(220, 220, 220), linewidth=1)

    # --- Problem ---
    y = 9.2
    ax.text(0.6, y, "Problem", fontsize=13, fontweight="bold", color=DARK, fontfamily="sans-serif")
    y -= 0.3
    ax.text(0.6, y,
            "Modern AI relies on the transformer architecture, which demands significant compute.\n"
            "Improving model quality at small parameter scales enables better AI on limited hardware.",
            fontsize=9, color=_c(60, 60, 60), fontfamily="sans-serif", va="top", linespacing=1.6)

    # --- Approach ---
    y -= 0.75
    ax.text(0.6, y, "Approach", fontsize=13, fontweight="bold", color=DARK, fontfamily="sans-serif")
    y -= 0.3
    ax.text(0.6, y,
            "We designed a split-stream hybrid architecture that processes sequences through two parallel\n"
            "pathways — a sliding-window attention stream for local patterns and a selective state-space\n"
            "model (SSM) stream for long-range memory — merged via a KAN (Kolmogorov-Arnold Network)\n"
            "fusion layer. Compared against a parameter-matched 6-layer transformer baseline under\n"
            "identical training conditions (same data, optimizer, learning rate, seeds).",
            fontsize=9, color=_c(60, 60, 60), fontfamily="sans-serif", va="top", linespacing=1.6)

    # --- Key Results banner ---
    y -= 1.55
    banner = plt.Rectangle((0.5, y - 0.05), 7.5, 0.45, facecolor=_c(39, 174, 96, 0.08),
                            edgecolor=GREEN, linewidth=1.2, joinstyle="round")
    ax.add_patch(banner)
    ax.text(4.25, y + 0.17,
            "Key Result:  46% lower perplexity at 1024 context  |  <1% memory overhead  |  Replicated across 3 seeds",
            ha="center", va="center", fontsize=10, fontweight="bold", color=GREEN, fontfamily="sans-serif")

    # --- Results table ---
    y -= 0.65
    ax.text(0.6, y, "Results", fontsize=13, fontweight="bold", color=DARK, fontfamily="sans-serif")
    y -= 0.35

    headers = ["Context", "Baseline PPL", "Hybrid PPL", "Improvement", "Memory Δ", "Speed Δ"]
    rows = [
        ["256",  "14.10 ± 0.33", "8.50 ± 0.27",  "39.7%", "+0.8%", "−80%"],
        ["512",  "13.04 ± 0.30", "7.63 ± 0.15",  "41.5%", "+0.1%", "−84%"],
        ["1024", "13.11 ± 0.05", "7.10 ± 0.04",  "45.9%", "+0.05%", "−67%"],
    ]
    col_x = [0.6, 1.5, 3.0, 4.5, 5.7, 6.7]
    col_colors = ["black", BLUE, ORANGE, GREEN, GREEN, RED]
    col_weights = ["normal", "normal", "normal", "bold", "normal", "normal"]

    for i, h in enumerate(headers):
        ax.text(col_x[i], y, h, fontsize=8.5, fontweight="bold", color=DARK, fontfamily="sans-serif")
    y -= 0.08
    ax.plot([0.5, 7.9], [y, y], color=_c(200, 200, 200), linewidth=0.8)

    for row in rows:
        y -= 0.3
        for i, cell in enumerate(row):
            ax.text(col_x[i], y, cell, fontsize=9, fontweight=col_weights[i],
                    color=col_colors[i], fontfamily="sans-serif")

    # --- Ablation ---
    y -= 0.55
    ax.text(0.6, y, "Ablation Study (1024 tokens)", fontsize=13, fontweight="bold",
            color=DARK, fontfamily="sans-serif")
    y -= 0.35

    abl_headers = ["Configuration", "PPL", "vs Full Model"]
    abl_rows = [
        ["Full hybrid (KAN fusion)", "7.10", "—"],
        ["Sum fusion (no KAN)", "9.98", "+41% worse"],
        ["SSM only (no local)", "14.73", "+107% worse"],
        ["Local only (no SSM)", "16.15", "+127% worse"],
    ]
    abl_x = [0.6, 3.8, 5.2]
    abl_colors_col2 = [GREEN, RED, RED, RED]

    for i, h in enumerate(abl_headers):
        ax.text(abl_x[i], y, h, fontsize=8.5, fontweight="bold", color=DARK, fontfamily="sans-serif")
    y -= 0.08
    ax.plot([0.5, 7.9], [y, y], color=_c(200, 200, 200), linewidth=0.8)

    for ri, row in enumerate(abl_rows):
        y -= 0.3
        ax.text(abl_x[0], y, row[0], fontsize=9, color=_c(60, 60, 60), fontfamily="sans-serif")
        ax.text(abl_x[1], y, row[1], fontsize=9, fontweight="bold", color=_c(60, 60, 60), fontfamily="sans-serif")
        ax.text(abl_x[2], y, row[2], fontsize=9, fontweight="bold", color=abl_colors_col2[ri], fontfamily="sans-serif")

    # --- Mathematical Foundation ---
    y -= 0.5
    ax.text(0.6, y, "Mathematical Foundation", fontsize=12, fontweight="bold",
            color=DARK, fontfamily="sans-serif")
    y -= 0.28
    ax.text(0.6, y,
            "SSM derived from ODE: ds/dt = −λ(t)·s(t) + λ(t)·b(t). Solved via integrating factor,\n"
            "discretized to: sₜ = (1−αₜ)·sₜ₋₁ + αₜ·bₜ. KAN uses 8 radial basis functions for nonlinear fusion.",
            fontsize=8.5, color=_c(60, 60, 60), fontfamily="sans-serif", va="top", linespacing=1.5)

    # --- Conclusion ---
    y -= 0.7
    ax.text(0.6, y, "Conclusion", fontsize=12, fontweight="bold", color=DARK, fontfamily="sans-serif")
    y -= 0.28
    ax.text(0.6, y,
            "The hybrid achieves 40–46% lower perplexity than a parameter-matched transformer across all\n"
            "context lengths, with <1% memory overhead. Tradeoff: 67–84% slower throughput due to sequential\n"
            "SSM recurrence. Ablations confirm every component contributes. Replicated (n=3, 95% CIs).\n"
            "Original memory-reduction hypothesis was not supported; reported the observed quality gain.",
            fontsize=8.5, color=_c(60, 60, 60), fontfamily="sans-serif", va="top", linespacing=1.5)

    # --- Footer ---
    ax.plot([0.6, 7.9], [0.55, 0.55], color=_c(200, 200, 200), linewidth=0.8)
    ax.text(4.25, 0.3,
            "3 random seeds  •  95% confidence intervals  •  3 ablations  •  Parameter gap <0.5%  •  All code written from scratch",
            ha="center", va="center", fontsize=8, color=GRAY, fontfamily="sans-serif", style="italic")

    fig.savefig(os.path.join(OUTDIR, "executive_summary.png"), dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved executive_summary.png to {OUTDIR}")


if __name__ == "__main__":
    make_exec_summary()
