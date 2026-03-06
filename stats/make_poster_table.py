import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def _color(r, g, b, a=1.0):
    return (r / 255, g / 255, b / 255, a)


BLUE = _color(66, 103, 172)
ORANGE = _color(221, 132, 82)
LIGHT_BLUE = _color(66, 103, 172, 0.08)
LIGHT_ORANGE = _color(221, 132, 82, 0.08)
HEADER_BG = _color(44, 62, 80)
HEADER_FG = "white"
WHITE = (1, 1, 1, 1)
GRAY_LINE = _color(200, 200, 200)
GREEN = _color(39, 174, 96)
RED = _color(192, 57, 43)


def make_main_table():
    columns = [
        "Context\nlength",
        "Baseline\nPPL",
        "Hybrid\nPPL",
        "Improvement",
        "Hybrid\ntok/s",
        "Baseline\ntok/s",
    ]

    data = [
        ["256",   "14.10 ± 0.33", "8.50 ± 0.27",  "39.7%",  "12,698",  "63,910"],
        ["512",   "13.04 ± 0.30", "7.63 ± 0.15",  "41.5%",   "6,752",  "43,181"],
        ["1024",  "13.11 ± 0.05", "7.10 ± 0.04",  "45.9%",   "5,353",  "15,978"],
    ]

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    fig.text(
        0.5, 0.95,
        "Main Results: Baseline vs Hybrid (3 seeds, 95% CI)",
        ha="center", va="top", fontsize=16, fontweight="bold",
        fontfamily="sans-serif",
    )

    n_cols = len(columns)
    n_rows = len(data)
    col_widths = [1.2, 1.8, 1.8, 1.4, 1.6, 1.6]
    row_height = 0.9
    x_start = 0.3
    y_header = 4.0

    col_x = [x_start]
    for w in col_widths:
        col_x.append(col_x[-1] + w)

    for i, (header, w) in enumerate(zip(columns, col_widths)):
        rect = plt.Rectangle(
            (col_x[i], y_header), w, row_height,
            facecolor=HEADER_BG, edgecolor="white", linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            col_x[i] + w / 2, y_header + row_height / 2,
            header, ha="center", va="center",
            fontsize=10, fontweight="bold", color=HEADER_FG,
            fontfamily="sans-serif",
        )

    for row_idx, row in enumerate(data):
        y = y_header - (row_idx + 1) * row_height
        bg = _color(245, 247, 250) if row_idx % 2 == 0 else WHITE

        for col_idx, (cell, w) in enumerate(zip(row, col_widths)):
            rect = plt.Rectangle(
                (col_x[col_idx], y), w, row_height,
                facecolor=bg, edgecolor=GRAY_LINE, linewidth=0.8,
            )
            ax.add_patch(rect)

            color = "black"
            fw = "normal"
            fs = 11
            if col_idx == 1:
                color = BLUE
            elif col_idx == 2:
                color = ORANGE
            elif col_idx == 3:
                color = GREEN
                fw = "bold"
                fs = 12
            elif col_idx == 4:
                color = ORANGE
            elif col_idx == 5:
                color = BLUE

            ax.text(
                col_x[col_idx] + w / 2, y + row_height / 2,
                cell, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=color,
                fontfamily="sans-serif",
            )

    fig.text(
        0.5, 0.04,
        "Parameter-matched comparison  |  Same optimizer, LR, data, seeds  |  Lower PPL = better quality",
        ha="center", va="bottom", fontsize=9, color="gray",
        fontfamily="sans-serif", style="italic",
    )

    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    fig.savefig(os.path.join(OUTDIR, "poster_main_table.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_ablation_table():
    columns = ["Configuration", "PPL", "vs Full Model"]

    data = [
        ["Full hybrid (KAN fusion)",  "7.10",   "—"],
        ["Sum fusion (no KAN)",       "9.98",   "+41%  worse"],
        ["SSM only (no local)",       "14.73",  "+107%  worse"],
        ["Local only (no SSM)",       "16.15",  "+127%  worse"],
    ]

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.set_xlim(0, 7.5)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    fig.text(
        0.5, 0.95,
        "Ablation Study — Every Component Matters",
        ha="center", va="top", fontsize=16, fontweight="bold",
        fontfamily="sans-serif",
    )

    col_widths = [3.0, 1.5, 2.5]
    row_height = 0.85
    x_start = 0.25
    y_header = 4.5

    col_x = [x_start]
    for w in col_widths:
        col_x.append(col_x[-1] + w)

    for i, (header, w) in enumerate(zip(columns, col_widths)):
        rect = plt.Rectangle(
            (col_x[i], y_header), w, row_height,
            facecolor=HEADER_BG, edgecolor="white", linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            col_x[i] + w / 2, y_header + row_height / 2,
            header, ha="center", va="center",
            fontsize=11, fontweight="bold", color=HEADER_FG,
            fontfamily="sans-serif",
        )

    row_colors = [
        GREEN,
        _color(230, 126, 34),
        RED,
        RED,
    ]

    for row_idx, row in enumerate(data):
        y = y_header - (row_idx + 1) * row_height
        bg = _color(245, 247, 250) if row_idx % 2 == 0 else WHITE

        for col_idx, (cell, w) in enumerate(zip(row, col_widths)):
            rect = plt.Rectangle(
                (col_x[col_idx], y), w, row_height,
                facecolor=bg, edgecolor=GRAY_LINE, linewidth=0.8,
            )
            ax.add_patch(rect)

            color = "black"
            fw = "normal"
            ha = "center"
            if col_idx == 0:
                ha = "left"
                x_pos = col_x[col_idx] + 0.15
            else:
                x_pos = col_x[col_idx] + w / 2

            if col_idx == 1:
                fw = "bold"
            if col_idx == 2:
                color = row_colors[row_idx]
                fw = "bold" if row_idx > 0 else "normal"

            ax.text(
                x_pos, y + row_height / 2,
                cell, ha=ha, va="center",
                fontsize=11, fontweight=fw, color=color,
                fontfamily="sans-serif",
            )

    fig.text(
        0.5, 0.06,
        "Context = 1024 tokens  |  Removing any component degrades quality significantly",
        ha="center", va="bottom", fontsize=9, color="gray",
        fontfamily="sans-serif", style="italic",
    )

    fig.tight_layout(rect=[0, 0.1, 1, 0.92])
    fig.savefig(os.path.join(OUTDIR, "poster_ablation_table.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_summary_card():
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis("off")

    fig.text(
        0.5, 0.95,
        "Key Findings at a Glance",
        ha="center", va="top", fontsize=18, fontweight="bold",
        fontfamily="sans-serif",
    )

    items = [
        ("46%",    "lower perplexity at 1024 context",    GREEN),
        ("40-46%", "improvement across all context lengths", GREEN),
        ("<0.5%",  "parameter difference (fair comparison)", BLUE),
        ("3",      "random seeds with 95% confidence intervals", BLUE),
        ("3",      "ablations — every component matters",  ORANGE),
    ]

    y_start = 0.78
    y_step = 0.15
    for i, (big_num, desc, color) in enumerate(items):
        y = y_start - i * y_step
        fig.text(
            0.15, y, big_num,
            ha="right", va="center",
            fontsize=22, fontweight="bold", color=color,
            fontfamily="sans-serif",
        )
        fig.text(
            0.18, y, desc,
            ha="left", va="center",
            fontsize=13, color=_color(60, 60, 60),
            fontfamily="sans-serif",
        )

    fig.savefig(os.path.join(OUTDIR, "poster_summary_card.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_tradeoff_table():
    columns = [
        "Context\nlength",
        "Quality\ngain",
        "Memory\noverhead",
        "Speed\ncost",
        "PPL per\n1K tok/s",
        "Verdict",
    ]

    # quality gain = % PPL reduction (higher = better)
    # memory overhead = % increase in peak mem (lower = better)
    # speed cost = % reduction in tok/s (lower = better)
    # PPL per 1K tok/s = hybrid PPL / (hybrid tok/s / 1000) — lower = more efficient
    data = [
        ["256",  "−40%",  "+0.8%",  "−80%",  "0.67",  "Best quality gain"],
        ["512",  "−42%",  "+0.1%",  "−84%",  "1.13",  "Lowest memory cost"],
        ["1024", "−46%",  "+0.05%", "−67%",  "1.33",  "Strongest overall"],
    ]

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    fig.text(
        0.5, 0.95,
        "Tradeoff Analysis: What You Gain vs What You Pay",
        ha="center", va="top", fontsize=16, fontweight="bold",
        fontfamily="sans-serif",
    )

    col_widths = [1.2, 1.3, 1.3, 1.3, 1.3, 2.1]
    row_height = 0.9
    x_start = 0.3
    y_header = 4.0

    col_x = [x_start]
    for w in col_widths:
        col_x.append(col_x[-1] + w)

    for i, (header, w) in enumerate(zip(columns, col_widths)):
        rect = plt.Rectangle(
            (col_x[i], y_header), w, row_height,
            facecolor=HEADER_BG, edgecolor="white", linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            col_x[i] + w / 2, y_header + row_height / 2,
            header, ha="center", va="center",
            fontsize=10, fontweight="bold", color=HEADER_FG,
            fontfamily="sans-serif",
        )

    for row_idx, row in enumerate(data):
        y = y_header - (row_idx + 1) * row_height
        bg = _color(245, 247, 250) if row_idx % 2 == 0 else WHITE

        for col_idx, (cell, w) in enumerate(zip(row, col_widths)):
            rect = plt.Rectangle(
                (col_x[col_idx], y), w, row_height,
                facecolor=bg, edgecolor=GRAY_LINE, linewidth=0.8,
            )
            ax.add_patch(rect)

            color = "black"
            fw = "normal"
            fs = 11
            if col_idx == 1:
                color = GREEN
                fw = "bold"
            elif col_idx == 2:
                color = GREEN
            elif col_idx == 3:
                color = RED
            elif col_idx == 5:
                color = BLUE
                fw = "bold"
                fs = 10

            ax.text(
                col_x[col_idx] + w / 2, y + row_height / 2,
                cell, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=color,
                fontfamily="sans-serif",
            )

    fig.text(
        0.5, 0.04,
        "Quality gain = PPL reduction  |  Memory & speed = hybrid vs baseline  |  Negative = hybrid lower",
        ha="center", va="bottom", fontsize=9, color="gray",
        fontfamily="sans-serif", style="italic",
    )

    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    fig.savefig(os.path.join(OUTDIR, "poster_tradeoff_table.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_main_table()
    make_ablation_table()
    make_summary_card()
    make_tradeoff_table()
    print(f"Saved poster tables to {OUTDIR}")
