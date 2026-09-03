#!/usr/bin/env python3
"""Generate a figures4papers-style pilot from the registered RR2 memory table.

This is an original renderer.  It follows the public visual conventions in
ChenLiu-1996/figures4papers at commit
6790a93af3552539d955d77181c818916e1700b7 without copying repository code.
The output is a candidate visual only and is not merged into the manuscript.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


OUT_DIR = Path(__file__).resolve().parent

# Exact values from tables/rr2_memory_table.tex (GiB, N=32, rank median).
POLICIES = (
    "Full-copy\nMaterialized",
    "Full-copy\nBorrowed",
    "Shared-doc\nMaterialized",
    "Shared-doc\nBorrowed",
)
FINAL_ALLOC = np.array([4.901, 4.890, 2.229, 2.229])
SETUP_GENERATION_PEAK = np.array([4.920, 4.907, 2.843, 2.843])
GENERATION_INCREMENT = np.array([0.019, 1.951, 0.019, 1.950])

BLUE_MAIN = "#0F4D92"
BLUE_LIGHT = "#8FB7DB"
RED_STRONG = "#B64342"
RED_LIGHT = "#E9A6A1"
NEUTRAL = "#4D4D4D"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def annotate(ax: plt.Axes, bars, values: np.ndarray) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.07,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.6,
            color=NEUTRAL,
        )


def main() -> None:
    apply_style()
    x = np.arange(len(POLICIES), dtype=float)
    colors = [RED_STRONG, RED_LIGHT, BLUE_MAIN, BLUE_LIGHT]
    hatches = ["", "///", "", "///"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.1, 2.55),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )

    width = 0.34
    left = axes[0]
    bars_final = left.bar(
        x - width / 2,
        FINAL_ALLOC,
        width,
        color=colors,
        edgecolor="black",
        linewidth=0.65,
    )
    bars_peak = left.bar(
        x + width / 2,
        SETUP_GENERATION_PEAK,
        width,
        color=colors,
        edgecolor="black",
        linewidth=0.65,
        alpha=0.46,
    )
    for bars in (bars_final, bars_peak):
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
    annotate(left, bars_final, FINAL_ALLOC)
    left.set_title("Allocator endpoints", loc="left", fontweight="bold")
    left.set_ylabel("GiB above post-priming baseline")
    left.set_xticks(x, POLICIES)
    left.set_ylim(0, 5.65)
    left.tick_params(axis="x", length=0, pad=4)
    left.grid(axis="y", color="#D8D8D8", linewidth=0.45, alpha=0.65)
    left.set_axisbelow(True)
    left.legend(
        handles=(
            Patch(facecolor="#767676", edgecolor="black", label="Final allocation"),
            Patch(
                facecolor="#767676",
                edgecolor="black",
                alpha=0.46,
                label="Setup + generation peak",
            ),
        ),
        loc="upper right",
        fontsize=6.8,
    )

    right = axes[1]
    bars_increment = right.bar(
        x,
        GENERATION_INCREMENT,
        0.62,
        color=colors,
        edgecolor="black",
        linewidth=0.65,
    )
    for bar, hatch in zip(bars_increment, hatches):
        bar.set_hatch(hatch)
    annotate(right, bars_increment, GENERATION_INCREMENT)
    right.set_title("Generation-phase increment", loc="left", fontweight="bold")
    right.set_ylabel("GiB above start of generation")
    right.set_xticks(x, ("FC\nMat.", "FC\nBor.", "SD\nMat.", "SD\nBor."))
    right.set_ylim(0, 2.35)
    right.tick_params(axis="x", length=0, pad=4)
    right.grid(axis="y", color="#D8D8D8", linewidth=0.45, alpha=0.65)
    right.set_axisbelow(True)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.90, bottom=0.24, wspace=0.34)
    for suffix in ("pdf", "png"):
        fig.savefig(
            OUT_DIR / f"memory_phase_pilot.{suffix}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.035,
            facecolor="white",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
