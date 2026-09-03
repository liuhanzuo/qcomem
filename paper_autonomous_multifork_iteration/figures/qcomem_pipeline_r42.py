#!/usr/bin/env python3
"""Render the Q-CoMem write/read pipeline for the R42 manuscript.

The renderer uses only Matplotlib vector primitives and fixed layout constants.
It intentionally contains no empirical values: the figure explains mechanism
and the retention--online-work trade-off without making performance claims.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path(__file__).resolve().parent
STEM = "qcomem_pipeline_r42"

# Okabe--Ito-derived, colorblind-safe palette.
INK = "#25313C"
MUTED = "#607080"
GRID = "#D7DEE4"
PANEL = "#F7F9FA"
WHITE = "#FFFFFF"
BLUE = "#0072B2"
BLUE_PALE = "#E5F2F9"
SKY = "#56B4E9"
SKY_PALE = "#EAF7FC"
GREEN = "#009E73"
GREEN_PALE = "#E7F6F1"
ORANGE = "#E69F00"
ORANGE_PALE = "#FFF3D8"
VERMILLION = "#D55E00"
VERMILLION_PALE = "#FCECE4"
PURPLE = "#CC79A7"
PURPLE_PALE = "#F8EDF4"


def apply_style() -> None:
    """Set publication-oriented deterministic Matplotlib defaults."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": WHITE,
            "savefig.edgecolor": WHITE,
        }
    )


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    face: str,
    edge: str,
    radius: float = 0.018,
    linewidth: float = 1.0,
    zorder: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def flow_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    linewidth: float = 1.2,
    zorder: int = 4,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=8.5,
            linewidth=linewidth,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def stage_label(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    color: str,
) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.4,
        fontweight="bold",
        color=color,
        zorder=5,
    )


def draw_store(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    rounded_box(ax, x, y, w, h, face=GREEN_PALE, edge=GREEN, linewidth=1.15)
    # Minimal database glyph, kept subordinate to the label.
    gx = x + 0.17 * w
    for yy in (y + 0.34 * h, y + 0.50 * h, y + 0.66 * h):
        ax.plot(
            [gx, gx + 0.16 * w],
            [yy, yy],
            transform=ax.transAxes,
            color=GREEN,
            linewidth=1.0,
            solid_capstyle="round",
            zorder=5,
        )
    ax.text(
        x + 0.61 * w,
        y + 0.52 * h,
        "STORE",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.1,
        fontweight="bold",
        color=INK,
        zorder=5,
    )


def draw_hybrid_state(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    rounded_box(ax, x, y, w, h, face=GREEN_PALE, edge=GREEN, linewidth=1.15)
    ax.text(
        x + 0.5 * w,
        y + 0.84 * h,
        "RETAINED COMPLETE HYBRID SPLIT STATE",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.65,
        fontweight="bold",
        color=GREEN,
        zorder=5,
    )
    # Rows are listed bottom-to-top so the visual reading order is residual,
    # attention KV, then convolutional/recurrent state.
    rows = (
        ("lower conv. / recurrent state", PURPLE_PALE, PURPLE),
        ("lower full-attention KV", SKY_PALE, BLUE),
        (r"split residual  $\mathbf{h}_j$", BLUE_PALE, BLUE),
    )
    row_h = 0.155 * h
    row_y = y + 0.10 * h
    for index, (label, face, edge) in enumerate(rows):
        yy = row_y + index * 0.195 * h
        rounded_box(
            ax,
            x + 0.055 * w,
            yy,
            0.89 * w,
            row_h,
            face=face,
            edge=edge,
            radius=0.008,
            linewidth=0.65,
            zorder=3,
        )
        ax.text(
            x + 0.5 * w,
            yy + 0.5 * row_h,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.8,
            color=INK,
            zorder=5,
        )


def draw_ownership_split(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    rounded_box(ax, x, y, w, h, face=WHITE, edge=PURPLE, linewidth=1.1)
    ax.text(
        x + 0.5 * w,
        y + 0.86 * h,
        "OWNERSHIP SPLIT",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.9,
        fontweight="bold",
        color=PURPLE,
        zorder=5,
    )
    gap = 0.035 * w
    cell_w = 0.43 * w
    cell_h = 0.47 * h
    cell_y = y + 0.19 * h
    rounded_box(
        ax,
        x + 0.05 * w,
        cell_y,
        cell_w,
        cell_h,
        face=GREEN_PALE,
        edge=GREEN,
        radius=0.009,
        linewidth=0.7,
        zorder=3,
    )
    rounded_box(
        ax,
        x + 0.05 * w + cell_w + gap,
        cell_y,
        cell_w,
        cell_h,
        face=VERMILLION_PALE,
        edge=VERMILLION,
        radius=0.009,
        linewidth=0.7,
        zorder=3,
    )
    ax.text(
        x + 0.05 * w + 0.5 * cell_w,
        cell_y + 0.56 * cell_h,
        "immutable\ndocument view",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.55,
        linespacing=1.05,
        color=INK,
        zorder=5,
    )
    ax.text(
        x + 0.05 * w + cell_w + gap + 0.5 * cell_w,
        cell_y + 0.60 * cell_h,
        "request-local\nmutable fork",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.55,
        linespacing=1.05,
        color=INK,
        zorder=5,
    )
    ax.text(
        x + 0.05 * w + cell_w + gap + 0.5 * cell_w,
        cell_y + 0.17 * cell_h,
        "COW / rebind",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=4.9,
        color=VERMILLION,
        zorder=5,
    )


def draw_forkaudit_bracket(
    ax: plt.Axes, x_left: float, x_right: float, y: float
) -> None:
    """Draw a quiet validation bracket below the ownership transition."""
    height = 0.018
    ax.plot(
        [x_left, x_left, x_right, x_right],
        [y + height, y, y, y + height],
        transform=ax.transAxes,
        color=PURPLE,
        linewidth=0.9,
        zorder=4,
    )
    ax.text(
        0.5 * (x_left + x_right),
        y - 0.009,
        "ForkAudit  ·  ownership, alias, rebinding checks",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.15,
        color=PURPLE,
        zorder=5,
    )


def draw_tradeoff_rail(ax: plt.Axes) -> None:
    rounded_box(
        ax,
        0.025,
        0.018,
        0.95,
        0.095,
        face="#F2F5F7",
        edge=GRID,
        radius=0.014,
        linewidth=0.75,
        zorder=1,
    )
    ax.text(
        0.045,
        0.065,
        "smaller retained bytes",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.15,
        fontweight="bold",
        color=GREEN,
        zorder=5,
    )
    ax.add_patch(
        FancyArrowPatch(
            (0.35, 0.065),
            (0.65, 0.065),
            transform=ax.transAxes,
            arrowstyle="<->",
            mutation_scale=8,
            linewidth=0.9,
            color=MUTED,
            zorder=4,
        )
    )
    ax.text(
        0.50,
        0.086,
        "RETENTION / ONLINE-WORK TRADE-OFF",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=4.8,
        fontweight="bold",
        color=MUTED,
        zorder=5,
    )
    ax.text(
        0.955,
        0.065,
        "added dequant. + suffix work",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=6.15,
        fontweight="bold",
        color=ORANGE,
        zorder=5,
    )


def render() -> tuple[Path, Path]:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.25), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Lane backgrounds.
    rounded_box(
        ax, 0.015, 0.615, 0.97, 0.35, face=PANEL, edge=GRID, radius=0.018, linewidth=0.8, zorder=0
    )
    rounded_box(
        ax, 0.015, 0.205, 0.97, 0.35, face=PANEL, edge=GRID, radius=0.018, linewidth=0.8, zorder=0
    )
    stage_label(ax, 0.035, 0.928, "1  OFFLINE WRITE", color=BLUE)
    stage_label(ax, 0.035, 0.518, "2  ONLINE READ", color=ORANGE)

    # Offline write, left to right.
    top_y = 0.700
    top_h = 0.135
    rounded_box(ax, 0.040, top_y, 0.105, top_h, face=BLUE_PALE, edge=BLUE)
    ax.text(
        0.0925,
        top_y + 0.5 * top_h,
        "DOCUMENT",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.25,
        fontweight="bold",
        color=INK,
        zorder=5,
    )

    rounded_box(ax, 0.177, 0.683, 0.118, 0.168, face=BLUE_PALE, edge=BLUE)
    ax.text(
        0.236,
        0.780,
        "LOWER LAYERS",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.55,
        fontweight="bold",
        color=BLUE,
        zorder=5,
    )
    ax.text(
        0.236,
        0.731,
        r"$[0,j)$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.0,
        color=INK,
        zorder=5,
    )

    draw_hybrid_state(ax, 0.330, 0.653, 0.295, 0.228)

    rounded_box(ax, 0.663, 0.675, 0.145, 0.184, face=ORANGE_PALE, edge=ORANGE)
    ax.text(
        0.7355,
        0.806,
        "STATE-TYPE PACKING",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.2,
        fontweight="bold",
        color=ORANGE,
        zorder=5,
    )
    ax.text(
        0.7355,
        0.757,
        "Q4 / Q8",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
        zorder=5,
    )
    ax.text(
        0.7355,
        0.711,
        "groupwise",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.7,
        color=MUTED,
        zorder=5,
    )

    draw_store(ax, 0.846, top_y, 0.114, top_h)

    for start, end in (
        ((0.147, 0.767), (0.174, 0.767)),
        ((0.297, 0.767), (0.327, 0.767)),
        ((0.628, 0.767), (0.660, 0.767)),
        ((0.811, 0.767), (0.843, 0.767)),
    ):
        flow_arrow(ax, start, end)

    # Persistent handoff to online read.
    flow_arrow(ax, (0.903, 0.696), (0.903, 0.477), color=GREEN, linewidth=1.35)

    # Online read proceeds right to left, continuing the snake flow.
    rounded_box(ax, 0.827, 0.340, 0.133, 0.137, face=ORANGE_PALE, edge=ORANGE)
    ax.text(
        0.8935,
        0.4085,
        "FETCH +\nDEQUANTIZE",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.85,
        fontweight="bold",
        linespacing=1.0,
        color=INK,
        zorder=5,
    )

    draw_ownership_split(ax, 0.526, 0.307, 0.252, 0.203)

    rounded_box(ax, 0.366, 0.331, 0.116, 0.155, face=BLUE_PALE, edge=BLUE)
    ax.text(
        0.424,
        0.4085,
        "QUERY\nLOWER-LAYER\nREPLAY",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=4.95,
        fontweight="bold",
        linespacing=0.92,
        color=INK,
        zorder=5,
    )

    rounded_box(ax, 0.201, 0.331, 0.122, 0.155, face=SKY_PALE, edge=SKY)
    ax.text(
        0.262,
        0.438,
        "SUFFIX",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.6,
        fontweight="bold",
        color=BLUE,
        zorder=5,
    )
    ax.text(
        0.262,
        0.400,
        r"$[j,L)$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.5,
        color=INK,
        zorder=5,
    )
    ax.text(
        0.262,
        0.363,
        "reconstruct",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.4,
        color=MUTED,
        zorder=5,
    )

    rounded_box(ax, 0.040, 0.340, 0.116, 0.137, face=GREEN_PALE, edge=GREEN)
    ax.text(
        0.098,
        0.4085,
        "DECODE",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.3,
        fontweight="bold",
        color=INK,
        zorder=5,
    )

    for start, end in (
        ((0.824, 0.4085), (0.781, 0.4085)),
        ((0.523, 0.4085), (0.485, 0.4085)),
        ((0.363, 0.4085), (0.326, 0.4085)),
        ((0.198, 0.4085), (0.159, 0.4085)),
    ):
        flow_arrow(ax, start, end)

    draw_forkaudit_bracket(ax, 0.515, 0.790, 0.270)
    draw_tradeoff_rail(ax)

    pdf_path = OUT_DIR / f"{STEM}.pdf"
    png_path = OUT_DIR / f"{STEM}.png"
    fixed_time = datetime(2026, 9, 2, tzinfo=timezone.utc)
    pdf_metadata = {
        "Title": "Q-CoMem offline-write and online-read pipeline",
        "Author": "Anonymous",
        "Subject": "Mechanism schematic without empirical claims",
        "Keywords": "Q-CoMem, retained state, quantization, ForkAudit",
        "Creator": "Deterministic Matplotlib renderer",
        "CreationDate": fixed_time,
        "ModDate": fixed_time,
    }
    fig.savefig(pdf_path, metadata=pdf_metadata)
    fig.savefig(
        png_path,
        dpi=300,
        metadata={"Software": "Deterministic Matplotlib renderer"},
    )
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    render()


if __name__ == "__main__":
    main()
