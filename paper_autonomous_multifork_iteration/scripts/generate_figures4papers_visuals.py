#!/usr/bin/env python3
"""Generate deterministic ForkAudit architecture and teaser figures.

The visual grammar follows the public figures4papers scientific-figure-making
guidance (minimal axes, Helvetica/Arial-like sans serif, semantic blue/green/
red/neutral palette, print-safe encodings, vector-first export).  The drawing
code and scientific content are original to ForkAudit.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"

BLUE = "#0F4D92"
BLUE_LIGHT = "#E9F1F8"
GREEN = "#2E7D32"
GREEN_LIGHT = "#EAF4E8"
RED = "#B64342"
RED_LIGHT = "#F6E6E4"
ORANGE = "#C76D20"
TEAL = "#42949E"
CHARCOAL = "#272727"
MID = "#767676"
LIGHT = "#D8D8D8"
PALE = "#F5F5F5"
WHITE = "#FFFFFF"


def publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.0,
            "font.weight": "regular",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "forkaudit-figures4papers-v4",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.color": CHARCOAL,
            "lines.solid_capstyle": "butt",
        }
    )


def canvas(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    return fig, ax


def rect(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    edge: str = CHARCOAL,
    face: str = WHITE,
    lw: float = 0.9,
    hatch: str | None = None,
    zorder: int = 1,
):
    patch = Rectangle(
        (x, y),
        w,
        h,
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        hatch=hatch,
        joinstyle="miter",
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def label(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 6.6,
    weight: str = "regular",
    ha: str = "center",
    va: str = "center",
    color: str = CHARCOAL,
    rotation: float = 0,
    linespacing: float = 1.15,
    zorder: int = 5,
):
    return ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        ha=ha,
        va=va,
        color=color,
        rotation=rotation,
        linespacing=linespacing,
        zorder=zorder,
    )


def arrow(
    ax,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    color: str = CHARCOAL,
    lw: float = 0.9,
    style: str = "-|>",
    linestyle: str = "-",
    mutation_scale: float = 7.0,
    zorder: int = 3,
):
    patch = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        linestyle=linestyle,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def panel_heading(ax, x0: float, x1: float, text: str) -> None:
    label(ax, (x0 + x1) / 2, 0.955, text, size=8.2, weight="semibold")
    ax.plot([x0, x1], [0.918, 0.918], color=CHARCOAL, lw=0.7, clip_on=False)


def export(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    common = dict(bbox_inches="tight", pad_inches=0.025, facecolor=WHITE)
    fig.savefig(
        OUT / f"{stem}.pdf",
        metadata={
            "Title": stem,
            "Author": "Anonymous",
            "Creator": "ForkAudit deterministic Matplotlib renderer",
            "CreationDate": None,
            "ModDate": None,
        },
        **common,
    )
    # Matplotlib otherwise injects the wall-clock time into dc:date, making
    # identical vector renders hash differently across runs.
    fig.savefig(OUT / f"{stem}.svg", metadata={"Title": stem, "Date": None}, **common)
    fig.savefig(
        OUT / f"{stem}.png",
        dpi=600,
        metadata={"Software": "ForkAudit deterministic Matplotlib renderer"},
        **common,
    )
    plt.close(fig)


def draw_architecture() -> None:
    fig, ax = canvas(7.25, 3.05)

    # Region 1: a compact whole-group ownership factorial.
    panel_heading(ax, 0.015, 0.265, "Ownership factorial")
    x_cols = [0.082, 0.176]
    y_rows = [0.565, 0.315]
    cw, ch = 0.080, 0.188
    label(ax, x_cols[0] + cw / 2, 0.805, "Full-copy\nKV", size=5.2, weight="semibold")
    label(ax, x_cols[1] + cw / 2, 0.805, "Shared-doc\nKV", size=5.2, weight="semibold")
    label(ax, 0.033, y_rows[0] + ch / 2, "Materialized\nGDN", size=5.7, ha="center")
    label(ax, 0.033, y_rows[1] + ch / 2, "Borrowed\nGDN", size=5.7, ha="center")

    cell_text = [
        ("KV copy", "GDN own"),
        ("KV share RO", "GDN own"),
        ("KV copy", "GDN borrow"),
        ("KV share RO", "GDN borrow"),
    ]
    k = 0
    for row, y in enumerate(y_rows):
        for col, x in enumerate(x_cols):
            rect(ax, x, y, cw, ch, edge=LIGHT, face=WHITE, lw=0.75)
            top, bottom = cell_text[k]
            # Shape/line redundancy: solid lines are local materializations;
            # dashed lines denote a read-only shared/borrowed relation.
            kv_ls = "--" if col else "-"
            gdn_ls = "--" if row else "-"
            ax.plot([x + 0.008, x + 0.026], [y + 0.137, y + 0.137], color=BLUE, lw=1.2, ls=kv_ls)
            ax.plot([x + 0.008, x + 0.026], [y + 0.099, y + 0.099], color=GREEN, lw=1.2, ls=gdn_ls)
            label(ax, x + 0.032, y + 0.137, top, size=4.1, ha="left")
            label(ax, x + 0.032, y + 0.099, bottom, size=4.1, ha="left")
            label(ax, x + cw / 2, y + 0.038, f"primary cell {k + 1}", size=4.8, color=MID)
            k += 1
    label(ax, 0.169, 0.225, "four paired cells; one pair controls the whole group", size=5.2, color=MID)
    arrow(ax, 0.270, 0.475, 0.303, 0.475, lw=1.0)
    label(ax, 0.286, 0.510, "same schedule", size=5.5, color=MID)

    # Region 2: phase-aware request lifecycle, one representative request.
    panel_heading(ax, 0.295, 0.790, "Request lifecycle")
    stage_centers = [0.455, 0.605, 0.735]
    for x, txt in zip(stage_centers, ["Setup", "Post-first-transition", "Final"]):
        label(ax, x, 0.823, txt, size=7.0, weight="semibold")

    label(ax, 0.307, 0.700, "KV cache", size=6.7, weight="semibold", ha="left")
    label(ax, 0.307, 0.430, "GDN state", size=6.7, weight="semibold", ha="left")
    rect(ax, 0.307, 0.530, 0.085, 0.120, edge=BLUE, face=BLUE_LIGHT, lw=1.0)
    label(ax, 0.3495, 0.590, "Document\nKV", size=6.0)
    rect(ax, 0.307, 0.260, 0.085, 0.120, edge=GREEN, face=GREEN_LIGHT, lw=1.0)
    label(ax, 0.3495, 0.320, "Persistent\nGDN base", size=6.0)
    arrow(ax, 0.392, 0.590, 0.410, 0.590, color=BLUE)
    arrow(ax, 0.392, 0.320, 0.410, 0.320, color=GREEN)

    # Setup cells.
    rect(ax, 0.410, 0.510, 0.102, 0.160, edge=BLUE, face=WHITE, lw=1.0)
    label(ax, 0.461, 0.590, "document\nregion", size=6.0)
    rect(ax, 0.410, 0.240, 0.102, 0.160, edge=GREEN, face=WHITE, lw=1.0)
    label(ax, 0.461, 0.320, "setup state\nlocal / RO", size=5.8)

    # First-transition boundary.
    ax.plot([0.526, 0.526], [0.205, 0.745], color=ORANGE, lw=1.1)
    label(ax, 0.526, 0.775, "first write: rebind", size=5.6, color=ORANGE)

    # Post-transition KV: policy-dependent document plus private tail.
    for x in [0.540, 0.674]:
        doc_w, tail_w = 0.085, 0.035
        rect(ax, x, 0.510, doc_w, 0.160, edge=BLUE, face=BLUE_LIGHT, lw=1.0)
        rect(ax, x + doc_w, 0.510, tail_w, 0.160, edge=TEAL, face=WHITE, lw=1.0, hatch="///")
        label(ax, x + doc_w / 2, 0.590, "document\nregion", size=5.9)
        label(ax, x + doc_w + tail_w / 2, 0.590, "private\nappend", size=4.5, rotation=90)

    # Post-transition GDN is request-private and peer-disjoint.
    for x in [0.540, 0.674]:
        rect(ax, x, 0.240, 0.120, 0.160, edge=GREEN, face=GREEN_LIGHT, lw=1.0, hatch="..")
        label(ax, x + 0.060, 0.320, "private mutable", size=6.0)

    arrow(ax, 0.512, 0.590, 0.540, 0.590, color=BLUE)
    arrow(ax, 0.660, 0.590, 0.674, 0.590, color=BLUE)
    arrow(ax, 0.512, 0.320, 0.540, 0.320, color=GREEN)
    arrow(ax, 0.660, 0.320, 0.674, 0.320, color=GREEN)
    label(ax, 0.600, 0.172, "policy-bound document / private append / transition-local rebind", size=5.5, color=MID)

    # Region 3: replay classes, explicitly not execution stages.
    panel_heading(ax, 0.815, 0.985, "Audit replay")
    spine_x = 0.850
    ax.plot([spine_x, spine_x], [0.245, 0.755], color=CHARCOAL, lw=1.0)
    arrow(ax, spine_x, 0.270, spine_x, 0.230, lw=1.0)
    ys = [0.700, 0.575, 0.450, 0.325]
    names = ["ownership", "call contract", "FP32 oracle", "live faults"]
    colors = [BLUE, BLUE, TEAL, GREEN]
    markers = ["o", "D", "s", "^"]
    for y, name, color, marker in zip(ys, names, colors, markers):
        ax.scatter([spine_x], [y], s=62, marker=marker, facecolors=WHITE, edgecolors=color, linewidths=1.0, zorder=5)
        label(ax, 0.882, y, name, size=6.3, ha="left")
    for y0, y1 in [(0.635, 0.700), (0.555, 0.575), (0.410, 0.450), (0.305, 0.325)]:
        ax.plot([0.790, 0.833], [y0, y1], color=MID, lw=0.7, ls=(0, (1.2, 2.2)))
    label(ax, 0.900, 0.190, "bounded evidence", size=6.2, weight="semibold")
    label(ax, 0.900, 0.150, "not a new execution stage", size=5.5, color=MID)

    export(fig, "rr2_architecture_figures4papers")


def draw_teaser() -> None:
    fig, ax = canvas(7.25, 2.12)
    # Light separators preserve one dominant left-to-right reading path.
    ax.plot([0.292, 0.292], [0.10, 0.92], color=LIGHT, lw=0.7)
    ax.plot([0.690, 0.690], [0.10, 0.92], color=LIGHT, lw=0.7)

    # 1. Ambiguity/problem.
    label(ax, 0.145, 0.925, "Hybrid prefix fork", size=8.2, weight="semibold")
    rect(ax, 0.030, 0.640, 0.083, 0.150, edge=BLUE, face=BLUE_LIGHT)
    label(ax, 0.0715, 0.715, "Document\nKV", size=6.2)
    rect(ax, 0.030, 0.365, 0.083, 0.150, edge=GREEN, face=GREEN_LIGHT)
    label(ax, 0.0715, 0.440, "GDN\nbase", size=6.2)
    req_y = [0.705, 0.515, 0.325]
    for i, y in enumerate(req_y):
        rect(ax, 0.190, y - 0.055, 0.072, 0.110, edge=CHARCOAL, face=WHITE, lw=0.8)
        label(ax, 0.226, y, f"request {i if i < 2 else 'N'}", size=5.7)
        arrow(ax, 0.113, 0.715, 0.190, y + 0.020, color=BLUE, lw=0.75)
        arrow(ax, 0.113, 0.440, 0.190, y - 0.020, color=GREEN, lw=0.75, linestyle="--")
    ax.plot([0.197, 0.255], [0.245, 0.245], color=RED, lw=1.0, ls="--")
    label(ax, 0.145, 0.195, "mutable alias after a write?", size=5.2, color=RED)
    label(ax, 0.145, 0.095, "matching tokens alone do not prove isolation", size=5.7, color=MID)

    # 2. Structural idea.
    label(ax, 0.491, 0.925, "Phase-aware ownership contract", size=8.2, weight="semibold")
    xs = [0.325, 0.455, 0.585]
    for x, txt in zip(xs, ["Setup", "Post-first-transition", "Final"]):
        label(ax, x + 0.046, 0.790, txt, size=5.9, weight="semibold")
    label(ax, 0.314, 0.610, "KV", size=6.1, weight="semibold", ha="right")
    label(ax, 0.314, 0.365, "GDN", size=6.1, weight="semibold", ha="right")
    # KV lane.
    rect(ax, xs[0], 0.535, 0.092, 0.145, edge=BLUE, face=BLUE_LIGHT)
    label(ax, xs[0] + 0.046, 0.607, "document\npolicy", size=5.8)
    for x in xs[1:]:
        rect(ax, x, 0.535, 0.064, 0.145, edge=BLUE, face=BLUE_LIGHT)
        rect(ax, x + 0.064, 0.535, 0.028, 0.145, edge=TEAL, face=WHITE, hatch="///")
        label(ax, x + 0.032, 0.607, "document", size=5.4)
    # GDN lane.
    rect(ax, xs[0], 0.290, 0.092, 0.145, edge=GREEN, face=WHITE)
    label(ax, xs[0] + 0.046, 0.362, "local or\nRO alias", size=5.8)
    for x in xs[1:]:
        rect(ax, x, 0.290, 0.092, 0.145, edge=GREEN, face=GREEN_LIGHT, hatch="..")
        label(ax, x + 0.046, 0.362, "private\nmutable", size=5.7)
    for y, color in [(0.607, BLUE), (0.362, GREEN)]:
        arrow(ax, 0.427, y, 0.470, y, color=color, lw=0.9)
        arrow(ax, 0.562, y, 0.605, y, color=color, lw=0.9)
    ax.plot([0.442, 0.442], [0.245, 0.725], color=ORANGE, lw=1.0)
    label(ax, 0.491, 0.190, "4 primary cells / fixed schedule / replayable receipts", size=5.1, color=MID)
    arrow(ax, 0.300, 0.120, 0.675, 0.120, color=CHARCOAL, lw=0.8)
    for x, txt in zip([0.344, 0.435, 0.532, 0.626], ["ownership", "calls", "oracle", "mutants"]):
        label(ax, x, 0.085, txt, size=5.5)

    # 3. Bounded evidence.  These values map to registered claims and are not
    # inferred from the drawing.
    label(ax, 0.845, 0.925, "Bounded validation", size=8.2, weight="semibold")
    label(ax, 0.845, 0.855, "primary ForkAudit case study", size=5.2, color=MID)
    rows = [
        ("96", "factor configurations", "exact trajectories"),
        ("8", "dense-attention rows", "max rel. L2 = 0.001743"),
        ("9", "live injected faults", "intended gates"),
        ("N=32", "final allocated delta", "4.90 → 2.23 GiB"),
    ]
    for y, (value, item, outcome) in zip([0.735, 0.555, 0.375, 0.195], rows):
        label(ax, 0.724, y, value, size=8.5, weight="semibold", ha="left", color=BLUE)
        label(ax, 0.790, y + 0.025, item, size=5.8, ha="left", weight="semibold")
        label(ax, 0.790, y - 0.035, outcome, size=5.7, ha="left", color=MID)
        ax.plot([0.716, 0.978], [y - 0.090, y - 0.090], color=LIGHT, lw=0.55)
    label(ax, 0.845, 0.055, "one model · one schedule · no speed or completeness claim", size=4.8, color=RED)

    export(fig, "rr2_teaser_figures4papers")


def main() -> None:
    publication_style()
    draw_architecture()
    draw_teaser()


if __name__ == "__main__":
    main()
