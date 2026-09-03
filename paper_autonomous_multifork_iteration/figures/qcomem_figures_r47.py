#!/usr/bin/env python3
"""Generate the R47 Q-CoMem mechanism and quantization figures.

Rebuild of the R43 figures for the R47 corrections pass.  The visual grammar,
palette, and drawing primitives are unchanged and imported from
``qcomem_figures_r43``; what changes is the *label size budget*.  The R43
figures were authored on a 7.35 in canvas and included at 0.86/0.90 textwidth,
so their 5.8-6.5 pt nominal type rendered at roughly 4-5 pt against 8.9 pt body
text.  Here each figure is authored at the width it is included at
(5.5 in = \\textwidth), so nominal point size is the rendered point size, and no
label is below 8.0 pt.  Buying that room costs horizontal density, so the
single-row R43 compositions are reflowed into bands; no protocol element is
removed except Figure 1's retention/online-work band and Figure 2's scope
banner, both of which restated the surrounding prose verbatim.

Neither figure is an empirical result.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

from qcomem_figures_r43 import (
    BLUE,
    BLUE_2,
    BLUE_PALE,
    GREEN,
    GREEN_PALE,
    INK,
    LIGHT,
    MID,
    ORANGE,
    ORANGE_PALE,
    PALE,
    TEAL,
    TEAL_PALE,
    VIOLET,
    WHITE,
    arrow,
    box,
    elbow_arrow,
    export,
    label,
    publication_style,
)

# Every text object in these figures is drawn at one of these sizes.
BODY = 8.0
HEAD = 9.0

TEXTWIDTH_IN = 5.5


def canvas(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    return fig, ax


def band_heading(ax, x: float, y: float, index: str, text: str) -> None:
    label(ax, x, y, index, size=HEAD, weight="bold", ha="left", color=BLUE)
    label(ax, x + 0.022, y, text, size=HEAD, weight="semibold", ha="left")


def rule(ax, x0: float, x1: float, y: float) -> None:
    ax.plot([x0, x1], [y, y], color=LIGHT, lw=0.75, zorder=1)


def doc_glyph(ax, x: float, y: float, w: float, h: float) -> None:
    for dx, dy in [(0.008, 0.018), (0.004, 0.009), (0.0, 0.0)]:
        box(
            ax,
            x + dx,
            y + dy,
            w,
            h,
            edge=BLUE if dx == 0 else LIGHT,
            face=WHITE,
            lw=1.0 if dx == 0 else 0.7,
            radius=0.005,
            zorder=2 + int(dx == 0),
        )
    for frac in (0.70, 0.50, 0.30):
        ax.plot(
            [x + 0.008, x + 0.78 * w],
            [y + frac * h, y + frac * h],
            color=BLUE_2,
            lw=0.65,
            zorder=5,
        )


def layer_stack(ax, x: float, y: float, w: float, h: float) -> None:
    for (edge, face), (dx, dy) in zip(
        [(GREEN, GREEN_PALE), (BLUE, BLUE_PALE), (GREEN, GREEN_PALE)],
        [(0.0, 0.0), (0.005, 0.012), (0.010, 0.024)],
    ):
        box(ax, x + dx, y + dy, w, h, edge=edge, face=face, lw=0.9, radius=0.006, zorder=2)


def state_leaf(ax, x, y, w, h, text, *, edge, face) -> None:
    box(ax, x, y, w, h, edge=edge, face=face, lw=0.9, radius=0.006)
    ax.plot([x + 0.008, x + 0.008], [y + 0.008, y + h - 0.008], color=edge, lw=2.0, zorder=5)
    label(ax, x + 0.018, y + 0.5 * h, text, size=BODY, ha="left", weight="semibold")


def store_glyph(ax, x: float, y: float, w: float, h: float) -> None:
    ax.add_patch(
        Rectangle((x, y + 0.12 * h), w, 0.76 * h, lw=1.0, edgecolor=GREEN, facecolor=WHITE, zorder=2)
    )
    for yy, face in [(y + 0.62 * h, TEAL_PALE), (y + 0.39 * h, BLUE_PALE), (y + 0.16 * h, GREEN_PALE)]:
        ax.add_patch(
            Rectangle((x + 0.002, yy), w - 0.004, 0.22 * h, edgecolor="none", facecolor=face, zorder=3)
        )
    ax.add_patch(
        Ellipse((x + 0.5 * w, y + 0.88 * h), w, 0.22 * h, facecolor=WHITE, edgecolor=GREEN, lw=1.0, zorder=4)
    )
    ax.add_patch(
        Ellipse((x + 0.5 * w, y + 0.12 * h), w, 0.22 * h, facecolor=GREEN_PALE, edgecolor=GREEN, lw=1.0, zorder=4)
    )
    ax.plot([x, x], [y + 0.12 * h, y + 0.88 * h], color=GREEN, lw=1.0, zorder=5)
    ax.plot([x + w, x + w], [y + 0.12 * h, y + 0.88 * h], color=GREEN, lw=1.0, zorder=5)


def draw_pipeline() -> None:
    fig, ax = canvas(TEXTWIDTH_IN, 3.05)

    # ---------------- Band 1: offline Write ----------------
    band_heading(ax, 0.020, 0.962, "1", "Write once, offline")
    rule(ax, 0.020, 0.980, 0.934)

    doc_glyph(ax, 0.024, 0.782, 0.048, 0.100)
    label(ax, 0.048, 0.740, "document", size=BODY, weight="semibold")
    arrow(ax, 0.082, 0.828, 0.152, 0.828, color=BLUE, lw=1.05)

    layer_stack(ax, 0.162, 0.778, 0.090, 0.100)
    label(ax, 0.208, 0.740, "layers $[0,j)$", size=BODY, weight="semibold")
    arrow(ax, 0.266, 0.828, 0.296, 0.828, color=BLUE, lw=1.05)

    label(ax, 0.405, 0.912, "complete split state", size=BODY, color=MID)
    state_leaf(ax, 0.300, 0.830, 0.210, 0.048, "residual  $h_j^D$", edge=TEAL, face=TEAL_PALE)
    state_leaf(ax, 0.300, 0.768, 0.210, 0.048, "attention KV", edge=BLUE, face=BLUE_PALE)
    state_leaf(ax, 0.300, 0.706, 0.210, 0.048, "GDN state", edge=GREEN, face=GREEN_PALE)
    arrow(ax, 0.520, 0.812, 0.548, 0.812, color=ORANGE, lw=1.05)

    box(ax, 0.556, 0.722, 0.146, 0.166, edge=ORANGE, face=ORANGE_PALE, lw=1.15, radius=0.008)
    label(ax, 0.629, 0.850, "group-wise", size=BODY, weight="semibold", color=ORANGE)
    label(ax, 0.629, 0.806, "pack", size=BODY, weight="bold")
    for i in range(6):
        ax.add_patch(
            Rectangle(
                (0.574 + i * 0.019, 0.742),
                0.014,
                0.024,
                lw=0.55,
                edgecolor=ORANGE,
                facecolor=WHITE if i % 2 else "#FFD9B8",
                zorder=5,
            )
        )
    arrow(ax, 0.712, 0.812, 0.740, 0.812, color=ORANGE, lw=1.05)

    store_glyph(ax, 0.748, 0.730, 0.062, 0.158)
    label(ax, 0.822, 0.808, "Q-CoMem Store", size=BODY, weight="semibold", ha="left")

    # ---------------- Band 2: online Read ----------------
    band_heading(ax, 0.020, 0.660, "2", "Read per query, online")
    rule(ax, 0.020, 0.980, 0.632)

    box(ax, 0.020, 0.482, 0.072, 0.110, edge=GREEN, face=GREEN_PALE, lw=0.95, radius=0.006)
    label(ax, 0.056, 0.537, "Store", size=BODY, weight="semibold")
    arrow(ax, 0.100, 0.537, 0.132, 0.537, color=GREEN, lw=1.05)

    box(ax, 0.140, 0.482, 0.160, 0.110, edge=ORANGE, face=ORANGE_PALE, lw=1.0, radius=0.008)
    label(ax, 0.220, 0.563, "fetch +", size=BODY, color=ORANGE, weight="semibold")
    label(ax, 0.220, 0.511, "dequantize", size=BODY, weight="semibold")
    arrow(ax, 0.308, 0.537, 0.340, 0.537, color=BLUE, lw=1.0)

    box(ax, 0.348, 0.482, 0.192, 0.110, edge=BLUE, face=BLUE_PALE, lw=0.95, radius=0.007)
    label(ax, 0.444, 0.563, "immutable", size=BODY, weight="semibold", color=BLUE)
    label(ax, 0.444, 0.511, "document view", size=BODY)
    arrow(ax, 0.548, 0.537, 0.614, 0.537, color=ORANGE, lw=0.95, linestyle="--")
    label(ax, 0.581, 0.604, "COW / rebind", size=BODY, color=ORANGE)

    box(ax, 0.622, 0.482, 0.214, 0.110, edge=GREEN, face=GREEN_PALE, lw=0.95, radius=0.007)
    ax.plot([0.632, 0.632], [0.496, 0.578], color=GREEN, lw=2.0, zorder=5)
    label(ax, 0.734, 0.563, "request-local", size=BODY, weight="semibold", color=GREEN)
    label(ax, 0.734, 0.511, "mutable state", size=BODY)

    elbow_arrow(
        ax,
        [(0.729, 0.480), (0.729, 0.438), (0.290, 0.438), (0.290, 0.404)],
        color=GREEN,
        lw=0.85,
        linestyle="--",
    )

    ax.add_patch(Ellipse((0.050, 0.348), 0.052, 0.086, facecolor=WHITE, edgecolor=BLUE, lw=1.0, zorder=3))
    label(ax, 0.050, 0.348, "$q$", size=HEAD, weight="bold", color=BLUE)
    label(ax, 0.050, 0.272, "query", size=BODY, color=MID)
    arrow(ax, 0.080, 0.348, 0.188, 0.348, color=BLUE, lw=1.0)

    box(ax, 0.196, 0.294, 0.190, 0.110, edge=BLUE, face=BLUE_PALE, lw=0.95, radius=0.007)
    label(ax, 0.291, 0.375, "replay query", size=BODY, weight="semibold")
    label(ax, 0.291, 0.323, "through $[0,j)$", size=BODY)
    arrow(ax, 0.394, 0.348, 0.416, 0.348, color=BLUE, lw=1.0)

    box(ax, 0.424, 0.294, 0.178, 0.110, edge=TEAL, face=WHITE, lw=1.0, radius=0.007)
    label(ax, 0.513, 0.375, "merge at $j$", size=BODY, weight="semibold")
    label(ax, 0.513, 0.323, "$[h_j^D;\\,h_j^q]$", size=BODY)
    arrow(ax, 0.610, 0.348, 0.632, 0.348, color=TEAL, lw=1.0)

    box(ax, 0.640, 0.294, 0.172, 0.110, edge=BLUE, face=WHITE, lw=0.95, radius=0.007)
    label(ax, 0.726, 0.375, "suffix", size=BODY, weight="semibold")
    label(ax, 0.726, 0.323, "$[j,L)$", size=BODY)
    arrow(ax, 0.820, 0.348, 0.852, 0.348, color=BLUE, lw=1.0)
    label(ax, 0.860, 0.348, "decode", size=BODY, weight="semibold", ha="left")

    elbow_arrow(
        ax,
        [(0.160, 0.480), (0.160, 0.232), (0.513, 0.232), (0.513, 0.292)],
        color=TEAL,
        lw=0.9,
    )
    label(ax, 0.345, 0.192, "document $h_j^D$ bypasses replay", size=BODY, color=TEAL)

    # ---------------- Band 3: audit rail ----------------
    box(ax, 0.020, 0.028, 0.960, 0.092, edge=VIOLET, face=WHITE, lw=0.9, radius=0.008)
    label(ax, 0.036, 0.074, "ForkAudit rail:", size=BODY, weight="semibold", color=VIOLET, ha="left")
    for x, marker, color, name in [
        (0.290, "D", TEAL, "immutability"),
        (0.510, "o", BLUE, "private ownership"),
        (0.800, "^", GREEN, "COW / rebind"),
    ]:
        ax.scatter([x], [0.074], s=42, marker=marker, facecolors=WHITE, edgecolors=color, linewidths=1.05, zorder=5)
        label(ax, x + 0.017, 0.074, name, size=BODY, ha="left", color=INK)

    export(fig, "qcomem_pipeline_r47")


def layer_box(ax, x, y, w, h, index, kind) -> None:
    edge, face = (BLUE, BLUE_PALE) if kind == "Attn" else (GREEN, GREEN_PALE)
    box(ax, x, y, w, h, edge=edge, face=face, lw=0.9, radius=0.004)
    label(ax, x + 0.5 * w, y + 0.68 * h, str(index), size=BODY, weight="bold", color=edge)
    label(ax, x + 0.5 * w, y + 0.30 * h, kind, size=BODY, weight="semibold")
    ax.add_patch(
        Rectangle((x + 0.16 * w, y - 0.042), 0.68 * w, 0.022, edgecolor=edge, facecolor=WHITE, lw=0.7, zorder=4)
    )


def q_badge(ax, x, y, text) -> None:
    box(ax, x, y, 0.052, 0.060, edge=ORANGE, face=ORANGE_PALE, lw=1.0, radius=0.012)
    label(ax, x + 0.026, y + 0.030, text, size=BODY, weight="bold", color=ORANGE)


def draw_quantization_map() -> None:
    fig, ax = canvas(TEXTWIDTH_IN, 3.05)

    # ---------------- (a) backbone and split ----------------
    band_heading(ax, 0.020, 0.958, "a", "Hybrid backbone and split")
    rule(ax, 0.020, 0.980, 0.928)

    x0, w, gap, y, h = 0.022, 0.070, 0.011, 0.690, 0.150
    for i, kind in enumerate(["GDN", "GDN", "GDN", "Attn", "GDN", "GDN", "GDN"]):
        layer_box(ax, x0 + i * (w + gap), y, w, h, i, kind)
    label(ax, 0.305, 0.878, "lower prefix $[0,7)$", size=BODY, weight="semibold")
    label(ax, 0.305, 0.598, "retained-state sidecars", size=BODY, color=MID)

    split_x = x0 + 7 * (w + gap) - gap + 0.014
    ax.plot([split_x, split_x], [0.628, 0.868], color=ORANGE, lw=1.3, zorder=3)
    ax.scatter([split_x], [0.752], s=30, color=ORANGE, zorder=5)
    label(ax, split_x, 0.902, "$j=7$", size=BODY, weight="bold", color=ORANGE)
    ax.plot([split_x + 0.004, 0.652], [0.765, 0.765], color=BLUE, lw=0.8, zorder=3)

    box(ax, 0.652, 0.678, 0.328, 0.176, edge=INK, face=PALE, lw=0.9, radius=0.006)
    label(ax, 0.816, 0.812, "L7\u2013L39 online suffix", size=BODY, weight="bold")
    label(ax, 0.816, 0.766, "24 GDN + 9 Attn layers", size=BODY)
    label(ax, 0.816, 0.716, "no state kept above $j$", size=BODY, color=MID)
    label(ax, 0.020, 0.532, "weights and forward compute stay BF16", size=BODY, weight="semibold",
          color=BLUE, ha="left")

    # ---------------- (b) frozen policy ----------------
    band_heading(ax, 0.020, 0.458, "b", "Frozen Q4/Q4/Q8 policy")
    rule(ax, 0.020, 0.470, 0.428)

    for yy, text, edge, face, qtext in [
        (0.372, "residual  $h_7^D$", TEAL, TEAL_PALE, "Q4"),
        (0.288, "attention KV (L3)", BLUE, BLUE_PALE, "Q4"),
        (0.204, "GDN state (L0\u20132, L4\u20136)", GREEN, GREEN_PALE, "Q8"),
    ]:
        box(ax, 0.020, yy - 0.032, 0.328, 0.064, edge=edge, face=face, lw=0.9, radius=0.006)
        ax.plot([0.030, 0.030], [yy - 0.020, yy + 0.020], color=edge, lw=2.0, zorder=5)
        label(ax, 0.040, yy, text, size=BODY, ha="left", weight="semibold")
        arrow(ax, 0.352, yy, 0.370, yy, color=edge, lw=0.8)
        q_badge(ax, 0.376, yy - 0.030, qtext)

    label(ax, 0.020, 0.140, "cache bits $[8,8,8,4,8,8,8]$", size=BODY, ha="left", color=MID)
    bits = [8, 8, 8, 4, 8, 8, 8]
    bx0, bw, bgap = 0.022, 0.036, 0.008
    for i, bit in enumerate(bits):
        edge = BLUE if i == 3 else GREEN
        face = BLUE_PALE if i == 3 else GREEN_PALE
        ax.add_patch(
            Rectangle((bx0 + i * (bw + bgap), 0.052), bw, 0.050, edgecolor=edge, facecolor=face, lw=0.7, zorder=3)
        )
        label(ax, bx0 + i * (bw + bgap) + 0.5 * bw, 0.077, str(bit), size=BODY, weight="bold", color=edge)
        label(ax, bx0 + i * (bw + bgap) + 0.5 * bw, 0.020, str(i), size=BODY, color=MID)
    label(ax, 0.340, 0.020, "layer index", size=BODY, ha="left", color=MID)

    # ---------------- (c) group-wise pack ----------------
    band_heading(ax, 0.520, 0.458, "c", "Group-wise pack")
    rule(ax, 0.520, 0.980, 0.428)

    label(ax, 0.640, 0.402, "64 values / group", size=BODY, weight="semibold")
    for i in range(8):
        ax.add_patch(
            Rectangle(
                (0.526 + i * 0.029, 0.328),
                0.023,
                0.044,
                edgecolor=TEAL,
                facecolor=TEAL_PALE if i % 2 == 0 else WHITE,
                lw=0.65,
                zorder=3,
            )
        )
    label(ax, 0.772, 0.350, "$\\ldots$", size=BODY, ha="left", color=TEAL)
    arrow(ax, 0.640, 0.322, 0.640, 0.298, color=ORANGE, lw=0.85)

    box(ax, 0.520, 0.120, 0.460, 0.176, edge=ORANGE, face=ORANGE_PALE, lw=0.9, radius=0.006)
    label(ax, 0.750, 0.263, "$m=\\min(x),\\; u=\\max(x)$", size=BODY, weight="semibold")
    label(ax, 0.750, 0.208, "$s=(u-m)/(2^b-1)$;  $s=1$ if $u=m$", size=BODY)
    label(ax, 0.750, 0.153, "Read: $\\hat{x}=qs+m$ in FP32", size=BODY)

    box(ax, 0.520, 0.030, 0.226, 0.062, edge=ORANGE, face=WHITE, lw=0.9, radius=0.006)
    label(ax, 0.633, 0.061, "packed Q2/Q4/Q8", size=BODY, weight="bold", color=ORANGE)
    box(ax, 0.754, 0.030, 0.226, 0.062, edge=INK, face=PALE, lw=0.8, radius=0.006)
    label(ax, 0.867, 0.061, "BF16 scale/bias", size=BODY, weight="bold")

    export(fig, "qcomem_quantization_map_r47")


def main() -> None:
    publication_style()
    mpl.rcParams["font.size"] = BODY
    draw_pipeline()
    draw_quantization_map()


if __name__ == "__main__":
    main()
