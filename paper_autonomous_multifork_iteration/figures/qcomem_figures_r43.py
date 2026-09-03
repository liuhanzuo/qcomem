#!/usr/bin/env python3
"""Generate the R43 Q-CoMem mechanism and quantization figures.

The visual grammar follows the local figures4papers scientific-figure-making
skill: minimalist Matplotlib composition, semantic color, redundant encodings,
embedded/editable vector text, and deterministic PDF/SVG/PNG export.

Neither figure is an empirical result.  Figure 1 describes the Write/Read
protocol and its ownership boundary.  Figure 2 makes the quantization scope
explicit: retained document-state sidecars are packed, while model weights and
forward computation remain BF16.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parent

# figures4papers-compatible semantic palette.
INK = "#20262D"
MID = "#66717D"
LIGHT = "#D7DDE3"
PALE = "#F6F8FA"
WHITE = "#FFFFFF"
BLUE = "#0F4D92"
BLUE_2 = "#3775BA"
BLUE_PALE = "#E9F1F8"
GREEN = "#2F7D32"
GREEN_PALE = "#EAF4E8"
TEAL = "#42949E"
TEAL_PALE = "#E7F3F4"
ORANGE = "#D9690C"
ORANGE_PALE = "#FFF1E6"
VIOLET = "#8A4D86"
VIOLET_PALE = "#F4EBF3"


def publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica",
                "Arial",
                "Liberation Sans",
                "DejaVu Sans",
            ],
            "font.size": 7.0,
            "font.weight": "regular",
            "text.color": INK,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "qcomem-r43-figures4papers",
            "lines.solid_capstyle": "butt",
            "savefig.facecolor": WHITE,
            "savefig.edgecolor": WHITE,
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


def box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    edge: str = INK,
    face: str = WHITE,
    lw: float = 0.9,
    radius: float = 0.010,
    hatch: str | None = None,
    zorder: int = 2,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.003,rounding_size={radius}",
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
    size: float = 6.5,
    weight: str = "regular",
    ha: str = "center",
    va: str = "center",
    color: str = INK,
    rotation: float = 0,
    linespacing: float = 1.12,
    zorder: int = 6,
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
    color: str = INK,
    lw: float = 0.9,
    linestyle: str = "-",
    mutation_scale: float = 7.5,
    zorder: int = 4,
):
    patch = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
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


def elbow_arrow(
    ax,
    points: list[tuple[float, float]],
    *,
    color: str = INK,
    lw: float = 0.85,
    linestyle: str = "-",
    zorder: int = 3,
):
    for (x0, y0), (x1, y1) in zip(points[:-2], points[1:-1]):
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=color,
            lw=lw,
            ls=linestyle,
            zorder=zorder,
        )
    (x0, y0), (x1, y1) = points[-2], points[-1]
    arrow(
        ax,
        x0,
        y0,
        x1,
        y1,
        color=color,
        lw=lw,
        linestyle=linestyle,
        zorder=zorder,
    )


def panel_heading(ax, x0: float, x1: float, text: str, index: str) -> None:
    label(ax, x0, 0.949, index, size=8.8, weight="bold", ha="left", color=BLUE)
    label(ax, x0 + 0.028, 0.949, text, size=8.8, weight="semibold", ha="left")
    ax.plot([x0, x1], [0.910, 0.910], color=LIGHT, lw=0.75, zorder=1)


def draw_document(ax, x: float, y: float, w: float, h: float) -> None:
    for dx, dy in [(0.010, 0.020), (0.005, 0.010), (0.000, 0.000)]:
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
    for frac in (0.69, 0.51, 0.33):
        ax.plot(
            [x + 0.010, x + 0.78 * w],
            [y + frac * h, y + frac * h],
            color=BLUE_2,
            lw=0.65,
            zorder=5,
        )


def draw_layer_stack(ax, x: float, y: float, w: float, h: float) -> None:
    colors = [(GREEN, GREEN_PALE), (BLUE, BLUE_PALE), (GREEN, GREEN_PALE)]
    offsets = [(0.000, 0.000), (0.006, 0.014), (0.012, 0.028)]
    for (edge, face), (dx, dy) in zip(colors, offsets):
        box(
            ax,
            x + dx,
            y + dy,
            w,
            h,
            edge=edge,
            face=face,
            lw=0.9,
            radius=0.006,
            zorder=2,
        )
    label(ax, x + 0.5 * w + 0.006, y + 0.54 * h + 0.014, "lower hybrid\nlayers $[0,j)$", size=6.2, weight="semibold")


def draw_state_leaf(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    edge: str,
    face: str,
) -> None:
    box(ax, x, y, w, h, edge=edge, face=face, lw=0.9, radius=0.006)
    ax.plot([x + 0.010, x + 0.010], [y + 0.012, y + h - 0.012], color=edge, lw=2.0, zorder=5)
    label(ax, x + 0.019, y + 0.5 * h, text, size=6.0, ha="left", weight="semibold")


def draw_pack(ax, x: float, y: float, w: float, h: float) -> None:
    box(ax, x, y, w, h, edge=ORANGE, face=ORANGE_PALE, lw=1.15, radius=0.010)
    label(ax, x + 0.5 * w, y + 0.76 * h, "Groupwise", size=5.8, weight="semibold", color=ORANGE)
    label(ax, x + 0.5 * w, y + 0.58 * h, "pack", size=7.0, weight="bold")
    n = 6
    cell_w = 0.62 * w / n
    start = x + 0.19 * w
    for i in range(n):
        ax.add_patch(
            Rectangle(
                (start + i * cell_w, y + 0.22 * h),
                0.78 * cell_w,
                0.13 * h,
                linewidth=0.55,
                edgecolor=ORANGE,
                facecolor=WHITE if i % 2 else "#FFD9B8",
                zorder=5,
            )
        )


def draw_store(ax, x: float, y: float, w: float, h: float) -> None:
    # A three-band cylinder echoes the protocol-geometry state glyphs.
    ax.add_patch(
        Rectangle(
            (x, y + 0.12 * h),
            w,
            0.76 * h,
            linewidth=1.0,
            edgecolor=GREEN,
            facecolor=WHITE,
            zorder=2,
        )
    )
    for yy, face in [
        (y + 0.62 * h, TEAL_PALE),
        (y + 0.39 * h, BLUE_PALE),
        (y + 0.16 * h, GREEN_PALE),
    ]:
        ax.add_patch(
            Rectangle(
                (x + 0.003, yy),
                w - 0.006,
                0.22 * h,
                edgecolor="none",
                facecolor=face,
                zorder=3,
            )
        )
    ax.add_patch(Ellipse((x + 0.5 * w, y + 0.88 * h), w, 0.24 * h, facecolor=WHITE, edgecolor=GREEN, lw=1.0, zorder=4))
    ax.add_patch(Ellipse((x + 0.5 * w, y + 0.12 * h), w, 0.24 * h, facecolor=GREEN_PALE, edgecolor=GREEN, lw=1.0, zorder=4))
    ax.plot([x, x], [y + 0.12 * h, y + 0.88 * h], color=GREEN, lw=1.0, zorder=5)
    ax.plot([x + w, x + w], [y + 0.12 * h, y + 0.88 * h], color=GREEN, lw=1.0, zorder=5)
    label(ax, x + 0.5 * w, y - 0.045, "Q-CoMem\nStore", size=6.0, weight="semibold", va="top")


def draw_pipeline() -> None:
    # The wide source canvas preserves horizontal room for protocol labels;
    # final-size typography is validated after LaTeX inclusion at 5.5 inches.
    fig, ax = canvas(7.35, 3.30)
    panel_heading(ax, 0.018, 0.305, "Write once", "1")
    panel_heading(ax, 0.320, 0.455, "Retain", "2")
    panel_heading(ax, 0.470, 0.865, "Read per query", "3")
    panel_heading(ax, 0.880, 0.990, "Audit rail", "4")
    for x in (0.312, 0.462, 0.872):
        ax.plot([x, x], [0.175, 0.895], color=LIGHT, lw=0.65, zorder=1)

    # 1. Offline Write.
    draw_document(ax, 0.030, 0.572, 0.040, 0.145)
    label(ax, 0.052, 0.512, "Document $D$", size=6.0, weight="semibold")
    arrow(ax, 0.081, 0.645, 0.103, 0.645, color=BLUE, lw=1.05)
    draw_layer_stack(ax, 0.105, 0.565, 0.072, 0.145)
    arrow(ax, 0.190, 0.645, 0.207, 0.645, color=BLUE, lw=1.05)
    label(ax, 0.255, 0.782, "complete split state", size=5.8, color=MID)
    draw_state_leaf(ax, 0.205, 0.690, 0.099, 0.060, r"residual  $h_j^D$", edge=TEAL, face=TEAL_PALE)
    draw_state_leaf(ax, 0.205, 0.608, 0.099, 0.060, "attention KV", edge=BLUE, face=BLUE_PALE)
    draw_state_leaf(ax, 0.205, 0.526, 0.099, 0.060, "GDN state", edge=GREEN, face=GREEN_PALE)

    # 2. Retained entry.
    for yy, color in [(0.720, TEAL), (0.638, BLUE), (0.556, GREEN)]:
        elbow_arrow(ax, [(0.304, yy), (0.316, yy), (0.316, 0.640), (0.327, 0.640)], color=color, lw=0.75)
    draw_pack(ax, 0.329, 0.535, 0.064, 0.205)
    arrow(ax, 0.395, 0.638, 0.412, 0.638, color=ORANGE, lw=1.0)
    draw_store(ax, 0.414, 0.545, 0.039, 0.190)

    # 3. Online Read.  Residual bypass and lower-layer replay remain separate
    # until their ordered merge at depth j.
    arrow(ax, 0.454, 0.640, 0.477, 0.640, color=GREEN, lw=1.05)
    box(ax, 0.479, 0.559, 0.067, 0.165, edge=ORANGE, face=ORANGE_PALE, lw=1.0, radius=0.008)
    label(ax, 0.5125, 0.660, "Fetch +", size=6.0, weight="semibold", color=ORANGE)
    label(ax, 0.5125, 0.622, "dequantize", size=6.2, weight="semibold")
    label(ax, 0.5125, 0.580, "to BF16", size=5.8, color=MID)

    # Residual bypass.
    elbow_arrow(ax, [(0.546, 0.665), (0.560, 0.665), (0.560, 0.754), (0.578, 0.754)], color=TEAL, lw=0.9)
    box(ax, 0.573, 0.719, 0.085, 0.070, edge=TEAL, face=TEAL_PALE, lw=0.9, radius=0.020)
    label(ax, 0.6155, 0.754, r"document $h_j^D$", size=6.0, weight="semibold", color=TEAL)
    label(ax, 0.6155, 0.700, "bypasses replay", size=5.8, color=TEAL)
    elbow_arrow(ax, [(0.658, 0.754), (0.782, 0.754), (0.782, 0.655)], color=TEAL, lw=0.85)

    # Ownership boundary for lower cache leaves.
    arrow(ax, 0.546, 0.610, 0.568, 0.610, color=BLUE, lw=0.85)
    box(ax, 0.570, 0.543, 0.091, 0.115, edge=BLUE, face=BLUE_PALE, lw=0.9, radius=0.007)
    label(ax, 0.6155, 0.615, "immutable", size=6.0, weight="semibold", color=BLUE)
    label(ax, 0.6155, 0.578, "document view", size=6.0)
    arrow(ax, 0.6155, 0.539, 0.6155, 0.493, color=ORANGE, lw=0.85, linestyle="--")
    label(ax, 0.629, 0.516, "COW / rebind", size=5.8, color=ORANGE, ha="left")
    box(ax, 0.570, 0.370, 0.091, 0.115, edge=GREEN, face=GREEN_PALE, lw=0.9, radius=0.007)
    ax.plot([0.579, 0.579], [0.386, 0.469], color=GREEN, lw=2.0, zorder=5)
    label(ax, 0.620, 0.441, "request-local", size=6.0, weight="semibold", color=GREEN)
    label(ax, 0.620, 0.404, "mutable state", size=6.0)

    # Query replay.
    ax.add_patch(Ellipse((0.505, 0.280), 0.047, 0.090, facecolor=WHITE, edgecolor=BLUE, lw=1.0, zorder=3))
    label(ax, 0.505, 0.280, "$q$", size=7.0, weight="bold", color=BLUE)
    label(ax, 0.505, 0.218, "query", size=5.8, color=MID)
    elbow_arrow(ax, [(0.529, 0.280), (0.667, 0.280), (0.667, 0.344), (0.680, 0.344)], color=BLUE, lw=0.9)
    arrow(ax, 0.662, 0.425, 0.680, 0.425, color=GREEN, lw=1.0)
    draw_layer_stack(ax, 0.682, 0.326, 0.066, 0.132)
    label(ax, 0.715, 0.294, "query replay", size=5.8, color=MID)
    elbow_arrow(ax, [(0.761, 0.392), (0.774, 0.392), (0.774, 0.526)], color=BLUE, lw=0.9)
    label(ax, 0.768, 0.504, r"$h_j^q$", size=6.0, color=BLUE, ha="right")

    # Ordered split merge, suffix reconstruction, and decode.
    box(ax, 0.758, 0.528, 0.050, 0.128, edge=TEAL, face=WHITE, lw=1.0, radius=0.006)
    label(ax, 0.783, 0.608, r"$[h_j^D;$", size=6.0, weight="semibold")
    label(ax, 0.783, 0.568, r"$h_j^q]$", size=6.0, weight="semibold")
    label(ax, 0.783, 0.500, "merge at $j$", size=5.8, color=MID)
    arrow(ax, 0.808, 0.592, 0.815, 0.592, color=TEAL, lw=1.0)
    box(ax, 0.817, 0.526, 0.037, 0.132, edge=BLUE, face=WHITE, lw=0.95, radius=0.006)
    for yy, edge, face in [(0.612, GREEN, GREEN_PALE), (0.574, BLUE, BLUE_PALE), (0.536, GREEN, GREEN_PALE)]:
        ax.add_patch(Rectangle((0.823, yy), 0.025, 0.027, edgecolor=edge, facecolor=face, lw=0.6, zorder=4))
    label(ax, 0.8355, 0.685, "suffix", size=5.8, color=MID)
    label(ax, 0.8355, 0.472, r"$[j,L)$", size=5.8, color=MID)
    arrow(ax, 0.855, 0.592, 0.864, 0.592, color=BLUE, lw=1.0)

    # 4. Narrow, subordinate audit rail in the visual language of the appendix
    # protocol-geometry figure.
    rail_x = 0.974
    ax.plot([rail_x, rail_x], [0.300, 0.785], color=INK, lw=0.95, zorder=2)
    ys = [0.745, 0.575, 0.405]
    names = ["ownership", "immutability", "COW / rebind"]
    markers = ["o", "D", "^"]
    colors = [BLUE, TEAL, GREEN]
    for yy, name, marker, color in zip(ys, names, markers, colors):
        ax.scatter([rail_x], [yy], s=55, marker=marker, facecolors=WHITE, edgecolors=color, linewidths=1.05, zorder=5)
        label(ax, 0.955, yy, name, size=6.2, ha="right", color=INK)
    ax.plot([0.928, rail_x - 0.008], [0.245, 0.300], color=MID, lw=0.7, ls=(0, (1.4, 2.0)), zorder=2)
    label(ax, 0.949, 0.232, "ForkAudit", size=6.2, weight="semibold", color=VIOLET)
    label(ax, 0.949, 0.197, "validation rail", size=5.8, color=MID)

    # Capacity-first trade-off statement; no latency-speedup claim.
    ax.plot([0.045, 0.860], [0.115, 0.115], color=LIGHT, lw=0.75)
    arrow(ax, 0.455, 0.130, 0.245, 0.130, color=GREEN, lw=0.85)
    arrow(ax, 0.455, 0.130, 0.665, 0.130, color=ORANGE, lw=0.85)
    label(ax, 0.145, 0.080, "smaller retained entry / document", size=6.0, weight="semibold", color=GREEN)
    label(ax, 0.455, 0.165, "retention–online-work trade-off", size=5.8, weight="semibold", color=MID)
    label(ax, 0.755, 0.080, "more dequantization + replay work", size=6.0, weight="semibold", color=ORANGE)

    export(fig, "qcomem_pipeline_r43")


def layer_box(ax, x: float, y: float, w: float, h: float, index: int, kind: str) -> None:
    edge, face = (BLUE, BLUE_PALE) if kind == "Attn" else (GREEN, GREEN_PALE)
    box(ax, x, y, w, h, edge=edge, face=face, lw=0.9, radius=0.004)
    label(ax, x + 0.5 * w, y + 0.67 * h, str(index), size=6.5, weight="bold", color=edge)
    label(ax, x + 0.5 * w, y + 0.29 * h, kind, size=5.8, weight="semibold")
    # A small sidecar tab encodes retained state without coloring the BF16 layer
    # itself as quantized.
    ax.add_patch(Rectangle((x + 0.18 * w, y - 0.055), 0.64 * w, 0.026, edgecolor=edge, facecolor=WHITE, lw=0.7, zorder=4))


def q_badge(ax, x: float, y: float, text: str) -> None:
    box(ax, x, y, 0.043, 0.068, edge=ORANGE, face=ORANGE_PALE, lw=1.0, radius=0.018)
    label(ax, x + 0.0215, y + 0.034, text, size=6.5, weight="bold", color=ORANGE)


def draw_quantization_map() -> None:
    fig, ax = canvas(7.35, 2.20)
    panel_heading(ax, 0.018, 0.450, "Hybrid backbone and split", "a")
    panel_heading(ax, 0.470, 0.735, "Frozen policy", "b")
    panel_heading(ax, 0.755, 0.990, "Groupwise pack", "c")
    for x in (0.460, 0.745):
        ax.plot([x, x], [0.145, 0.895], color=LIGHT, lw=0.65, zorder=1)

    # (a) Exact Qwen3.5 lower-prefix geometry at j=7.  All layer boxes denote
    # BF16 weights/compute; the narrow tabs beneath them denote retained state.
    x0, gap, w, y, h = 0.022, 0.006, 0.043, 0.505, 0.190
    kinds = ["GDN", "GDN", "GDN", "Attn", "GDN", "GDN", "GDN"]
    for i, kind in enumerate(kinds):
        layer_box(ax, x0 + i * (w + gap), y, w, h, i, kind)
    label(ax, 0.187, 0.748, "lower prefix $[0,7)$", size=6.2, weight="semibold")
    label(ax, 0.187, 0.415, "retained-state sidecars", size=5.8, color=MID)
    split_x = x0 + 7 * (w + gap) + 0.002
    ax.plot([split_x, split_x], [0.390, 0.760], color=ORANGE, lw=1.25, zorder=3)
    ax.scatter([split_x], [0.575], s=32, color=ORANGE, zorder=5)
    label(ax, split_x, 0.802, "$j=7$", size=6.5, weight="bold", color=ORANGE)
    ax.plot([split_x + 0.003, 0.377], [0.600, 0.600], color=BLUE, lw=0.8, zorder=3)
    box(ax, 0.374, 0.480, 0.073, 0.235, edge=INK, face=PALE, lw=0.9, radius=0.006)
    label(ax, 0.4105, 0.661, "L7–L39", size=6.0, weight="bold")
    label(ax, 0.4105, 0.615, "33-layer", size=5.6)
    label(ax, 0.4105, 0.574, "online suffix", size=5.4)
    label(ax, 0.4105, 0.526, "24 GDN", size=5.2, color=MID)
    label(ax, 0.4105, 0.493, "+ 9 Attn", size=5.2, color=MID)
    label(ax, 0.232, 0.292, "All model weights and forward compute remain BF16", size=6.2, weight="semibold", color=BLUE)
    label(ax, 0.232, 0.232, "No upper-suffix document state is retained", size=5.8, color=MID)

    # (b) The frozen state-type policy used by the principal Q4/Q4/Q8 point.
    rows = [
        (0.680, r"boundary residual  $h_7^D$", TEAL, TEAL_PALE, "Q4"),
        (0.535, "attention KV  (L3)", BLUE, BLUE_PALE, "Q4"),
        (0.390, "GDN state  (L0–2, L4–6)", GREEN, GREEN_PALE, "Q8"),
    ]
    for yy, text, edge, face, qtext in rows:
        box(ax, 0.480, yy - 0.041, 0.176, 0.082, edge=edge, face=face, lw=0.9, radius=0.006)
        ax.plot([0.491, 0.491], [yy - 0.025, yy + 0.025], color=edge, lw=2.0, zorder=5)
        label(ax, 0.502, yy, text, size=5.8, ha="left", weight="semibold")
        arrow(ax, 0.658, yy, 0.675, yy, color=edge, lw=0.8)
        q_badge(ax, 0.678, yy - 0.034, qtext)
    label(ax, 0.602, 0.276, "lower-layer cache bits", size=5.8, color=MID)
    bits = [8, 8, 8, 4, 8, 8, 8]
    bx0, bw, bgap = 0.499, 0.024, 0.005
    for i, bit in enumerate(bits):
        edge = BLUE if i == 3 else GREEN
        face = BLUE_PALE if i == 3 else GREEN_PALE
        ax.add_patch(Rectangle((bx0 + i * (bw + bgap), 0.188), bw, 0.052, edgecolor=edge, facecolor=face, lw=0.7, zorder=3))
        label(ax, bx0 + i * (bw + bgap) + 0.5 * bw, 0.214, str(bit), size=5.8, weight="bold", color=edge)
        label(ax, bx0 + i * (bw + bgap) + 0.5 * bw, 0.164, str(i), size=5.6, color=MID)
    label(ax, 0.602, 0.130, "layer index", size=5.6, color=MID)

    # (c) A compact but exact affine-packing schematic.
    label(ax, 0.872, 0.795, "64 values / group", size=6.2, weight="semibold")
    cx0, cy, cw, ch = 0.772, 0.690, 0.020, 0.055
    for i in range(8):
        ax.add_patch(
            Rectangle(
                (cx0 + i * (cw + 0.004), cy),
                cw,
                ch,
                edgecolor=TEAL,
                facecolor=TEAL_PALE if i % 2 == 0 else WHITE,
                lw=0.65,
                zorder=3,
            )
        )
    label(ax, 0.971, cy + 0.5 * ch, "…", size=8.0, color=TEAL)
    arrow(ax, 0.872, 0.673, 0.872, 0.606, color=ORANGE, lw=0.85)
    box(ax, 0.779, 0.500, 0.186, 0.100, edge=ORANGE, face=ORANGE_PALE, lw=0.9, radius=0.006)
    label(ax, 0.872, 0.567, r"$m=\min(x),\quad u=\max(x)$", size=6.0, weight="semibold")
    label(ax, 0.872, 0.526, r"$s=(u-m)/(2^b-1)$", size=6.0)
    arrow(ax, 0.872, 0.496, 0.872, 0.432, color=ORANGE, lw=0.85)
    box(ax, 0.765, 0.322, 0.128, 0.104, edge=ORANGE, face=WHITE, lw=0.9, radius=0.006)
    label(ax, 0.829, 0.385, "packed unsigned", size=5.8, weight="semibold", color=ORANGE)
    label(ax, 0.829, 0.347, "Q4 / Q8 codes", size=5.8, weight="bold")
    box(ax, 0.904, 0.322, 0.083, 0.104, edge=INK, face=PALE, lw=0.8, radius=0.006)
    label(ax, 0.9455, 0.385, "BF16", size=5.8, weight="bold")
    label(ax, 0.9455, 0.347, "scale + bias", size=5.5)
    # Both packed codes and affine metadata feed the Read-side dequantizer.
    ax.plot([0.829, 0.829], [0.319, 0.285], color=TEAL, lw=0.8, ls="--", zorder=3)
    ax.plot([0.9455, 0.9455], [0.319, 0.285], color=TEAL, lw=0.8, ls="--", zorder=3)
    ax.plot([0.829, 0.9455], [0.285, 0.285], color=TEAL, lw=0.8, ls="--", zorder=3)
    arrow(ax, 0.887, 0.285, 0.887, 0.238, color=TEAL, lw=0.8, linestyle="--")
    label(ax, 0.877, 0.198, "Read: dequantize to BF16 request state", size=5.8, weight="semibold", color=TEAL)

    # The figure's key semantic boundary.
    ax.plot([0.020, 0.985], [0.083, 0.083], color=LIGHT, lw=0.75)
    label(
        ax,
        0.502,
        0.042,
        "Packed scope: retained per-document tensor state only — not model weights, total VRAM, or Python/shape metadata",
        size=5.8,
        weight="semibold",
        color=INK,
    )

    export(fig, "qcomem_quantization_map_r43")


def export(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    common = dict(bbox_inches="tight", pad_inches=0.025, facecolor=WHITE)
    fig.savefig(
        OUT / f"{stem}.pdf",
        metadata={
            "Title": stem,
            "Author": "Anonymous",
            "Creator": "Q-CoMem figures4papers-style deterministic renderer",
            "CreationDate": None,
            "ModDate": None,
        },
        **common,
    )
    fig.savefig(OUT / f"{stem}.svg", metadata={"Title": stem, "Date": None}, **common)
    fig.savefig(
        OUT / f"{stem}.png",
        dpi=600,
        metadata={"Software": "Q-CoMem deterministic Matplotlib renderer"},
        **common,
    )
    plt.close(fig)


def main() -> None:
    publication_style()
    draw_pipeline()
    draw_quantization_map()


if __name__ == "__main__":
    main()
