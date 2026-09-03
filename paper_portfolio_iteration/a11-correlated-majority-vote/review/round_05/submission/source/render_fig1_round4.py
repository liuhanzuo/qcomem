#!/usr/bin/env python3
"""Render Round-4 Figure 1 from hash-pinned frozen JSON inputs.

This is presentation code only. It makes no fit, selection, or scientific
recomputation. The resulting PDF/PNG is a code-native explanatory schematic.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_PDF = HERE / "fig1_round4.pdf"
OUT_PNG = HERE / "fig1_round4.png"

EXPECTED_INPUT_SHA256 = {
    HERE / "audit_evidence/results/fit_cal_test_r469_result.json": "b114c72d9ab1cf1a6ba1d2bd734433c06bd4d5cbd19bf93be0964edf6fc8a5f7",
    HERE / "audit_evidence/results/passrate_r467_result.json": "5c346416ab0d75d73d86cb9c0ec57a00d34e0d6a5e8a190929dd592b535bab93",
    HERE / "audit_evidence/results/margin_repair_r469_result.json": "c3edaa585a713948e8bcf94f15050027b4f4b87c1aa3828b1e3b30bbe04f16b0",
    ROOT / "evidence/repro_bundle_round4/recovered_outputs/drift_stress_r469_result.json": "1208deade2cb42a324bb948c93bf66ee72e68a93e78815aadcfbeb19880c3163",
}

# Keep presentation-only output reproducible across isolated package renders.
# The default is intentionally fixed; callers may supply the same value via
# SOURCE_DATE_EPOCH, which is recorded in the input manifest/provenance.
RENDER_EPOCH = int(os.environ.get("SOURCE_DATE_EPOCH", "1780000000"))

PALETTE = {
    "blue": "#0F4D92",
    "blue2": "#3775BA",
    "green": "#8BCF8B",
    "green_light": "#DDF3DE",
    "red": "#B64342",
    "red_light": "#F6CFCB",
    "orange": "#FFD700",
    "neutral": "#CFCECE",
    "dark": "#272727",
    "mid": "#4D4D4D",
    "pale": "#F6F7F8",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percent_half_up(value: float, digits: int = 1) -> str:
    """Format a fraction as a percent using the manuscript's decimal convention."""
    quantum = Decimal("1").scaleb(-digits)
    rounded = (Decimal(str(value)) * Decimal("100")).quantize(
        quantum, rounding=ROUND_HALF_UP
    )
    return f"{rounded:.{digits}f}%"


def render_metadata() -> tuple[dict[str, object], dict[str, str]]:
    stamp = datetime.fromtimestamp(RENDER_EPOCH, tz=timezone.utc)
    stamp_text = stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    pdf = {
        "Title": "A11 Figure 1: oracle, fitted score, and CAL",
        "Creator": "A11 deterministic Matplotlib Figure 1 renderer",
        "Producer": "A11 deterministic Matplotlib Figure 1 renderer",
        "CreationDate": stamp,
    }
    png = {
        "Software": "A11 deterministic Matplotlib Figure 1 renderer",
        "Creation Time": stamp_text,
    }
    return pdf, png


def read_frozen_inputs() -> tuple[dict, dict, dict, dict]:
    loaded: list[dict] = []
    for path, expected in EXPECTED_INPUT_SHA256.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"hash mismatch for {path}: expected {expected}, got {actual}")
        loaded.append(json.loads(path.read_text(encoding="utf-8")))
    return tuple(loaded)  # type: ignore[return-value]


def rounded_box(ax, x, y, w, h, *, fc, ec, lw=1.4, radius=0.025, z=1):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
    )
    ax.add_patch(box)
    return box


def label(ax, x, y, text, *, size=10, weight="normal", color=None,
          ha="left", va="center", z=5, style="normal"):
    ax.text(x, y, text, fontsize=size, fontweight=weight, color=color or PALETTE["dark"],
            ha=ha, va=va, zorder=z, style=style, linespacing=1.16)


def arrow(ax, start, end, *, color=None, lw=1.35, mutation=11, z=3):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=mutation,
                                linewidth=lw, color=color or PALETTE["dark"], zorder=z))


def panel_label(ax, x, text, *, size=10.6):
    """A compact panel title with a raw size that remains >=6.5pt in paper."""
    label(ax, x, 0.965, text, size=size, weight="bold", va="top")


def draw_panel_a(ax, n_rollouts: int, extreme_mass: float) -> None:
    panel_label(ax, 0.025, "(a)  Replay prefix (random; not chronology)")
    rounded_box(ax, 0.055, 0.775, 0.890, 0.105, fc=PALETTE["blue"], ec=PALETTE["blue"], lw=0)
    label(ax, 0.50, 0.827, "random prefix  •  sampled without replacement", size=8.8,
          weight="bold", color="white", ha="center")

    # An intentionally sparse U-shaped histogram using the frozen extreme-mass label.
    x0, y0, bw = 0.155, 0.555, 0.095
    heights = [0.145, 0.085, 0.040, 0.025, 0.040, 0.095, 0.150]
    for i, h in enumerate(heights):
        ax.add_patch(Rectangle((x0 + i * bw, y0), bw * 0.72, h,
                               facecolor=PALETTE["blue2"], edgecolor="none", alpha=0.86))
    ax.plot([x0 - 0.012, x0 + 7 * bw - 0.002], [y0, y0], color=PALETTE["mid"], lw=0.8)
    label(ax, 0.50, 0.713, f"{extreme_mass:.0%} of tasks at p≤.1 or p≥.9", size=8.35, ha="center")
    label(ax, 0.50, 0.520, "across-task count mixture", size=8.55, color=PALETTE["mid"], ha="center")
    arrow(ax, (0.50, 0.505), (0.50, 0.435), color=PALETTE["blue"])
    rounded_box(ax, 0.310, 0.355, 0.380, 0.090, fc=PALETTE["pale"], ec=PALETTE["blue"], lw=1.2)
    label(ax, 0.50, 0.400, "true law H⋆", size=8.85, weight="bold", color=PALETTE["blue"], ha="center")
    arrow(ax, (0.50, 0.345), (0.50, 0.285), color=PALETTE["blue"])

    # Dots deliberately show the random-prefix construction, not an observed timeline.
    n_drawn = 13
    xs = [0.185 + i * 0.053 for i in range(n_drawn)]
    for i, x in enumerate(xs):
        c = PALETTE["orange"] if i < 5 else PALETTE["neutral"]
        ax.add_patch(Circle((x, 0.235), 0.018, facecolor=c, edgecolor=PALETTE["mid"], linewidth=0.5))
    ax.plot([xs[0] - 0.022, xs[-1] + 0.022], [0.185, 0.185], color=PALETTE["mid"], lw=0.75)
    label(ax, 0.50, 0.125, f"prefix k; held remainder from N={n_rollouts} outcomes", size=8.25, ha="center")
    label(ax, 0.50, 0.050, "replay model only — no chronological-order claim", size=8.25,
          color=PALETTE["red"], ha="center", weight="bold")


def draw_panel_b(ax) -> None:
    panel_label(ax, 0.025, "(b)  Oracle H⋆ versus fitted score")
    # The two statements are stacked so that each text block keeps a readable
    # physical line length at single-column paper width.
    rounded_box(ax, 0.050, 0.585, 0.900, 0.220, fc="#EEF5FC", ec=PALETTE["blue"], lw=1.5)
    label(ax, 0.50, 0.743, "TRUE H⋆ — oracle theory", size=9.15, weight="bold", color=PALETTE["blue"], ha="center")
    label(ax, 0.50, 0.670, "c_H⋆(x,k) = P(flip | x,k)", size=8.45, weight="bold", ha="center")
    label(ax, 0.50, 0.613, "exact conditional • Theorem 1", size=8.25, ha="center")

    rounded_box(ax, 0.050, 0.365, 0.900, 0.165, fc="#FFF8E3", ec="#B07D00", lw=1.5)
    label(ax, 0.50, 0.478, "Ĥ_FIT — implemented fitted score", size=8.90,
          weight="bold", color="#7B5900", ha="center")
    label(ax, 0.50, 0.405, "s_ĤFIT(x,k), not c_H⋆", size=8.35, ha="center",
          color=PALETTE["red"], weight="bold")

    rounded_box(ax, 0.050, 0.135, 0.900, 0.170, fc=PALETTE["green_light"], ec="#4E8B4E", lw=1.4)
    label(ax, 0.50, 0.245, "FIT freeze → CAL EB UCB (Bonferroni)", size=8.55,
          weight="bold", color="#356C35", ha="center")
    label(ax, 0.50, 0.175, "Theorem 2: marginal replay guarantee", size=8.35,
          ha="center", weight="bold")
    label(ax, 0.50, 0.052, "TEST: descriptive; no fitted conditional certificate", size=8.25,
          color=PALETTE["red"], ha="center", weight="bold")


def draw_panel_c(ax, readout: dict, margin: dict, drift: dict) -> None:
    panel_label(ax, 0.020, "(c)  Replay quantities + synthetic exploratory drift", size=10.55)
    label(ax, 0.020, 0.822, "OMR α=.05 descriptive TEST replay-count reduction", size=8.35, color=PALETTE["mid"])
    methods = [
        ("FIXED-HOEF", readout["FIXED_HOEF_a0.05"]["saving_vs_full"], PALETTE["red_light"], PALETTE["red"]),
        ("FIXED-EB", readout["FIXED_EB_a0.05"]["saving_vs_full"], "#BFD7EE", PALETTE["blue2"]),
        ("BAYES-H", readout["BAYESH_a0.05"]["saving_vs_full"], PALETTE["green"], "#356C35"),
    ]
    yvals = [0.700, 0.580, 0.460]
    for (name, value, fill, edge), y in zip(methods, yvals):
        rounded_box(ax, 0.225, y - 0.031, 0.535, 0.062, fc="#F4F4F4", ec="#F4F4F4", lw=0)
        rounded_box(ax, 0.225, y - 0.031, 0.535 * value, 0.062, fc=fill, ec=edge, lw=1.0, radius=0.012)
        label(ax, 0.205, y, name, size=8.75, weight="bold", ha="right")
        suffix = percent_half_up(value)
        if name == "BAYES-H":
            suffix += f"  |  flip {readout['BAYESH_a0.05']['realized_flip']:.4f}"
        label(ax, min(0.845, 0.238 + 0.535 * value), y, suffix, size=8.65,
              weight="bold", ha="left")
    label(ax, 0.50, 0.365, "count-replay endpoint; not correctness or operational cost", size=8.35,
          color=PALETTE["mid"], ha="center")

    e3 = drift["results"]["E3_blockswap"]["0.05"]["0.15"]
    e3_margin = margin["results"]["a0.05_E3_blockswap_d0.15_g0.025"]
    rounded_box(ax, 0.030, 0.105, 0.940, 0.225, fc="#FFF6F4", ec=PALETTE["red"], lw=1.1)
    label(ax, 0.050, 0.278, "Synthetic / exploratory / nonconfirmatory drift diagnostic", size=8.75,
          color=PALETTE["red"], weight="bold")
    label(ax, 0.050, 0.202,
          f"E3, α=.05, δ=.15: BAYES-H {percent_half_up(e3['BAYESH']['flip'])} invalid; "
          f"FIXED-EB {percent_half_up(e3['FIXED_EB']['flip'])} valid.", size=8.45)
    label(ax, 0.050, 0.145,
          f"Explored margin γ=.025: flip {percent_half_up(e3_margin['flip'])}, reduction {percent_half_up(e3_margin['saving'])}; exploratory only.",
          size=8.30)
    label(ax, 0.50, 0.050, "Frozen JSON values; no natural-shift or online-policy claim", size=8.25,
          color=PALETTE["mid"], ha="center")


def main() -> None:
    fit_cal, passrate, margin, drift = read_frozen_inputs()
    readout = fit_cal["test_readout"]

    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    # At \textwidth=6.5in this 7.8in canvas is scaled by about 0.83.  The
    # smallest 8.25pt panel label therefore remains about 6.9pt at 100% page
    # render, rather than relying on zoomed inspection.
    fig = plt.figure(figsize=(7.8, 4.05), facecolor="white")
    axes = [
        fig.add_axes([0.045, 0.565, 0.430, 0.370]),
        fig.add_axes([0.525, 0.565, 0.430, 0.370]),
        fig.add_axes([0.045, 0.070, 0.910, 0.370]),
    ]
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
    fig.add_artist(plt.Line2D([0.045, 0.955], [0.505, 0.505], color="#D8D8D8", linewidth=1.0,
                              transform=fig.transFigure))
    draw_panel_a(axes[0], int(fit_cal["N"]), float(passrate["frac_extreme_p_le0.1_or_ge0.9"]))
    draw_panel_b(axes[1])
    draw_panel_c(axes[2], readout, margin, drift)
    pdf_metadata, png_metadata = render_metadata()
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.035, metadata=pdf_metadata)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.035, metadata=png_metadata)
    plt.close(fig)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
