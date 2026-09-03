#!/usr/bin/env python3
"""Render paper artifacts from the validated RR2 ForkAudit aggregate."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "evidence" / "rr2_formal_w" / "forkaudit-summary.json"
DEFAULT_MUTANTS = ROOT / "evidence" / "rr2_formal_w" / "mutant-outcomes.json"
DEFAULT_CENSUS = ROOT / "evidence" / "r39_independent_slot_census" / "r39-independent-slot-census-trial1907355-20260826a" / "audit" / "formal-aggregate.json"
DEFAULT_DUAL = ROOT / "evidence" / "r39_dual_producer_repeat" / "formal_h20" / "r39-dual-producer-repeat-20260826a-formal-complete.tar.gz"
DEFAULT_FALCON = ROOT / "evidence" / "r39_falcon_h1_transfer_v2" / "formal_h20" / "20260827a" / "validation.json"
GIB = 2**30

BLUE = "#4C78A8"
LIGHT_BLUE = "#DCEAF7"
ORANGE = "#F58518"
LIGHT_ORANGE = "#FCE3CE"
GREEN = "#2F855A"
LIGHT_GREEN = "#D9F0E3"
INK = "#1F2937"
GRAY = "#6B7280"
PDF_METADATA = {
    "Title": "ForkAudit RR2 paper artifact",
    "Author": "Anonymous",
    "Creator": "generate_rr2_paper_artifacts.py",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_dual_summary(path: Path):
    member = "r39-dual-producer-repeat-20260826a/audit/dual-producer-summary.json"
    with tarfile.open(path, "r:gz") as archive:
        handle = archive.extractfile(member)
        if handle is None:
            raise AssertionError(f"missing {member}")
        return json.loads(handle.read().decode("utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def policy_label(value: str) -> str:
    labels = {
        "vllm-q16-fresh-full-copy-control": "Full-copy KV",
        "vllm-q16-shared-document-reuse": "Shared-doc KV",
        "materialize-request-base-functional-rebind": "Materialized GDN base",
        "borrow-immutable-base-functional-rebind": "Borrowed GDN base",
    }
    return labels[value]


def validate(summary: dict, mutants: dict) -> None:
    require(summary["passed"] is True, "RR2 aggregate did not pass")
    require(summary["formal_ready"] is True, "RR2 aggregate is not formal-ready")
    require(summary["scientific_run_valid"] is True, "RR2 scientific run invalid")
    require(summary["scientific_outcome"] == "valid_positive", "RR2 outcome is not positive")
    require(summary["factorial_four_cell_exact"] is True, "factorial exactness failed")
    require(summary["oracle_all_ranks_passed"] is True, "an oracle row failed")
    require(summary["rank_count"] == 8, "RR2 rank count drift")
    require(summary["mutant_campaign"]["passed"] is True, "mutant campaign failed")
    require(len(summary["memory_matrix"]["cells"]) == 12, "memory matrix must have 12 cells")
    rows = mutants["rows"]
    require([row["mutant_id"] for row in rows] == [f"M{i}" for i in range(1, 10)], "mutant IDs drift")
    for row in rows:
        require(row["matched_clean_classification"] == "clean_pass", "matched clean failed")
        require(row["mutant_classification"] == "detected_expected_gate", "mutant escaped")
        require(row["expected_gate_id"] == row["observed_gate_id"], "wrong mutant gate")
        require(row["restoration_verified"] is True, "mutant restoration failed")


def draw_box(ax, x, y, width, height, text, facecolor, edgecolor, fontsize=8.2):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.1,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize, color=INK)


def render_factorial_figure(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Fixed 2×2 ownership factorial and phase boundary", fontsize=11, weight="bold", color=INK, pad=10)

    columns = [(0.23, "Full-copy KV", "private document copy"),
               (0.62, "Shared-document KV", "read-only shared document")]
    rows = [(0.53, "Materialized GDN", "private setup base"),
            (0.13, "Borrowed GDN", "read-only base alias")]
    width, height = 0.34, 0.30

    for x, title, _ in columns:
        ax.text(x + width / 2, 0.91, title, ha="center", va="center", fontsize=10, weight="bold", color=INK)
    for y, title, _ in rows:
        ax.text(0.19, y + height / 2, title, ha="right", va="center", fontsize=8.7, weight="bold", color=INK)

    for y, _, gdn_setup in rows:
        for x, _, kv_setup in columns:
            patch = FancyBboxPatch(
                (x, y), width, height,
                boxstyle="round,pad=0.018,rounding_size=0.02",
                facecolor="#FAFAFA", edgecolor="#CBD5E1", linewidth=1.0,
            )
            ax.add_patch(patch)
            ax.text(x + 0.018, y + 0.235, "KV setup", fontsize=7.2, weight="bold", color=BLUE, ha="left")
            ax.text(x + 0.105, y + 0.235, kv_setup, fontsize=7.4, color=INK, ha="left")
            ax.text(x + 0.018, y + 0.165, "GDN setup", fontsize=7.2, weight="bold", color=ORANGE, ha="left")
            ax.text(x + 0.105, y + 0.165, gdn_setup, fontsize=7.4, color=INK, ha="left")
            ax.annotate("first transition", xy=(x + 0.17, y + 0.085), xytext=(x + 0.17, y + 0.125),
                        ha="center", va="center", fontsize=6.7, color=GRAY,
                        arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 0.9})
            ax.text(x + width / 2, y + 0.035, "60 private mutable tensors/request",
                    fontsize=7.3, weight="bold", color=GREEN, ha="center", va="center")
    ax.text(0.60, 0.025, "All four cells: canonical byte-digest-equivalent outputs/state; base and peer ranges disjoint after transition",
            ha="center", va="bottom", fontsize=7.3, color=GRAY)
    fig.tight_layout(pad=0.6)
    fig.savefig(output, bbox_inches="tight", metadata=PDF_METADATA)
    plt.close(fig)


def _architecture_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    *,
    edgecolor: str = INK,
    facecolor: str = "white",
    fontsize: float = 6.4,
    linestyle: str = "-",
    linewidth: float = 0.9,
) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            linestyle=linestyle,
        )
    )
    if label:
        ax.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=INK,
        )


def _architecture_arrow(ax, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 0.8, "shrinkA": 0, "shrinkB": 0},
    )


def _architecture_bracket(ax, x: float, y0: float, y1: float, direction: float = 1.0) -> None:
    tick = 0.008 * direction
    ax.plot([x, x], [y0, y1], color=INK, lw=0.8)
    ax.plot([x, x + tick], [y0, y0], color=INK, lw=0.8)
    ax.plot([x, x + tick], [y1, y1], color=INK, lw=0.8)


def render_architecture_figure(output: Path) -> None:
    """Render the evidence-neutral ForkAudit lifecycle as an editable vector."""

    fig, ax = plt.subplots(figsize=(7.0, 2.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Orthogonal setup-policy axes. These choices apply to the entire request group.
    ax.text(0.015, 0.895, "KV setup", fontsize=8.0, weight="bold", color=INK, ha="left")
    _architecture_bracket(ax, 0.018, 0.695, 0.845)
    ax.text(0.032, 0.807, "full copy", fontsize=7.2, color=INK, ha="left", va="center")
    ax.text(0.032, 0.733, "shared paged", fontsize=7.2, color=INK, ha="left", va="center")
    ax.text(0.072, 0.575, r"$\times$", fontsize=12, color=INK, ha="center", va="center")
    ax.text(0.015, 0.475, "GDN setup", fontsize=8.0, weight="bold", color=INK, ha="left")
    _architecture_bracket(ax, 0.018, 0.275, 0.425)
    ax.text(0.032, 0.387, "materialized", fontsize=7.2, color=INK, ha="left", va="center")
    ax.text(0.032, 0.313, "borrowed immutable", fontsize=7.2, color=INK, ha="left", va="center")

    # Separate immutable sources for the two state families.
    ax.text(0.195, 0.895, "KV cache", fontsize=8.0, weight="bold", color=INK, ha="center")
    _architecture_box(ax, 0.148, 0.675, 0.095, 0.140, "Document\nKV", edgecolor=BLUE, facecolor="#F7FAFC", fontsize=7.0)
    ax.text(0.195, 0.475, "GDN state", fontsize=8.0, weight="bold", color=INK, ha="center")
    _architecture_box(
        ax,
        0.148,
        0.255,
        0.095,
        0.140,
        "Persistent\nGDN base",
        edgecolor=BLUE,
        facecolor="#F7FAFC",
        fontsize=6.5,
    )

    # Phase headers and separators.
    phase_centers = [(0.410, "Setup"), (0.625, "Post-first-transition"), (0.835, "Final")]
    for x, label in phase_centers:
        ax.text(x, 0.935, label, fontsize=8.6, weight="bold", color=INK, ha="center", va="center")
    for x in (0.515, 0.730):
        ax.plot([x, x], [0.155, 0.895], color="#CBD5E1", lw=0.85)
    ax.plot([0.130, 0.955], [0.525, 0.525], color="#CBD5E1", lw=0.8, linestyle=(0, (5, 5)))

    kv_y = 0.650
    gdn_y = 0.315
    box_h = 0.085
    ax.text(0.280, kv_y + box_h / 2, r"$r_0\ldots r_N$", fontsize=7.2, color=INK, ha="right", va="center")
    ax.text(0.280, gdn_y + box_h / 2, r"$r_0\ldots r_N$", fontsize=7.2, color=INK, ha="right", va="center")

    # Setup relations: the policy axes determine the whole group's ownership.
    ax.text(0.405, 0.820, "full copy: request-local", fontsize=6.6, color=INK, ha="center")
    ax.text(0.405, 0.780, "shared paged: shared read-only", fontsize=6.6, color=BLUE, ha="center")
    ax.text(0.405, 0.485, "materialized: request-local", fontsize=6.6, color=INK, ha="center")
    ax.text(0.405, 0.445, "borrowed: read-only alias", fontsize=6.6, color=BLUE, ha="center")
    _architecture_box(ax, 0.300, kv_y, 0.205, box_h, "document region", edgecolor=BLUE, facecolor="#F7FAFC", fontsize=7.0)
    _architecture_box(ax, 0.300, gdn_y, 0.205, box_h, "setup state", edgecolor=BLUE, facecolor="#F7FAFC", fontsize=7.0)
    _architecture_arrow(ax, 0.243, 0.745, 0.292, kv_y + box_h / 2)
    _architecture_arrow(ax, 0.243, 0.325, 0.292, gdn_y + box_h / 2)

    # The document ownership remains policy-dependent; only the append tail is always private.
    _architecture_box(ax, 0.545, kv_y, 0.115, box_h, "document region", edgecolor=BLUE, facecolor="#F7FAFC", fontsize=6.6)
    _architecture_box(ax, 0.660, kv_y, 0.060, box_h, "private\nappend", edgecolor=GREEN, facecolor="#F4FBF7", fontsize=5.8)
    _architecture_arrow(ax, 0.505, kv_y + box_h / 2, 0.538, kv_y + box_h / 2)
    _architecture_box(ax, 0.760, kv_y, 0.115, box_h, "document region", edgecolor=BLUE, facecolor="#F7FAFC", fontsize=6.6)
    _architecture_box(ax, 0.875, kv_y, 0.060, box_h, "private\nappend", edgecolor=GREEN, facecolor="#F4FBF7", fontsize=5.8)
    _architecture_arrow(ax, 0.720, kv_y + box_h / 2, 0.753, kv_y + box_h / 2)

    # Every completed GDN request rebinds to private mutable storage.
    _architecture_box(ax, 0.545, gdn_y, 0.175, box_h, "private mutable", edgecolor=GREEN, facecolor="#F4FBF7", fontsize=7.0)
    _architecture_arrow(ax, 0.505, gdn_y + box_h / 2, 0.538, gdn_y + box_h / 2)
    _architecture_box(ax, 0.760, gdn_y, 0.175, box_h, "private mutable", edgecolor=GREEN, facecolor="#F4FBF7", fontsize=7.0)
    _architecture_arrow(ax, 0.720, gdn_y + box_h / 2, 0.753, gdn_y + box_h / 2)

    # A single restrained transition marker replaces iconography.
    ax.plot([0.518, 0.518], [0.260, 0.745], color=ORANGE, lw=1.0)

    # Replay rail. These are audit classes, not pass/fail claims.
    rail_y = 0.105
    ax.plot([0.145, 0.945], [rail_y, rail_y], color=INK, lw=0.85)
    ax.text(0.205, rail_y + 0.025, "ForkAudit replay", fontsize=7.3, weight="bold", color=INK, ha="center", va="bottom")
    checks = [(0.365, "ownership"), (0.535, "call contract"), (0.705, "FP32 oracle"), (0.875, "live mutants")]
    for x, label in checks:
        ax.plot([x, x], [rail_y, 0.175], color=GRAY, lw=0.7, linestyle=(0, (3, 3)))
        ax.text(x, 0.022, label, fontsize=6.8, color=INK, ha="center", va="bottom")

    _architecture_bracket(ax, 0.950, 0.290, 0.735, direction=-1.0)
    ax.text(0.960, 0.512, "same execution\nschedule", fontsize=6.8, color=INK, ha="left", va="center")

    fig.savefig(output, bbox_inches="tight", pad_inches=0.03, metadata=PDF_METADATA)
    plt.close(fig)


def render_dashboard(summary: dict, mutants: dict, census: dict, dual: dict, falcon: dict, output: Path) -> None:
    values = summary["oracle_relative_l2_by_rank"]
    # Historical aggregate v1 used oracle_max_relative_l2 for the tolerance.
    # Prefer the schema-correct v2 name while preserving byte-identical replay
    # of the registered v1 aggregate.
    threshold = summary.get("oracle_relative_l2_tolerance", summary["oracle_max_relative_l2"])
    require(max(values) <= threshold, "observed oracle error exceeds tolerance")
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.8,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
    blue = "#0F4D92"
    blue_fill = "#E9F1F8"
    green = "#2F6F2B"
    green_fill = "#EAF4E8"
    orange = "#B95717"
    teal = "#16788A"
    neutral = "#6F7478"
    group_fill = "#F3F5F6"
    require(census["passed"] is True, "census did not pass")
    require(census["audited_captures"] == 6, "census capture count drift")
    require(census["audited_row_observations"] == 1080, "census row count drift")
    require(census["audited_relation_observations"] == 96660, "census relation count drift")
    require(census["producer_manifest_used_as_expectation"] is False, "producer manifest became census authority")
    require(dual["passed"] is True, "dual-producer repeat did not pass")
    require(len(dual["producer_receipts"]) == 2, "dual producer count drift")
    require(sum(len(row["observer_pids"]) for row in dual["producer_receipts"]) == 4, "observer count drift")
    require(dual["matched_semantic_coordinates"] == 1080, "dual row count drift")
    require(dual["matched_relation_labels"] == 96660, "dual relation count drift")
    scientific = falcon["scientific_validation"]
    require(falcon["status"] == "PASS", "Falcon validation did not pass")
    require(scientific["rank_count"] == 8, "Falcon rank count drift")
    require(scientific["generated_token_exact"] == 96, "Falcon token count drift")
    require(scientific["full_fp32_logit_byte_exact"] == 96, "Falcon logit count drift")
    require(scientific["semantic_family_rows_exact"] == 13824, "Falcon state-row count drift")

    with mpl.rc_context(style):
        fig = plt.figure(figsize=(7.35, 2.88), facecolor="white")
        gs = fig.add_gridspec(1, 3, width_ratios=[1.42, 1.38, 1.78], wspace=0.48)

        # a. Selected numerical evidence.  Values and the fixed threshold are
        # unchanged; the figures4papers treatment uses direct annotations and
        # only a light y-grid.
        ax = fig.add_subplot(gs[0, 0])
        x_values = list(range(8))
        ax.plot(x_values, values, marker="o", color=blue, linewidth=2.0, markersize=4.5, zorder=3)
        ax.axhline(threshold, color=orange, linestyle=(0, (4, 2)), linewidth=1.6, zorder=2)
        ax.text(
            7.0,
            threshold + threshold * 0.025,
            "fixed tolerance 0.005",
            color=orange,
            fontsize=9.0,
            ha="right",
            va="bottom",
        )
        max_index = values.index(max(values))
        ax.annotate(
            f"max {max(values):.6f}",
            xy=(max_index, max(values)),
            xytext=(max_index - 0.15, max(values) + threshold * 0.17),
            fontsize=9.0,
            color=blue,
            ha="center",
            arrowprops={"arrowstyle": "-", "color": blue, "lw": 0.8},
        )
        ax.set_ylim(0, threshold * 1.18)
        ax.set_xticks(x_values)
        ax.set_xlabel("pre-specified rank / window", fontsize=9.5)
        ax.set_ylabel(r"relative $L_2$", fontsize=9.5)
        ax.tick_params(labelsize=8.8, width=0.9, length=3)
        ax.grid(axis="y", color="#E3E6E8", linewidth=0.7, zorder=0)
        ax.set_title("a  Independent FP32 oracle", fontsize=11.0, weight="bold", loc="left", pad=8)
        ax.text(0.02, 0.75, "8 / 8 pass", transform=ax.transAxes, fontsize=9.2, weight="bold", color=blue)

        # b. Every cell displays the exact registered observable families, not
        # the underspecified historical phrase "byte match".
        ax = fig.add_subplot(gs[0, 1])
        ax.set_xlim(-0.92, 2.03)
        ax.set_ylim(-0.34, 2.16)
        ax.axis("off")
        column_starts = [0.00, 1.18]
        for y in range(2):
            for x in column_starts:
                patch = FancyBboxPatch(
                    (x, y + 0.13),
                    0.76,
                    0.70,
                    boxstyle="round,pad=0.02,rounding_size=0.04",
                    facecolor=green_fill,
                    edgecolor=green,
                    linewidth=1.4,
                )
                ax.add_patch(patch)
                ax.text(x + 0.38, y + 0.65, "token / logit", ha="center", va="center", fontsize=6.8, color=INK)
                ax.text(x + 0.38, y + 0.45, "KV / GDN", ha="center", va="center", fontsize=7.0, color=INK)
                ax.text(x + 0.38, y + 0.23, "MATCH", ha="center", va="center", fontsize=8.2, color=green, weight="bold")
        ax.text(0.38, 2.00, "Full-copy\nKV", ha="center", va="center", fontsize=7.0, color=INK)
        ax.text(1.56, 2.00, "Shared-doc\nKV", ha="center", va="center", fontsize=7.0, color=INK)
        ax.text(-0.22, 1.50, "Materialized\nGDN", ha="right", va="center", fontsize=7.8, color=INK)
        ax.text(-0.22, 0.50, "Borrowed\nGDN", ha="right", va="center", fontsize=7.8, color=INK)
        ax.text(
            0.50,
            -0.02,
            "4 cells x 3 fan-outs x 8 books",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7.8,
            color=neutral,
        )
        ax.set_title("b  Cross-cell equivalence", fontsize=11.0, weight="bold", loc="left", pad=8)

        # c. Replace constructed-fault prominence with the independently
        # enumerated coverage checks and bounded second-configuration transfer.
        ax = fig.add_subplot(gs[0, 2])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("c  Coverage and bounded transfer", fontsize=11.0, weight="bold", loc="left", pad=8)
        cards = [
            (
                0.69,
                blue,
                blue_fill,
                "PREPRODUCER CENSUS",
                "180 slots x 6 captures",
                "1,080 rows · 96,660 relations",
                "producer rows are not authority",
            ),
            (
                0.37,
                teal,
                "#E7F5F7",
                "DUAL REPEAT",
                "2 executions · 4 observers",
                "1,080 + 96,660 exact / run",
                "zero tolerance",
            ),
            (
                0.05,
                green,
                green_fill,
                "FALCON-H1 TRANSFER",
                "8 / 8 ranks",
                "96 token cmp. · 96 FP32-logit cmp.",
                "13,824 state rows · declared config only",
            ),
        ]
        for y, accent, face, title, headline, detail, boundary in cards:
            ax.add_patch(
                FancyBboxPatch(
                    (0.01, y),
                    0.98,
                    0.26,
                    boxstyle="round,pad=0.012,rounding_size=0.025",
                    facecolor=face,
                    edgecolor=accent,
                    linewidth=1.0,
                )
            )
            ax.text(0.04, y + 0.215, title, fontsize=8.4, weight="bold", color=accent, ha="left", va="center")
            ax.text(0.04, y + 0.145, headline, fontsize=9.0, weight="bold", color=INK, ha="left", va="center")
            ax.text(0.04, y + 0.085, detail, fontsize=7.9, color=INK, ha="left", va="center")
            ax.text(0.04, y + 0.032, boundary, fontsize=7.0, color=neutral, ha="left", va="center")
            ax.text(0.955, y + 0.215, "PASS", fontsize=7.3, weight="bold", color=accent, ha="right", va="center")

        fig.subplots_adjust(left=0.072, right=0.995, bottom=0.20, top=0.84, wspace=0.48)
        fig.savefig(output, bbox_inches="tight", pad_inches=0.04, dpi=300, metadata=PDF_METADATA)
        plt.close(fig)


def render_memory_table(summary: dict, output: Path) -> None:
    cells = [cell for cell in summary["memory_matrix"]["cells"] if cell["resident_count"] == 32]
    require(len(cells) == 4, "expected four N=32 memory cells")
    lines = [
        r"\begin{table}[H]",
        r"\caption{Primary 8$\times$H20-3e allocator results at $N=32$ (median across ranks; rank values coincide). Vanilla full-copy is the external control; the paged-prefix row is the closest same-stack prefix-sharing baseline. The remaining rows isolate recurrent-base ownership. All cells retain the persistent document source. Values are GiB above the frozen post-priming baseline; generation increment is the additional peak above allocation present at generation start.}",
        r"\label{tab:rr2-memory}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\begin{tabular}{@{}p{0.25\linewidth}p{0.25\linewidth}rrr@{}}",
        r"\toprule",
        r"Empirical arm & KV / GDN setup & Final & Peak & Gen. incr. \\",
        r"\midrule",
    ]
    cells.sort(key=lambda c: ("shared" in c["kv_policy"], "borrow" in c["gdn_base_policy"]))
    for cell in cells:
        a = cell["allocator_median_across_ranks"]
        key = (cell["kv_policy"], cell["gdn_base_policy"])
        arm = {
            (
                "vllm-q16-fresh-full-copy-control",
                "materialize-request-base-functional-rebind",
            ): "Vanilla full-copy",
            (
                "vllm-q16-fresh-full-copy-control",
                "borrow-immutable-base-functional-rebind",
            ): "GDN ownership ablation",
            (
                "vllm-q16-shared-document-reuse",
                "materialize-request-base-functional-rebind",
            ): "Paged-prefix baseline",
            (
                "vllm-q16-shared-document-reuse",
                "borrow-immutable-base-functional-rebind",
            ): "Audited hybrid fork",
        }[key]
        lines.append(
            f"{arm} & {policy_label(cell['kv_policy'])} / {policy_label(cell['gdn_base_policy'])} & "
            f"{a['after_generation_current_allocated_delta_bytes']/GIB:.3f} & "
            f"{a['setup_plus_generation_peak_allocated_delta_bytes']/GIB:.3f} & "
            f"{a['generation_peak_allocated_delta_bytes']/GIB:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def render_mutant_table(mutants: dict, output: Path) -> None:
    names = {
        "M1": "reservation alias",
        "M2": "sequence swap",
        "M3": "omit tail COW",
        "M4": "GDN--base alias",
        "M5": "GDN peer alias",
        "M6": "position off-by-one",
        "M7": "materialized mask",
        "M8": "wrong callable",
        "M9": "dense KV view",
    }
    lines = [
        r"\begin{table}[H]",
        r"\caption{Separate primary all-gates-on reference. Every separately rebuilt matched-clean cell passed; every mutant reached its target gate fixed before execution and was restored before disposal.}",
        r"\label{tab:rr2-mutants}",
        r"\centering\footnotesize",
        r"\begin{tabular}{@{}clll@{}}",
        r"\toprule",
        r"ID & Injected live fault & Expected/observed gate & Primary outcome \\",
        r"\midrule",
    ]
    for row in mutants["rows"]:
        mid = row["mutant_id"]
        gate = row["expected_gate_id"].replace("_", r"\_")
        lines.append(f"{mid} & {names[mid]} & " + r"\texttt{" + gate + r"} & target gate reached \\ ")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--mutants", type=Path, default=DEFAULT_MUTANTS)
    parser.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    parser.add_argument("--dual", type=Path, default=DEFAULT_DUAL)
    parser.add_argument("--falcon", type=Path, default=DEFAULT_FALCON)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args()
    summary = load_json(args.summary)
    mutants = load_json(args.mutants)
    census = load_json(args.census)
    dual = load_dual_summary(args.dual)
    falcon = load_json(args.falcon)
    validate(summary, mutants)
    figures = args.output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    render_dashboard(summary, mutants, census, dual, falcon, figures / "rr2_validation_dashboard_r39.pdf")
    print(f"observed_oracle_max={max(summary['oracle_relative_l2_by_rank']):.17g}")
    print("factorial_cells=12; ranks=8; census_rows=1080; falcon_rows=13824")


if __name__ == "__main__":
    main()
