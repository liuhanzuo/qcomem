#!/usr/bin/env python3
"""Render Figure 2 from the frozen critical-radius result JSON only.

This script performs no fitting, selection, or scientific recomputation.  It
hash-checks the supplied result and redraws its stored curves using the local
publication style.  Run from any directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_INPUT = PROJECT / "remote_snapshot/results/tv_conservation_r484_result.json"
DEFAULT_OUTPUT = HERE / "fig_tau_conservation.png"
EXPECTED_INPUT_SHA256 = "6aa9b051891fbc9d53254223e92dd0980e40031b403b4ee1698ac8f9a90f9d34"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    actual_hash = sha256(args.input)
    if actual_hash != EXPECTED_INPUT_SHA256:
        raise SystemExit(
            f"refusing to render unexpected input: {actual_hash} != "
            f"{EXPECTED_INPUT_SHA256}"
        )
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    spec = payload["alpha_grid"]
    alpha = np.arange(spec["lo"], spec["hi"] + spec["step"] / 2, spec["step"])
    if len(alpha) != 399:
        raise AssertionError(f"unexpected alpha grid length: {len(alpha)}")

    styles = {
        "omr_shard0": ("#0F4D92", "-", "OMR shard 0"),
        "omr_shard1": ("#5B8FD1", "--", "OMR shard 1"),
        "rlve": ("#2D8C72", "-.", "RLVE"),
        "openr1": ("#B64342", "-", "OpenR1 ($M=2$)"),
    }
    for name in styles:
        tau = payload["curves"][name]["tau"]
        if len(tau) != len(alpha):
            raise AssertionError(f"{name}: {len(tau)} values for {len(alpha)} levels")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.linewidth": 1.15,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "lines.linewidth": 2.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.15, 3.65))
    for name, (color, linestyle, label) in styles.items():
        ax.plot(
            alpha,
            payload["curves"][name]["tau"],
            color=color,
            linestyle=linestyle,
            linewidth=2.1,
            label=label,
            zorder=3,
        )

    # Reference levels are short axis ticks rather than full-height grid lines.
    for level in payload["reference_alphas"]:
        ax.plot(
            [level],
            [0.005],
            marker="|",
            color="#6B7280",
            markersize=8,
            markeredgewidth=1.2,
            clip_on=False,
            zorder=5,
        )
    ax.text(
        0.103,
        0.04,
        r"reference $\alpha$ levels",
        color="#6B7280",
        fontsize=7.8,
        ha="left",
        va="bottom",
    )

    # These are frozen OpenR1 rule-change thresholds, not uncertainty bands.
    openr1_color = styles["openr1"][0]
    for breakpoint in payload["curves"]["openr1"]["certs"]:
        ax.axvline(
            breakpoint,
            color=openr1_color,
            linewidth=1.0,
            linestyle=":",
            alpha=0.8,
            zorder=1,
        )
    ax.annotate(
        "OpenR1 rule changes\n$0.0460$ and $0.0785$",
        xy=(0.0785, 0.83),
        xytext=(0.103, 0.88),
        color=openr1_color,
        fontsize=8,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "color": openr1_color, "lw": 0.9},
    )

    ax.set_xlim(0, 0.205)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"certificate level $\alpha$")
    ax.set_ylabel(r"critical assumed-ball radius $\tau^*(\alpha)$")
    ax.set_title(
        "Fixed-rule sensitivity over an assumed TV ball",
        loc="left",
        fontweight="semibold",
        pad=20,
    )
    ax.text(
        0,
        1.025,
        "Deterministic diagnostic; not target-population containment",
        transform=ax.transAxes,
        color="#5B6472",
        fontsize=8.4,
        ha="left",
        va="bottom",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.5, width=0.9)
    ax.legend(loc="upper left", ncol=2, frameon=False, handlelength=2.8)

    fig.tight_layout(pad=0.7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "Matplotlib; frozen-result renderer"},
    )
    plt.close(fig)
    print(f"input_sha256={actual_hash}")
    print(f"output={args.output}")
    print(f"output_sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
