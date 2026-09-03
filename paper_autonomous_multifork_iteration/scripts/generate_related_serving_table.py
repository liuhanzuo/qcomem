#!/usr/bin/env python3
"""Generate the same-protocol serving-framework table from frozen summaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]
EVIDENCE = PAPER / "evidence" / "related_work_same_protocol"
VLLM = EVIDENCE / "related-vllm-prefix-bootstrap-f-20260820a" / "summary.json"
SGLANG = EVIDENCE / "related-sglang-radix-node-20260821c" / "summary.json"
HYPIC = EVIDENCE / "hypic-same-protocol-20260821c" / "summary.json"
OUT = EVIDENCE / "serving_panel_summary.json"
TABLE = PAPER / "tables" / "related_serving_table.tex"

EXPECTED = {
    "vllm": "625282cff4a7a371c2c5f4c55f4a4173b9a304cbcb0717365bbc254e660ed137",
    "sglang": "509b0c6a148313eac1ab7f5d6011bff80bd0e7e8c91fc054b34a20f54a17070d",
    "hypic": "0543b491e70ddfaf6d40651b1f1babec652bd9c8f2a5f9d0cca7305cc2cb1b3d",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path, expected: str) -> dict:
    require(sha256(path) == expected, f"summary drift: {path}")
    value = json.loads(path.read_text())
    require(value["scientific_run_valid"] is True, "invalid scientific run")
    require(value["scientific_outcome"] == "valid_positive", "non-positive run")
    require(value["cache_hit_observed"] == [True] * 8, "cache hit drift")
    require(value["cache_off_vs_on_prediction_exact"] == [True] * 8, "prediction drift")
    require(value["pairs"] == [
        ["qasper", 6], ["qasper", 7], ["qasper", 8], ["qasper", 9],
        ["2wikimqa", 6], ["2wikimqa", 7], ["2wikimqa", 8], ["2wikimqa", 9],
    ], "workload assignment drift")
    return value


def row(system: str, phase: str, value: dict) -> dict:
    phase_value = value["phases"][phase]
    return {
        "system": system,
        "phase": phase,
        "mean_f1_points": 100.0 * phase_value["mean_f1"],
        "median_ttft_seconds": phase_value["median_ttft_seconds"],
        "median_tpot_ms": 1000.0 * phase_value["median_tpot_seconds"],
        "median_generated_tokens_per_second": phase_value["median_generated_tokens_per_second"],
        "cache_hits": "8/8" if phase == "cache_on" else "--",
        "prediction_exact_vs_off": "8/8" if phase == "cache_on" else "reference",
    }


def hypic_row(phase: str, value: dict) -> dict:
    phase_value = value["modes"][phase]
    exact = sum(phase_value["prediction_text_exact_vs_full_recompute"])
    hits = sum(token is not None and token > 0 for token in phase_value["cached_tokens"])
    return {
        "system": "HYPIC/SGLang 0.5.14",
        "phase": phase,
        "mean_f1_points": 100.0 * phase_value["mean_f1"],
        "median_ttft_seconds": phase_value["median_ttft_seconds"],
        "median_tpot_ms": 1000.0 * phase_value["median_of_per_request_mean_post_first_token_seconds"],
        "median_generated_tokens_per_second": phase_value["median_generated_tokens_per_second"],
        "cache_hits": "--" if phase == "full_recompute" else f"{hits}/8",
        "prediction_exact_vs_off": "reference" if phase == "full_recompute" else f"{exact}/8",
    }


def main() -> None:
    vllm = load(VLLM, EXPECTED["vllm"])
    sglang = load(SGLANG, EXPECTED["sglang"])
    require(sha256(HYPIC) == EXPECTED["hypic"], "HYPIC summary drift")
    hypic = json.loads(HYPIC.read_text())
    require(hypic["scientific_run_valid"] is True, "invalid HYPIC run")
    require(hypic["protocol_validity"] == "passed", "HYPIC protocol invalid")
    rows = [
        row("vLLM 0.26", "cache_off", vllm),
        row("vLLM 0.26", "cache_on", vllm),
        row("SGLang 0.5.17", "cache_off", sglang),
        row("SGLang 0.5.17", "cache_on", sglang),
        hypic_row("full_recompute", hypic),
        hypic_row("prefix_cache", hypic),
        hypic_row("transition_rope_recompute", hypic),
    ]
    summary = {
        "schema": "forkaudit-related-serving-panel-v1",
        "status": "verified_complete",
        "source_summaries": {
            "vllm": {"path": str(VLLM.relative_to(PAPER)), "sha256": EXPECTED["vllm"]},
            "sglang": {"path": str(SGLANG.relative_to(PAPER)), "sha256": EXPECTED["sglang"]},
            "hypic": {"path": str(HYPIC.relative_to(PAPER)), "sha256": EXPECTED["hypic"]},
        },
        "fixed_protocol": {
            "model": "Qwen3.5-35B-A3B",
            "datasets": ["Qasper", "2WikiMQA"],
            "source_indices": [6, 7, 8, 9],
            "workloads": 8,
            "input_token_cap": 4096,
            "max_generated_tokens": 32,
            "decoding": "greedy",
            "hardware": "one independent NVIDIA H20-3e per workload",
            "timing_boundary": "OpenAI-compatible streaming client wall-clock",
        },
        "rows": rows,
        "boundary": (
            "Cache-on/off predictions are exact only within each framework pair. "
            "These HTTP serving timings are not pooled with in-process CoMem adapter timings. "
            "HYPIC is an official-code TP=1 adaptation; the paper's Qwen3.5 panels use TP=2."
        ),
    }
    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    labels = {
        ("vLLM 0.26", "cache_off"): ("vLLM 0.26", "off"),
        ("vLLM 0.26", "cache_on"): ("vLLM 0.26", "align on"),
        ("SGLang 0.5.17", "cache_off"): ("SGLang 0.5.17", "off"),
        ("SGLang 0.5.17", "cache_on"): ("SGLang 0.5.17", "Radix on"),
        ("HYPIC/SGLang 0.5.14", "full_recompute"): ("HYPIC code", "full recompute"),
        ("HYPIC/SGLang 0.5.14", "prefix_cache"): ("HYPIC code", "prefix cache"),
        ("HYPIC/SGLang 0.5.14", "transition_rope_recompute"): ("HYPIC", "transition+seam 8"),
    }
    lines = [
        r"\begin{table}[H]",
        r"\caption{Unpooled timing/quality-only serving context on Qwen3.5 and the same eight-item Qasper/2WikiMQA slice.  Each row uses one H20-3e per item, a 4,096-token cap, greedy decoding, and at most 32 generated tokens.  Times are OpenAI-compatible streaming client wall-clock; F1 is the eight-item mean.  Hit is interpreted only against the disabled/full-recompute reference inside each framework block.  No cross-framework speedup, retained-state, scheduler-throughput, capacity, or broad-quality claim is made.}",
        r"\label{tab:serving-controls}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\begin{tabular}{@{}llrrrrc@{}}",
        r"\toprule",
        "Framework & Prefix reuse & F1 & TTFT (s) & TPOT (ms) & tok/s & Hit " + r"\\",
        r"\midrule",
    ]
    for index, value in enumerate(rows):
        framework, cache = labels[(value["system"], value["phase"])]
        if index in (2, 4):
            lines.append(r"\midrule")
        lines.append(
            f"{framework} & {cache} & {value['mean_f1_points']:.2f} & "
            f"{value['median_ttft_seconds']:.3f} & {value['median_tpot_ms']:.2f} & "
            f"{value['median_generated_tokens_per_second']:.2f} & "
            f"{value['cache_hits']} \\\\" 
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.4mm}",
        r"\parbox{0.97\textwidth}{\footnotesize vLLM uses experimental Mamba \texttt{align}; SGLang 0.5.17 uses \texttt{extra\_buffer} Radix caching.  The HYPIC rows use the authors' commit 98147c0 on SGLang 0.5.14, TP=1, full \texttt{transition\_rope\_recompute}, and an eight-token seam.  Its full and prefix rows are controls from that same codebase; published TP=2 results and the separate retained-state receipt cohort are not inserted here.}",
        r"\end{table}",
        "",
    ])
    TABLE.write_text("\n".join(lines))
    print(json.dumps({"rows": len(rows), "output": str(TABLE)}, sort_keys=True))


if __name__ == "__main__":
    main()
