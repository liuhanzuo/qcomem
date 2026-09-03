#!/usr/bin/env python3
"""Generate the H20 quality--speed--state table from the frozen deployment run."""

from __future__ import annotations

import csv
import hashlib
import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from statistics import median


PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
RUN_ROOT = REPO / "results/gpu-deployment-validation-20260812i"
SOURCE = RUN_ROOT / "deployment-summary.json"
GPU_INVENTORY = RUN_ROOT / "gpus-before.csv"
HYPIC_SOURCE = (
    PAPER
    / "evidence/related_work_same_protocol/hypic-same-protocol-20260821c/summary.json"
)
HYPIC_STORE_ACCEPTANCE = (
    PAPER
    / "evidence/related_work_same_protocol/hypic-retained-state-r34-trial1892234/acceptance.json"
)
OUT_ROOT = PAPER / "evidence/h20_deployment_benchmark"
TABLE = PAPER / "tables/h20_deployment_table.tex"

EXPECTED_SOURCE_SHA256 = "1574741fdebe9b378196ea70f4af0efc7da1ba7941b543bb67519e596d10c835"
EXPECTED_GPU_SHA256 = "50eef376f4c8f4325924cceca941831f45efdb159930b61a954a30530d51b415"
EXPECTED_HYPIC_SHA256 = "0543b491e70ddfaf6d40651b1f1babec652bd9c8f2a5f9d0cca7305cc2cb1b3d"
# Filled only after the independently replayed acceptance is frozen locally.
EXPECTED_HYPIC_STORE_ACCEPTANCE_SHA256 = "15dbee59e8f422a944cdcc2bd67c276b359b34327230569bd14f9afdb787cbec"
EXPECTED_COMEM_SOURCE_CONFIGS = {
    "dense-recompute",
    "full-prefix-q16",
    "qcomem-d7-r16-a16-l16",
    "qcomem-d7-r8-a8-l8",
    "qcomem-d7-r4-a4-l8",
    "qcomem-d7-mixed",
    "qcomem-d7-r4-a4-l4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def format_mib(byte_count: int) -> str:
    return f"{Decimal(byte_count) / Decimal(1024 * 1024):.2f}"


def main() -> None:
    require(sha256(SOURCE) == EXPECTED_SOURCE_SHA256, "deployment summary drift")
    require(sha256(GPU_INVENTORY) == EXPECTED_GPU_SHA256, "GPU inventory drift")
    require(sha256(HYPIC_SOURCE) == EXPECTED_HYPIC_SHA256, "HYPIC summary drift")
    require(
        sha256(HYPIC_STORE_ACCEPTANCE) == EXPECTED_HYPIC_STORE_ACCEPTANCE_SHA256,
        "HYPIC Store acceptance drift",
    )
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    require(payload.get("status") == "completed", "deployment run not complete")
    require(payload.get("all_exactness_gates_passed") is True, "exactness gate failed")
    require(payload.get("shards") == 8 and payload.get("rows") == 168, "run cardinality drift")
    require(payload.get("workload") == "longbench", "workload drift")
    metadata = payload.get("workload_metadata", {})
    require(metadata.get("datasets") == ["2wikimqa", "qasper"], "dataset drift")
    require(metadata.get("source_index_start") == 6, "source range drift")
    require(metadata.get("test_v2_consumed") is False, "test-v2 must remain unread")

    with GPU_INVENTORY.open(newline="", encoding="utf-8") as handle:
        gpu_rows = list(csv.DictReader(handle, skipinitialspace=True))
    require(len(gpu_rows) == 8, "expected eight GPUs")
    require({row["name"] for row in gpu_rows} == {"NVIDIA H20-3e"}, "GPU model drift")
    require(len({row["uuid"] for row in gpu_rows}) == 8, "GPU UUIDs must be distinct")

    by_config = {row["config"]: row for row in payload["summary_overall_config"]}
    require(set(by_config) == EXPECTED_COMEM_SOURCE_CONFIGS, "source configuration set drift")
    display = [
        ("dense-recompute", "Vanilla dense", "--", "vanilla"),
        (
            "full-prefix-q16",
            "Full-prefix KV cache",
            "Q16",
            "related_work_baseline",
        ),
        ("qcomem-d7-r8-a8-l8", "CoMem state", "Q8", "ours"),
        ("qcomem-d7-r4-a4-l8", "CoMem mixed state", "Q4/Q4/Q8", "ours"),
        ("qcomem-d7-mixed", "CoMem per-layer mixed", "mixed", "ours"),
        ("qcomem-d7-r4-a4-l4", "CoMem state", "Q4", "ours"),
    ]

    rows = []
    for config, method, state_bits, role in display:
        source_row = by_config[config]
        require(source_row["workloads"] == 8, f"workload count drift for {config}")
        require(source_row["measurements"] == 24, f"repeat count drift for {config}")
        store_mib = source_row["persistent_document_nbytes"]["median"] / (1024.0 * 1024.0)
        rows.append(
            {
                "config": config,
                "method": method,
                "role": role,
                "state_bits": state_bits,
                "persistent_store_mib_median": store_mib,
                "ttft_seconds_median": source_row["ttft_seconds"]["median"],
                "tpot_ms_median": 1000.0 * source_row["median_tpot_seconds"]["median"],
                "throughput_tokens_per_second_median": source_row["throughput_tokens_per_second"]["median"],
                "longbench_f1_points_mean": 100.0 * source_row["f1"]["mean"],
            }
        )

    hypic = json.loads(HYPIC_SOURCE.read_text(encoding="utf-8"))
    require(hypic.get("scientific_run_valid") is True, "invalid HYPIC run")
    require(hypic.get("protocol_validity") == "passed", "HYPIC protocol gate")
    require(hypic.get("official_commit") == "98147c01909004e66d98bcb18b886927d41b0ee5", "HYPIC source drift")
    store_acceptance = json.loads(HYPIC_STORE_ACCEPTANCE.read_text(encoding="utf-8"))
    require(
        store_acceptance.get("schema_version")
        == "hypic-rwd5-trial1892234-external-store-acceptance-v1",
        "HYPIC Store acceptance schema drift",
    )
    require(
        store_acceptance.get("status") == "passed_external_replay_16_of_16",
        "HYPIC Store acceptance status drift",
    )
    require(store_acceptance.get("job_id") == 247699, "HYPIC Store job drift")
    require(store_acceptance.get("trial_id") == 1892234, "HYPIC Store trial drift")
    require(
        store_acceptance.get("official_commit")
        == "98147c01909004e66d98bcb18b886927d41b0ee5",
        "HYPIC Store source drift",
    )
    terminal_cells = store_acceptance.get("terminal_cells", {})
    require(
        terminal_cells.get("passed") == terminal_cells.get("expected") == 16,
        "HYPIC Store terminal-cell coverage drift",
    )
    replay = store_acceptance.get("external_replay", {})
    require(replay.get("passed") is True and replay.get("rows") == 16,
            "HYPIC Store external replay drift")
    store_by_mode: dict[str, dict[str, object]] = {}
    expected_store_medians = {
        "prefix_cache": 146_309_120,
        "transition_rope_recompute": 339_834_880,
    }
    for mode, expected_median in expected_store_medians.items():
        mode_store = store_acceptance.get("modes", {}).get(mode, {})
        values = mode_store.get("payload_bytes")
        require(
            isinstance(values, list)
            and len(values) == 8
            and all(isinstance(value, int) and value > 0 for value in values),
            f"HYPIC Store payload coverage drift: {mode}",
        )
        recomputed_median = int(median(values))
        require(recomputed_median == expected_median, f"HYPIC Store median drift: {mode}")
        require(
            mode_store.get("median_payload_bytes") == recomputed_median,
            f"HYPIC Store recorded median drift: {mode}",
        )
        store_by_mode[mode] = {
            "payload_bytes": values,
            "median_payload_bytes": recomputed_median,
            "median_payload_mib_exact": str(
                Decimal(recomputed_median) / Decimal(1024 * 1024)
            ),
        }
    comem_q8_store_bytes = int(
        by_config["qcomem-d7-r8-a8-l8"]["persistent_document_nbytes"]["median"]
    )
    require(comem_q8_store_bytes == 16_664_352, "CoMem Q8 Store median drift")
    store_ratios = {
        "hypic_over_prefix": Fraction(
            expected_store_medians["transition_rope_recompute"],
            expected_store_medians["prefix_cache"],
        ),
        "prefix_over_comem_q8": Fraction(
            expected_store_medians["prefix_cache"], comem_q8_store_bytes
        ),
        "hypic_over_comem_q8": Fraction(
            expected_store_medians["transition_rope_recompute"], comem_q8_store_bytes
        ),
    }

    hypic_display = [
        ("full_recompute", "Official HYPIC code: full recompute", "--"),
        ("prefix_cache", "Official HYPIC code: prefix cache", "Radix prefix"),
        ("transition_rope_recompute", "HYPIC (transition + seam 8)", "segment transition"),
    ]
    hypic_rows = []
    for key, method, state in hypic_display:
        value = hypic["modes"][key]
        require(len(value["per_row_f1"]) == 8, f"HYPIC row count drift: {key}")
        require(value["post_first_token_denominator_rows"] == 8, f"HYPIC TPOT denominator drift: {key}")
        hypic_rows.append(
            {
                "config": key,
                "method": method,
                "state_bits": state,
                "persistent_store_mib_median": (
                    None
                    if key == "full_recompute"
                    else format_mib(int(store_by_mode[key]["median_payload_bytes"]))
                ),
                "ttft_seconds_median": value["median_ttft_seconds"],
                "tpot_ms_median": 1000.0 * value["median_of_per_request_mean_post_first_token_seconds"],
                "throughput_tokens_per_second_median": value["median_generated_tokens_per_second"],
                "longbench_f1_points_mean": 100.0 * value["mean_f1"],
                "prediction_exact_vs_full": sum(value["prediction_text_exact_vs_full_recompute"]),
            }
        )

    shard_rows = []
    for index in range(8):
        path = RUN_ROOT / f"deployment-shard-{index}.json"
        require(path.is_file(), f"missing raw shard {index}")
        shard_rows.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})

    summary = {
        "schema_version": "forkaudit-h20-deployment-context-v2",
        "status": "verified_separate_deployment_context",
        "source_summary": {
            "file": SOURCE.name,
            "bytes": SOURCE.stat().st_size,
            "sha256": sha256(SOURCE),
        },
        "source_shards": shard_rows,
        "gpu_inventory": {
            "file": GPU_INVENTORY.name,
            "sha256": sha256(GPU_INVENTORY),
            "gpu_count": 8,
            "gpu_name": "NVIDIA H20-3e",
            "distinct_uuid_count": 8,
        },
        "model": "Qwen3.5-35B-A3B (BF16 weights)",
        "runtime": "PyTorch 2.11.0+cu129 / Transformers 5.14.1",
        "workload": {
            "name": "LongBench validation slice",
            "datasets": ["Qasper", "2WikiMQA"],
            "workloads": 8,
            "timing_repeats_per_configuration": 3,
            "maximum_input_tokens": 4096,
            "maximum_generated_tokens": 32,
        },
        "rows": rows,
        "hypic_rows": hypic_rows,
        "hypic_source": {
            "file": str(HYPIC_SOURCE.relative_to(PAPER)),
            "sha256": EXPECTED_HYPIC_SHA256,
            "official_commit": hypic["official_commit"],
        },
        "hypic_store_acceptance": {
            "file": str(HYPIC_STORE_ACCEPTANCE.relative_to(PAPER)),
            "sha256": EXPECTED_HYPIC_STORE_ACCEPTANCE_SHA256,
            "schema_version": store_acceptance["schema_version"],
            "status": store_acceptance["status"],
            "job_id": store_acceptance["job_id"],
            "trial_id": store_acceptance["trial_id"],
            "terminal_cells": terminal_cells,
            "external_replay": replay,
            "denominator": store_acceptance["denominator"],
            "modes": store_by_mode,
            "comem_q8_median_payload_bytes": comem_q8_store_bytes,
            "comparisons": {
                name: {
                    "exact_fraction": f"{value.numerator}/{value.denominator}",
                    "decimal": float(value),
                    "display": f"{float(value):.2f}x",
                }
                for name, value in store_ratios.items()
            },
        },
        "interpretation": (
            "The first block evaluates vanilla dense, full-prefix KV reuse, and CoMem variants in "
            "one in-process Transformers cohort. The second block is an official-code HYPIC TP=1 "
            "timing/quality adaptation using SGLang 0.5.14 and a streaming client, augmented by an "
            "independent 16-cell receipt cohort for retained-document Store. Both use the same "
            "checkpoint, eight-item slice, H20-3e hardware, input cap, and decoding budget, but "
            "their runtimes and timing harnesses differ, so no cross-block speedup is reported."
        ),
        "store_boundary": (
            "Store is median retained-document tensor payload under a target-entry-owned physical "
            "byte-range-union denominator. It excludes Python metadata, allocator/preallocated "
            "pools, NVML/process deltas, admission capacity, and timing. The HYPIC Store values "
            "come from 16 externally accepted measurement cells, not the separate 24-cell "
            "timing/quality cohort."
        ),
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        r"\begin{table}[H]",
        r"\caption{Unpooled same-checkpoint H20 context on the eight-item Qasper/2WikiMQA slice.  The first block is an in-process Transformers deployment cohort with three timing repeats per item.  HYPIC~\citep{liu2026hypic} timing/F1 comes from a separate 24-cell TP=1 official-code cohort; its Store values come from an independent receipt-instrumented 8+8-cell cohort.  All use Qwen3.5-35B-A3B, one H20-3e per item, a 4,096-token cap, greedy decoding, and at most 32 generated tokens.  Timing is interpreted within block; no cross-block speedup, production-capacity, broad-quality, or ForkAudit-ownership claim is made.}",
        r"\label{tab:h20-deployment}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"Method & Persistent state & Store (MiB) & TTFT (s) & TPOT (ms) & tok/s & F1 \\",
        r"\midrule",
        r"\multicolumn{7}{@{}l}{\emph{Transformers 5.14.1 in-process deployment cohort}}\\",
    ]
    for row in rows:
        store = "--" if row["config"] == "dense-recompute" else f"{row['persistent_store_mib_median']:.2f}"
        cells = [
            row["method"],
            row["state_bits"],
            store,
            f"{row['ttft_seconds_median']:.3f}",
            f"{row['tpot_ms_median']:.2f}",
            f"{row['throughput_tokens_per_second_median']:.2f}",
            f"{row['longbench_f1_points_mean']:.2f}",
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{@{}l}{\emph{HYPIC commit 98147c0 timing/F1; receipt-instrumented Store; SGLang 0.5.14, TP=1}}\\")
    for row in hypic_rows:
        store = (
            "--"
            if row["config"] == "full_recompute"
            else row["persistent_store_mib_median"]
        )
        cells = [
            row["method"],
            row["state_bits"],
            store,
            f"{row['ttft_seconds_median']:.3f}",
            f"{row['tpot_ms_median']:.2f}",
            f"{row['throughput_tokens_per_second_median']:.2f}",
            f"{row['longbench_f1_points_mean']:.2f}",
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.5mm}",
            r"\parbox{0.97\textwidth}{\footnotesize Store is the median retained-document tensor payload: persistent document bytes for CoMem/full-prefix reuse and the union of target-entry-owned physical tensor byte ranges for Prefix Cache/HYPIC.  It excludes Python metadata, preallocated pools, NVML/process deltas, and admission capacity.  HYPIC timing/F1 uses streaming-client wall-clock from 24 cells; HYPIC Store uses a separate 16-cell externally replayed receipt cohort.  Published Qwen3.5 HYPIC panels use TP=2, whereas this fixed same-slice adaptation uses TP=1.}",
            r"\end{table}",
            "",
        ]
    )
    TABLE.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
