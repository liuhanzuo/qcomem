from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from qcomem_qwen35_paged_integration import KERNEL_MODE
from run_downstream import atomic_json


class AggregateError(RuntimeError):
    pass


CONFIGS = (
    "dense-native-functional",
    "paged-q16",
    "paged-q8",
    "paged-q4",
)
PAGED_CONFIGS = CONFIGS[1:]


def _require_gate(shard: dict[str, Any]) -> None:
    gate = shard.get("gate", {})
    if gate.get("passed") is not True:
        raise AggregateError(f"rank {shard.get('rank')} top-level gate failed")
    native = gate.get("native_functional", {})
    native_same = gate.get("native_same_caller", {})
    paged = gate.get("paged_same_caller", {})
    counts = gate.get("config_derived_counts", {})
    if native.get("passed") is not True or native_same.get("passed") is not True:
        raise AggregateError("native functional component gate failed")
    if native_same.get("baseline") != "stock-transformers-mutable-eager":
        raise AggregateError("native same-caller baseline is not stock mutable eager")
    if native.get("config_derived") is not True or native_same.get(
        "config_derived"
    ) is not True:
        raise AggregateError("native layer counts are not config-derived")
    if native.get("expected_linear_layer_count") != counts.get(
        "linear_layer_count"
    ):
        raise AggregateError("linear layer gate/count mismatch")
    if native.get("expected_full_attention_layer_count") != counts.get(
        "full_attention_layer_count"
    ):
        raise AggregateError("full layer gate/count mismatch")
    if paged.get("passed") is not True:
        raise AggregateError("paged same-caller gate failed")
    if paged.get("intercept", {}).get("verified") is not True:
        raise AggregateError("paged intercept coverage was not verified")
    if paged.get("intercept", {}).get("dense_fallback_calls") != 0:
        raise AggregateError("paged gate used dense fallback")
    if gate.get("benchmark_authorization", {}).get(
        "benchmark_gate_passed"
    ) is not True:
        raise AggregateError("benchmark ran without correctness authorization")


def _require_measurement(name: str, value: dict[str, Any], tokens: int) -> None:
    if value.get("config") != name:
        raise AggregateError(f"measurement key/config mismatch: {name}")
    if value.get("production_ttft_optimization_claim_allowed") is not False:
        raise AggregateError(f"{name} permits a production TTFT claim")
    per_query = value.get("per_query")
    if not isinstance(per_query, list) or len(per_query) < 2:
        raise AggregateError(f"{name} is not a multi-query measurement")
    if value.get("queries_per_document") != len(per_query):
        raise AggregateError(f"{name} query-count metadata mismatch")
    if value.get("warmup_count") != 1:
        raise AggregateError(f"{name} did not use the frozen one-config warmup")
    if any(len(row.get("generated_token_ids", [])) != tokens for row in per_query):
        raise AggregateError(f"{name} generated-token count mismatch")
    positive_fields = (
        "persistent_total_resident_nbytes",
        "multi_query_active_total_resident_nbytes",
        "cuda_peak_allocated_bytes",
        "nvml_sampled_peak_process_bytes",
        "ttft_seconds",
        "median_tpot_seconds",
    )
    if any(float(value.get(field, 0)) <= 0 for field in positive_fields):
        raise AggregateError(f"{name} is missing positive runtime/memory fields")
    if value.get("persistent_gdn_base_immutable") is not True:
        raise AggregateError(f"{name} did not preserve persistent GDN base")
    rebind = value.get("query_linear_rebind", {})
    if rebind.get("verified") is not True or rebind.get("fallback_layers") != []:
        # JSON serializes the frozen empty tuple as an empty list.
        raise AggregateError(f"{name} did not verify every query-local GDN rebind")
    if rebind.get("request_count") != len(per_query):
        raise AggregateError(f"{name} query-local GDN request count mismatch")
    if value.get("full_document_staging_copy_nbytes") != 0:
        raise AggregateError(f"{name} retained a full-document request staging copy")
    if name.startswith("paged-"):
        if value.get("kernel_mode") != KERNEL_MODE:
            raise AggregateError(f"{name} kernel mode mismatch")
        if value.get("persistent_full_pages_shared") is not True:
            raise AggregateError(f"{name} did not share persistent pages")
        intercept = value.get("intercept", {})
        if intercept.get("verified") is not True:
            raise AggregateError(f"{name} paged intercept gate missing")
        if intercept.get("dense_fallback_calls") != 0:
            raise AggregateError(f"{name} used dense attention fallback")
        materialized = int(intercept.get("max_single_unpack_page_nbytes", 0))
        dense_full = int(intercept.get("max_dense_full_kv_nbytes", 0))
        if materialized <= 0 or dense_full <= 0 or materialized >= dense_full:
            raise AggregateError(f"{name} page materialization is not bounded")
        payload = int(value.get("persistent_paged_document_nbytes", 0))
        dense = int(value.get("dense_document_kv_nbytes", 0))
        if name == "paged-q16":
            if payload != dense:
                raise AggregateError("Q16 payload must equal dense BF16 K/V bytes")
        elif not 0 < payload < dense:
            raise AggregateError(f"{name} packed payload is not smaller than dense")
        if value.get("query_shared_document_nbytes") != payload:
            raise AggregateError(f"{name} request does not share page payload")


def aggregate(
    run_dir: Path,
    *,
    expected_shards: int,
    expected_data_sha256: str,
    expected_source_revision: str,
    expected_source_indices: tuple[int, ...],
    expected_workloads: int,
    expected_max_new_tokens: int,
) -> dict[str, Any]:
    paths = sorted(run_dir.glob("paged-real-shard-*.json"))
    if len(paths) != expected_shards:
        raise AggregateError(
            f"expected {expected_shards} shards, found {len(paths)}"
        )
    shards = [json.loads(path.read_text()) for path in paths]
    ranks = sorted(int(shard.get("rank", -1)) for shard in shards)
    if ranks != list(range(expected_shards)):
        raise AggregateError(f"rank coverage mismatch: {ranks}")
    if {int(shard.get("world_size", -1)) for shard in shards} != {
        expected_shards
    }:
        raise AggregateError("world_size differs from expected shards")
    if {shard.get("kernel_mode") for shard in shards} != {KERNEL_MODE}:
        raise AggregateError("top-level kernel_mode mismatch")
    if any(
        shard.get("production_ttft_optimization_claim_allowed") is not False
        for shard in shards
    ):
        raise AggregateError("reference shard incorrectly permits a TTFT claim")
    model_shas = {shard.get("model_manifest_sha256") for shard in shards}
    if len(model_shas) != 1 or None in model_shas:
        raise AggregateError("model manifest SHA mismatch")
    for shard in shards:
        if shard.get("status") != "completed_shard":
            raise AggregateError(f"rank {shard.get('rank')} shard is incomplete")
        metadata = shard.get("workload_metadata", {})
        if metadata.get("data_sha256") != expected_data_sha256:
            raise AggregateError("data SHA mismatch")
        if metadata.get("source_revisions") != [expected_source_revision]:
            raise AggregateError("source revision mismatch")
        if metadata.get("test_v2_consumed") is not False:
            raise AggregateError("test-v2 consumed or metadata missing")
        protocol = shard.get("protocol", {})
        if protocol.get("benchmark_bits") != [16, 8, 4]:
            raise AggregateError("benchmark bit matrix differs from Q16,Q8,Q4")
        if protocol.get("warmup_count") != 1:
            raise AggregateError("protocol did not freeze one warmup per config")
        _require_gate(shard)

    rows = [row for shard in shards for row in shard.get("rows", [])]
    if len(rows) != expected_workloads:
        raise AggregateError(
            f"expected {expected_workloads} workload rows, found {len(rows)}"
        )
    indices = sorted({int(row.get("source_index", -1)) for row in rows})
    if indices != sorted(set(expected_source_indices)):
        raise AggregateError(f"source index coverage mismatch: {indices}")
    if any(index >= 68 for index in indices):
        raise AggregateError("test-v2 source index detected")
    if len({row.get("workload_id") for row in rows}) != expected_workloads:
        raise AggregateError("workload IDs are duplicated")

    forward_order = CONFIGS
    reverse_order = tuple(reversed(CONFIGS))
    observed_orders: set[tuple[str, ...]] = set()
    for row in rows:
        rank = int(row.get("rank", -1))
        if rank < 0:
            raise AggregateError("workload row is missing rank")
        if "repeats the same frozen query" not in row.get(
            "multi_query_semantics", ""
        ):
            raise AggregateError("multi-query semantics are not explicit")
        measurements = row.get("measurements")
        if not isinstance(measurements, dict) or set(measurements) != set(CONFIGS):
            raise AggregateError("measurement config matrix is incomplete")
        order = tuple(row.get("measurement_order", ()))
        if order not in (forward_order, reverse_order):
            raise AggregateError("measurement order is not frozen AB/BA")
        expected_order = forward_order if rank % 2 == 0 else reverse_order
        if order != expected_order:
            raise AggregateError("measurement order does not match rank parity")
        observed_orders.add(order)
        for name in CONFIGS:
            _require_measurement(
                name, measurements[name], expected_max_new_tokens
            )
        paired = row.get("paired")
        if not isinstance(paired, dict) or set(paired) != set(PAGED_CONFIGS):
            raise AggregateError("paired comparison matrix is incomplete")
        for name in PAGED_CONFIGS:
            pair = paired[name]
            if not isinstance(pair.get("generated_tokens_exact"), bool):
                raise AggregateError(f"{name} token parity is missing")
            if name == "paged-q16" and pair["generated_tokens_exact"] is not True:
                raise AggregateError("Q16 generated tokens differ from dense-native")
            for field in (
                "persistent_total_resident_ratio_vs_stock",
                "multi_query_active_ratio_vs_stock",
                "cuda_peak_ratio_vs_stock",
                "nvml_peak_ratio_vs_stock",
                "ttft_ratio_vs_stock_reference_only",
                "tpot_ratio_vs_stock_reference_only",
            ):
                if float(pair.get(field, 0)) <= 0:
                    raise AggregateError(f"{name} paired ratio {field} is missing")
    if len(rows) > 1 and observed_orders != {forward_order, reverse_order}:
        raise AggregateError("paired benchmark did not cover both AB and BA orders")

    def median_config(config: str, field: str) -> float:
        return statistics.median(
            float(row["measurements"][config][field]) for row in rows
        )

    by_config = {}
    for name in CONFIGS:
        values = {
            "persistent_total_resident_nbytes": median_config(
                name, "persistent_total_resident_nbytes"
            ),
            "multi_query_active_total_resident_nbytes": median_config(
                name, "multi_query_active_total_resident_nbytes"
            ),
            "query_private_nbytes": median_config(name, "query_private_nbytes"),
            "cuda_peak_allocated_bytes": median_config(
                name, "cuda_peak_allocated_bytes"
            ),
            "nvml_sampled_peak_process_bytes": median_config(
                name, "nvml_sampled_peak_process_bytes"
            ),
            "ttft_seconds_reference_only": median_config(name, "ttft_seconds"),
            "median_tpot_seconds_reference_only": median_config(
                name, "median_tpot_seconds"
            ),
            "auditable_corpus_capacity_documents": median_config(
                name, "auditable_corpus_capacity_documents"
            ),
            "auditable_corpus_capacity_with_active_queries": median_config(
                name, "auditable_corpus_capacity_with_active_queries"
            ),
        }
        if name in PAGED_CONFIGS:
            values.update(
                {
                    "persistent_paged_document_nbytes": median_config(
                        name, "persistent_paged_document_nbytes"
                    ),
                    "dense_document_kv_nbytes": median_config(
                        name, "dense_document_kv_nbytes"
                    ),
                    "payload_compression_ratio": median_config(
                        name, "dense_document_kv_nbytes"
                    )
                    / median_config(name, "persistent_paged_document_nbytes"),
                }
            )
        by_config[name] = values

    paired_median = {
        name: {
            field: statistics.median(
                float(row["paired"][name][field]) for row in rows
            )
            for field in (
                "persistent_total_resident_ratio_vs_stock",
                "multi_query_active_ratio_vs_stock",
                "cuda_peak_ratio_vs_stock",
                "nvml_peak_ratio_vs_stock",
                "ttft_ratio_vs_stock_reference_only",
                "tpot_ratio_vs_stock_reference_only",
            )
        }
        | {
            "generated_token_exact_rate": statistics.fmean(
                bool(row["paired"][name]["generated_tokens_exact"])
                for row in rows
            )
        }
        for name in PAGED_CONFIGS
    }
    result = {
        "status": "passed",
        "kernel_mode": KERNEL_MODE,
        "production_ttft_optimization_claim_allowed": False,
        "claim_scope": (
            "stock-vs-native and Q16 same-caller correctness; paired dense-native "
            "vs Q16/Q8/Q4 persistent/active memory, capacity, and reference latency; "
            "not production TTFT optimization"
        ),
        "shards": expected_shards,
        "workloads": len(rows),
        "source_indices": indices,
        "data_sha256": expected_data_sha256,
        "source_revision": expected_source_revision,
        "model_manifest_sha256": next(iter(model_shas)),
        "test_v2_consumed": False,
        "gates": {
            "all_stock_vs_native_same_caller_passed": True,
            "all_stock_vs_q16_paged_same_caller_passed": True,
            "all_full_attention_intercepts_verified": True,
            "dense_fallback_calls": 0,
        },
        "median_by_config": by_config,
        "median_paired_vs_dense_native_functional": paired_median,
        "rows": rows,
    }
    atomic_json(run_dir / "paged-real-summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-shards", type=int, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-source-indices", type=int, nargs="+", required=True)
    parser.add_argument("--expected-workloads", type=int, required=True)
    parser.add_argument("--expected-max-new-tokens", type=int, required=True)
    args = parser.parse_args()
    result = aggregate(
        args.run_dir,
        expected_shards=args.expected_shards,
        expected_data_sha256=args.expected_data_sha256,
        expected_source_revision=args.expected_source_revision,
        expected_source_indices=tuple(args.expected_source_indices),
        expected_workloads=args.expected_workloads,
        expected_max_new_tokens=args.expected_max_new_tokens,
    )
    print(json.dumps(result["median_by_config"], sort_keys=True))


if __name__ == "__main__":
    main()
