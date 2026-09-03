#!/usr/bin/env python3
"""Recompute and render the paper's evidence-grounded figures and tables.

The script treats the eight resident-shard JSON files as the source of truth and
checks the aggregate summary against them.  It intentionally separates the
controlled full-attention pool, post-pack production peak, and maximum peak
across the recorded lifecycle phases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib as mpl
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


MIB = 2**20
GIB = 2**30
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT.parent / "results" / "gpu-qwen35-vllm-paged-multifork-resident-20260814a"

BLUE = "#4C78A8"
LIGHT_BLUE = "#DCEAF7"
ORANGE = "#F58518"
LIGHT_ORANGE = "#FCE3CE"
GREEN = "#3A923A"
LIGHT_GREEN = "#DDEEDB"
GRAY = "#6B7280"
LIGHT_GRAY = "#E5E7EB"
INK = "#1F2937"
PDF_METADATA = {
    "Title": "ForkAudit paper artifact",
    "Author": "Anonymous",
    "Creator": "generate_paper_artifacts.py",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def median_int(values: list[int]) -> int:
    value = statistics.median(values)
    require(float(value).is_integer(), f"non-integral byte median: {value}")
    return int(value)


def extract_metrics(summary_path: Path, shard_paths: list[Path]) -> dict[str, Any]:
    summary = load_json(summary_path)
    shards = [load_json(path) for path in shard_paths]

    require(summary["passed"] is True, "aggregate summary did not pass")
    require(len(shards) == 8, f"expected 8 shards, found {len(shards)}")
    ranks = sorted(shard["rank"] for shard in shards)
    require(ranks == list(range(8)), f"unexpected ranks: {ranks}")
    require(all(shard["passed"] is True for shard in shards), "a raw shard did not pass")

    ns = summary["resident_counts"]
    require(ns == [1, 2, 4, 8, 16, 32], f"unexpected resident counts: {ns}")
    require(all(shard["resident_counts"] == ns for shard in shards), "shard N grids differ")

    first = shards[0]
    config = first["static"]["protocol_config"]
    geometry = first["static"]["model_geometry"]
    versions = first["static"]["environment"]["observed_versions"]
    full_layers = geometry["full_attention_layer_indices"]
    require(geometry["num_hidden_layers"] == 40, "unexpected model depth")
    require(len(full_layers) == 10, "unexpected number of full-attention layers")
    require(geometry["linear_attention_layer_count"] == 30, "unexpected GDN layer count")
    require(config["max_new_tokens"] == 8, "unexpected generation length")
    require(config["pg19_document_tokens"] == 4095, "unexpected document length")
    require(config["page_size"] == 128, "unexpected page size")
    require(config["quantization"] == "Q16", "unexpected quantization label")
    require("KVQuantMode.NONE" in summary["kernel_descriptor"][2], "Q16 was not BF16 KV")

    capacity_by_n = {row["resident_count"]: row for row in summary["capacity_matrix"]}
    capacity_rows: list[dict[str, Any]] = []
    coverage = {
        "request_pairs": [],
        "logit_steps": [],
        "logical_kv_layer_digests": [],
        "gdn_request_digests": [],
        "fused_attention_calls": [],
    }

    aggregate = {
        "request_pairs": 0,
        "logit_steps": 0,
        "logical_kv_layer_digests": 0,
        "gdn_request_digests": 0,
        "gdn_tensors_covered": 0,
        "fused_attention_calls": 0,
        "dense_fallback_calls": 0,
        "full_kv_concatenations": 0,
        "materialized_attention_mask_nbytes": 0,
        "mask_validation_host_syncs": 0,
        "position_ids_validation_host_syncs": 0,
    }

    for n in ns:
        raw_rows = []
        n_counts = {key: 0 for key in coverage}
        for shard in shards:
            row = next(item for item in shard["rows"] if item["resident_count"] == n)
            raw_rows.append(row)
            parity = row["parity"]
            require(parity["passed"] is True, f"parity failed at rank={shard['rank']}, N={n}")
            require(parity["request_count"] == n, "parity request count mismatch")
            require(parity["all_request_token_trajectories_exact"] is True, "token mismatch")
            require(parity["all_request_full_vocab_step_logits_exact"] is True, "logit hash mismatch")
            require(
                parity["all_request_full_vocab_step_logits_runtime_torch_equal"] is True,
                "runtime torch.equal logit mismatch",
            )
            require(parity["all_request_logical_kv_exact"] is True, "logical KV mismatch")
            require(parity["all_request_gdn_state_exact"] is True, "GDN mismatch")

            fresh_traj = {x["request_index"]: x for x in row["fresh"]["generation"]["trajectories"]}
            reuse_traj = {x["request_index"]: x for x in row["reuse"]["generation"]["trajectories"]}
            fresh_kv = {x["request_index"]: x for x in row["fresh"]["request_logical_kv_after_generation"]}
            reuse_kv = {x["request_index"]: x for x in row["reuse"]["request_logical_kv_after_generation"]}
            fresh_gdn = {x["request_index"]: x for x in row["fresh"]["request_gdn_after_generation"]}
            reuse_gdn = {x["request_index"]: x for x in row["reuse"]["request_gdn_after_generation"]}
            require(sorted(fresh_traj) == list(range(n)), "fresh request set mismatch")
            require(sorted(reuse_traj) == list(range(n)), "reuse request set mismatch")

            for request_index in range(n):
                ft, rt = fresh_traj[request_index], reuse_traj[request_index]
                require(ft["generated_token_ids"] == rt["generated_token_ids"], "token trajectory mismatch")
                require(
                    ft["full_vocab_step_logit_sha256"] == rt["full_vocab_step_logit_sha256"],
                    "full-vocabulary step-logit digest mismatch",
                )
                require(len(ft["full_vocab_step_logit_sha256"]) == config["max_new_tokens"], "step count mismatch")
                require(fresh_kv[request_index]["layer_sha256"] == reuse_kv[request_index]["layer_sha256"], "KV digest mismatch")
                require(len(fresh_kv[request_index]["layer_sha256"]) == len(full_layers), "KV layer count mismatch")
                require(fresh_gdn[request_index]["sha256"] == reuse_gdn[request_index]["sha256"], "GDN digest mismatch")
                require(fresh_gdn[request_index]["tensor_count"] == 60, "unexpected GDN tensor count")

            n_counts["request_pairs"] += n
            n_counts["logit_steps"] += n * config["max_new_tokens"]
            n_counts["logical_kv_layer_digests"] += n * len(full_layers)
            n_counts["gdn_request_digests"] += n

            for arm in ("fresh", "reuse"):
                intercepts = row[arm]["intercepts"]
                require(len(intercepts) == n, "intercept ledger count mismatch")
                for ledger in intercepts:
                    require(ledger["verified"] is True, "unverified attention ledger")
                    require(ledger["total_calls"] == 80, "unexpected calls per request")
                    require(len(ledger["calls"]) == ledger["total_calls"], "call ledger length mismatch")
                    aggregate["fused_attention_calls"] += ledger["total_calls"]
                    aggregate["dense_fallback_calls"] += ledger["dense_fallback_calls"]
                    aggregate["full_kv_concatenations"] += ledger["full_kv_concatenations"]
                    n_counts["fused_attention_calls"] += ledger["total_calls"]
                    for call in ledger["calls"]:
                        require(call["fused_gpu_kernel_calls"] == 1, "non-fused attention call")
                        aggregate["materialized_attention_mask_nbytes"] += call["materialized_attention_mask_nbytes"]
                        aggregate["mask_validation_host_syncs"] += call["mask_validation_host_syncs"]
                        aggregate["position_ids_validation_host_syncs"] += call["position_ids_validation_host_syncs"]

        for key in coverage:
            coverage[key].append(n_counts[key])
            aggregate[key] += n_counts[key] if key != "fused_attention_calls" else 0
        aggregate["gdn_tensors_covered"] += n_counts["gdn_request_digests"] * 60

        fresh_pool_raw = []
        reuse_pool_raw = []
        fresh_production_peak_raw = []
        reuse_production_peak_raw = []
        fresh_lifecycle_peak_raw = []
        reuse_lifecycle_peak_raw = []
        for raw in raw_rows:
            ft = raw["fresh"]["storage_after_generation"]["totals"]
            rt = raw["reuse"]["storage_after_generation"]["totals"]
            fresh_pool_raw.append(
                ft["source_document_allocated_nbytes"]
                + ft["source_private_reservation_nbytes"]
                + ft["fresh_duplicate_document_allocated_nbytes"]
                + ft["fresh_duplicate_private_reservation_nbytes"]
            )
            reuse_pool_raw.append(rt["source_document_allocated_nbytes"] + rt["source_private_reservation_nbytes"])
            for arm, values in (
                ("fresh", fresh_production_peak_raw),
                ("reuse", reuse_production_peak_raw),
            ):
                arm_row = raw[arm]
                recomputed_production_peak = max(
                    arm_row["resident_setup"]["allocator_after"]["peak_allocated_bytes"],
                    arm_row["generation"]["production_allocator_before_exactness"]["peak_allocated_bytes"],
                )
                require(
                    recomputed_production_peak
                    == arm_row["setup_plus_generation"]["combined_absolute_peak_allocated_bytes"],
                    f"{arm} legacy combined production peak disagrees with phase replay",
                )
                values.append(recomputed_production_peak)
            for arm, values in (
                ("fresh", fresh_lifecycle_peak_raw),
                ("reuse", reuse_lifecycle_peak_raw),
            ):
                arm_row = raw[arm]
                values.append(
                    max(
                        arm_row["common_document_prefill"]["allocator_after"]["peak_allocated_bytes"],
                        arm_row["common_q16_pack"]["allocator_after"]["peak_allocated_bytes"],
                        arm_row["resident_setup"]["allocator_after"]["peak_allocated_bytes"],
                        arm_row["generation"]["production_allocator_before_exactness"]["peak_allocated_bytes"],
                    )
                )

        fresh_pool = median_int(fresh_pool_raw)
        reuse_pool = median_int(reuse_pool_raw)
        fresh_production_peak = median_int(fresh_production_peak_raw)
        reuse_production_peak = median_int(reuse_production_peak_raw)
        fresh_lifecycle_peak = median_int(fresh_lifecycle_peak_raw)
        reuse_lifecycle_peak = median_int(reuse_lifecycle_peak_raw)
        aggregate_row = capacity_by_n[n]
        require(
            fresh_production_peak == int(aggregate_row["fresh"]["production_absolute_peak_allocated_median_bytes"]),
            "fresh peak disagrees with aggregate summary",
        )
        require(
            reuse_production_peak == int(aggregate_row["reuse"]["production_absolute_peak_allocated_median_bytes"]),
            "reuse peak disagrees with aggregate summary",
        )
        expected_fresh_pool = (80 + 90 * n) * MIB
        expected_reuse_pool = (80 + 5 * n) * MIB
        require(fresh_pool == expected_fresh_pool, "fresh analytic pool formula mismatch")
        require(reuse_pool == expected_reuse_pool, "reuse analytic pool formula mismatch")
        production_delta_peak = fresh_production_peak - reuse_production_peak
        lifecycle_delta_peak = fresh_lifecycle_peak - reuse_lifecycle_peak
        capacity_rows.append(
            {
                "resident_count": n,
                "fresh_pool_bytes": fresh_pool,
                "reuse_pool_bytes": reuse_pool,
                "controlled_pool_saved_bytes": fresh_pool - reuse_pool,
                "fresh_post_pack_production_peak_allocated_bytes": fresh_production_peak,
                "reuse_post_pack_production_peak_allocated_bytes": reuse_production_peak,
                "post_pack_production_peak_delta_bytes": production_delta_peak,
                "post_pack_production_peak_delta_fraction_of_fresh": production_delta_peak / fresh_production_peak,
                "post_pack_production_peak_ratio_fresh_over_reuse": fresh_production_peak / reuse_production_peak,
                "fresh_full_recorded_lifecycle_peak_allocated_bytes": fresh_lifecycle_peak,
                "reuse_full_recorded_lifecycle_peak_allocated_bytes": reuse_lifecycle_peak,
                "full_recorded_lifecycle_peak_delta_bytes": lifecycle_delta_peak,
                "full_recorded_lifecycle_peak_delta_fraction_of_fresh": lifecycle_delta_peak / fresh_lifecycle_peak,
                "full_recorded_lifecycle_peak_ratio_fresh_over_reuse": fresh_lifecycle_peak / reuse_lifecycle_peak,
                "pool_ratio_fresh_over_reuse": fresh_pool / reuse_pool,
                "rank_values_identical": len(set(fresh_pool_raw + reuse_pool_raw)) <= 2
                and len(set(fresh_production_peak_raw)) == 1
                and len(set(reuse_production_peak_raw)) == 1
                and len(set(fresh_lifecycle_peak_raw)) == 1
                and len(set(reuse_lifecycle_peak_raw)) == 1,
            }
        )

    require(aggregate["request_pairs"] == 504, "unexpected request-pair total")
    require(aggregate["logit_steps"] == 4032, "unexpected logit-step total")
    require(aggregate["logical_kv_layer_digests"] == 5040, "unexpected logical-KV total")
    require(aggregate["gdn_request_digests"] == 504, "unexpected GDN digest total")
    require(aggregate["gdn_tensors_covered"] == 30240, "unexpected GDN tensor coverage")
    require(aggregate["fused_attention_calls"] == 80640, "unexpected fused call total")
    require(aggregate["dense_fallback_calls"] == 0, "dense fallback observed")
    require(aggregate["full_kv_concatenations"] == 0, "full KV concatenation observed")
    require(aggregate["materialized_attention_mask_nbytes"] == 0, "attention mask allocation observed")
    require(aggregate["mask_validation_host_syncs"] == 0, "mask-validation host sync observed")
    require(aggregate["position_ids_validation_host_syncs"] == 0, "position-validation host sync observed")
    require(summary["same_kernel_full_logit_token_logical_kv_gdn_exact_fraction"] == 1.0, "summary exact fraction mismatch")

    # Recompute the cross-N oracle from raw per-request evidence rather than
    # trusting the aggregate pass bit. Each request index uses a nested prefix
    # of the same frozen query bank within a rank.
    cross_n_replayed_per_arm = {"fresh": 0, "reuse": 0}
    for shard in shards:
        for arm in ("fresh", "reuse"):
            baseline: dict[int, dict[str, Any]] = {}
            for n in ns:
                row = next(item for item in shard["rows"] if item["resident_count"] == n)
                trajectories = {item["request_index"]: item for item in row[arm]["generation"]["trajectories"]}
                logical_kv = {item["request_index"]: item for item in row[arm]["request_logical_kv_after_generation"]}
                gdn = {item["request_index"]: item for item in row[arm]["request_gdn_after_generation"]}
                for request_index in range(n):
                    signature = {
                        "generated_token_ids": trajectories[request_index]["generated_token_ids"],
                        "full_vocab_step_logit_sha256": trajectories[request_index]["full_vocab_step_logit_sha256"],
                        "logical_kv_layer_sha256": logical_kv[request_index]["layer_sha256"],
                        "gdn_sha256": gdn[request_index]["sha256"],
                    }
                    if request_index in baseline:
                        require(
                            signature == baseline[request_index],
                            f"cross-N raw replay mismatch rank={shard['rank']} arm={arm} request={request_index} N={n}",
                        )
                        cross_n_replayed_per_arm[arm] += 1
                    else:
                        baseline[request_index] = signature
    require(cross_n_replayed_per_arm == {"fresh": 248, "reuse": 248}, "cross-N replay count drift")
    require(summary["cross_n_prefix_isolation_exact"] is True, "aggregate cross-N pass bit drift")

    # Nested prefix fan-outs do not give every unique query a non-vacuous
    # cross-N comparison: indices 16--31 first appear at N=32.  Record the
    # exact denominator rather than allowing the aggregate pass bit to imply
    # broader coverage.
    appearances_per_query = [sum(request_index < n for n in ns) for request_index in range(max(ns))]
    cross_n_eligible_per_rank = sum(count >= 2 for count in appearances_per_query)
    cross_n_singleton_per_rank = sum(count == 1 for count in appearances_per_query)
    cross_n_comparisons_per_rank = sum(max(0, count - 1) for count in appearances_per_query)
    require(cross_n_eligible_per_rank == 16, "unexpected cross-N eligible-query count")
    require(cross_n_singleton_per_rank == 16, "unexpected cross-N singleton-query count")
    require(cross_n_comparisons_per_rank == 31, "unexpected cross-N comparison count")

    source_root = summary_path.parent
    evidence_namespace = (
        "anonymous_derivative"
        if isinstance(summary.get("anonymous_derivative"), dict)
        else "original_run"
    )

    return {
        "schema_version": "1.0.0",
        "evidence_namespace": evidence_namespace,
        "arm_key_aliases": {
            "fresh": "Source+Materialize",
            "reuse": "Shared-Document",
        },
        "source": {
            "summary": summary_path.relative_to(source_root).as_posix(),
            "summary_sha256": sha256(summary_path),
            "shards": [
                {"path": path.relative_to(source_root).as_posix(), "sha256": sha256(path)}
                for path in shard_paths
            ],
        },
        "protocol": {
            "rank_count": len(shards),
            "resident_counts": ns,
            "execution_order": config["execution_order"],
            "document_tokens": config["pg19_document_tokens"],
            "document_tail_tokens": first["document_tail_tokens"],
            "query_tokens": config["pg19_query_tokens"],
            "generated_tokens": config["max_new_tokens"],
            "appended_tokens_per_request": config["pg19_query_tokens"] + config["max_new_tokens"] - 1,
            "page_size": config["page_size"],
            "batch_per_request": config["batch_per_request"],
            "full_attention_layer_indices": full_layers,
            "gdn_layer_count": geometry["linear_attention_layer_count"],
            "versions": versions,
            "kernel": first["kernel_identity"]["module"] + "." + first["kernel_identity"]["qualname"],
            "kv_storage": "BF16 (artifact label Q16; KVQuantMode.NONE)",
            "scheduler": "round-major serial execution on one CUDA stream; all request objects strongly resident",
        },
        "capacity": capacity_rows,
        "coverage_by_n": coverage,
        "coverage_totals": aggregate,
        "arm_specific_model_steps": 2 * aggregate["logit_steps"],
        "cross_n_coverage": {
            "eligible_unique_queries": len(shards) * cross_n_eligible_per_rank,
            "singleton_unique_queries": len(shards) * cross_n_singleton_per_rank,
            "larger_fanout_comparisons": len(shards) * cross_n_comparisons_per_rank,
            "raw_replayed_comparisons_per_arm": cross_n_replayed_per_arm,
        },
        "cross_n_prefix_isolation_exact": True,
        "interpretation_boundaries": [
            "The controlled pool is source-retaining: it is not an optimized production full-copy baseline.",
            "Post-pack production and full recorded-lifecycle PyTorch peaks are distinct estimands; neither is NVML usage or service capacity.",
            "The run archived digests and runtime equality booleans, not raw logits or state tensors.",
            "All ranks yielded identical reported allocator values, so no stochastic uncertainty interval is claimed.",
            "The Q16 artifact label denotes BF16 KV storage with KVQuantMode.NONE, not low-bit KV quantization.",
            "Cross-N prefix consistency is non-vacuous for 128 of 256 unique queries; the other 128 occur only at N=32.",
        ],
    }


def style_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": GRAY,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def rounded_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str,
                facecolor: str, edgecolor: str, fontsize: float = 7.2, linewidth: float = 1.0) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = GRAY,
          linestyle: str = "-", mutation_scale: float = 8) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=0.9,
            linestyle=linestyle,
            color=color,
            shrinkA=1,
            shrinkB=1,
        )
    )


def render_ownership_contract(output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.25), constrained_layout=True)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    left, right = axes
    left.set_title("(a) Source+Materialize control", loc="left", fontweight="bold")
    rounded_box(left, (0.06, 0.76), 0.88, 0.13, "Common source remains resident\n80 MiB document KV + 5N MiB reservations", LIGHT_GRAY, GRAY)
    request_y = [0.52, 0.32, 0.12]
    labels = ["Request 1", "Request 2", r"Request $N$"]
    for y, label in zip(request_y, labels):
        rounded_box(left, (0.08, y), 0.29, 0.12, label, "white", GRAY)
        rounded_box(left, (0.43, y), 0.30, 0.12, "Full document\nKV copy (80 MiB)", LIGHT_BLUE, BLUE, 6.7)
        rounded_box(left, (0.78, y), 0.16, 0.12, "Private\n5 MiB", LIGHT_ORANGE, ORANGE, 6.7)
        arrow(left, (0.37, y + 0.06), (0.43, y + 0.06))
        arrow(left, (0.73, y + 0.06), (0.78, y + 0.06))
    left.text(0.50, 0.015, r"Controlled full-attention pool: $80 + 90N$ MiB", ha="center", fontsize=7.6, fontweight="bold")

    right.set_title("(b) Shared-Document layout", loc="left", fontweight="bold")
    rounded_box(right, (0.05, 0.76), 0.56, 0.13, "Immutable document KV pages\nshared by all requests (80 MiB)", LIGHT_BLUE, BLUE)
    rounded_box(right, (0.66, 0.76), 0.29, 0.13, "Immutable GDN\nprefix state", LIGHT_GREEN, GREEN)
    for y, label in zip(request_y, labels):
        rounded_box(right, (0.07, y), 0.25, 0.12, label, "white", GRAY)
        rounded_box(right, (0.38, y), 0.25, 0.12, "Private KV\ntail + append", LIGHT_ORANGE, ORANGE, 6.7)
        rounded_box(right, (0.69, y), 0.25, 0.12, "Request-local GDN\nfunctional rebind", LIGHT_GREEN, GREEN, 6.4)
        arrow(right, (0.25, 0.76), (0.49, y + 0.12), BLUE)
        arrow(right, (0.80, 0.76), (0.82, y + 0.12), GREEN)
        arrow(right, (0.32, y + 0.06), (0.38, y + 0.06))
    right.text(0.50, 0.015, r"Controlled full-attention pool: $80 + 5N$ MiB", ha="center", fontsize=7.6, fontweight="bold")
    fig.text(0.5, -0.012, "Only full-attention KV ownership differs between arms; GDN handling is identical.", ha="center", fontsize=7.2, color=GRAY)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02, metadata=PDF_METADATA)
    plt.close(fig)


def render_memory_denominators(metrics: dict[str, Any], output: Path) -> None:
    rows = metrics["capacity"]
    ns = [row["resident_count"] for row in rows]
    fresh_pool = [row["fresh_pool_bytes"] / MIB for row in rows]
    reuse_pool = [row["reuse_pool_bytes"] / MIB for row in rows]

    fig, (ax_curve, ax_ratio) = plt.subplots(1, 2, figsize=(7.15, 2.75), gridspec_kw={"width_ratios": [1.35, 1]}, constrained_layout=True)
    x = list(range(len(ns)))
    ax_curve.plot(x, fresh_pool, marker="o", color=ORANGE, linewidth=1.8, label=r"Source+Materialize: $80+90N$")
    ax_curve.plot(x, reuse_pool, marker="s", color=BLUE, linewidth=1.8, label=r"Shared-Document: $80+5N$")
    ax_curve.fill_between(x, reuse_pool, fresh_pool, color=ORANGE, alpha=0.12, label=r"Controlled saving: $85N$")
    ax_curve.set_xticks(x, [str(n) for n in ns])
    ax_curve.set_xlabel("Simultaneously resident request objects, $N$")
    ax_curve.set_ylabel("Controlled full-attention pool (MiB)")
    ax_curve.set_ylim(0, max(fresh_pool) * 1.08)
    ax_curve.grid(axis="y", color=LIGHT_GRAY, linewidth=0.7)
    ax_curve.legend(frameon=False, loc="upper left")
    ax_curve.set_title("(a) Deterministic page accounting", loc="left", fontweight="bold")
    ax_curve.annotate(
        "2,720 MiB gap at $N=32$",
        xy=(x[-1], (fresh_pool[-1] + reuse_pool[-1]) / 2),
        xytext=(x[-1] - 2.2, 1950),
        arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 0.8},
        fontsize=7.2,
    )

    last = rows[-1]
    ratios = [
        last["pool_ratio_fresh_over_reuse"],
        last["post_pack_production_peak_ratio_fresh_over_reuse"],
        last["full_recorded_lifecycle_peak_ratio_fresh_over_reuse"],
    ]
    labels = [
        "Controlled\nKV pool",
        "Post-pack production\nPyTorch peak",
        "Full recorded lifecycle\nPyTorch peak",
    ]
    colors = [ORANGE, BLUE, GREEN]
    y = [2, 1, 0]
    ax_ratio.barh(y, [ratio - 1 for ratio in ratios], left=1, color=colors, height=0.48)
    ax_ratio.axvline(1, color=INK, linewidth=0.8)
    ax_ratio.set_yticks(y, labels)
    ax_ratio.set_xlim(1, 13.2)
    ax_ratio.set_xlabel("Source+Materialize / Shared-Document at $N=32$")
    ax_ratio.grid(axis="x", color=LIGHT_GRAY, linewidth=0.7)
    ax_ratio.set_title("(b) The denominator changes the headline", loc="left", fontweight="bold")
    annotations = [
        f"{ratios[0]:.2f}x\n2,960 vs 240 MiB",
        f"{ratios[1]:.4f}x\n{last['fresh_post_pack_production_peak_allocated_bytes']/GIB:.2f} vs "
        f"{last['reuse_post_pack_production_peak_allocated_bytes']/GIB:.2f} GiB",
        f"{ratios[2]:.4f}x\n{last['fresh_full_recorded_lifecycle_peak_allocated_bytes']/GIB:.2f} vs "
        f"{last['reuse_full_recorded_lifecycle_peak_allocated_bytes']/GIB:.2f} GiB",
    ]
    for yi, ratio, label in zip(y, ratios, annotations):
        ax_ratio.text(min(ratio + 0.18, 12.45), yi, label, va="center", fontsize=7.4, fontweight="bold")
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02, metadata=PDF_METADATA)
    plt.close(fig)


def render_exactness_lattice(metrics: dict[str, Any], output: Path) -> None:
    ns = metrics["protocol"]["resident_counts"]
    coverage = metrics["coverage_by_n"]
    rows = [
        ("Token trajectories", coverage["request_pairs"]),
        ("Paired full-vocabulary logits", coverage["logit_steps"]),
        ("Logical KV (10 layers)", coverage["logical_kv_layer_digests"]),
        ("GDN state (60 tensors/request)", coverage["gdn_request_digests"]),
        ("Attention callable contract", coverage["fused_attention_calls"]),
    ]
    matrix = [[1 for _ in ns] for _ in rows]
    cmap = mpl.colors.ListedColormap([LIGHT_GREEN])

    fig, ax = plt.subplots(figsize=(7.15, 2.65), constrained_layout=True)
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ns)), [f"$N={n}$" for n in ns])
    ax.set_yticks(range(len(rows)), [name for name, _ in rows])
    ax.set_xlabel("Resident fan-out (eight independent PG-19 books/ranks per column)")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("white")
        spine.set_linewidth(1.0)
    for i, (_, counts) in enumerate(rows):
        for j, count in enumerate(counts):
            ax.text(j, i, f"{count:,}\npass", ha="center", va="center", color="#1F5F1F", fontsize=7.1, fontweight="bold")
    ax.set_xticks([x - 0.5 for x in range(1, len(ns))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(rows))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    totals = metrics["coverage_totals"]
    fig.text(
        0.5,
        -0.02,
        f"Totals: {totals['request_pairs']:,} request pairs; {totals['logit_steps']:,} paired logit comparisons; "
        f"{totals['logical_kv_layer_digests']:,} KV layer digests; {totals['gdn_request_digests']:,} GDN digests; "
        f"{totals['fused_attention_calls']:,} attention-call records. All compared artifacts passed.",
        ha="center",
        fontsize=7.1,
        color=GRAY,
    )
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02, metadata=PDF_METADATA)
    plt.close(fig)


def render_protocol_table(metrics: dict[str, Any], output: Path) -> None:
    p = metrics["protocol"]
    layers = ", ".join(str(x) for x in p["full_attention_layer_indices"])
    version_text = (
        f"PyTorch {p['versions']['torch']}; Transformers {p['versions']['transformers']}; "
        f"vLLM {p['versions']['vllm']}; Triton {p['versions']['triton']}"
    )
    lines = [
        r"\begin{table}[H]",
        r"\caption{Frozen protocol for the multi-resident case study. All values are replayed from E-MF-SHARDS.}",
        r"\label{tab:protocol}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}p{0.24\columnwidth}p{0.70\columnwidth}@{}}",
        r"\toprule",
        r"Component & Frozen setting \\",
        r"\midrule",
        r"Model & Qwen3.5-35B-A3B: 40 layers; 10 full-attention layers "
        + f"({layers}) and 30 GDN layers. "
        + r"\\",
        r"KV path & vLLM 0.26 Triton \texttt{unified\_attention}; BF16 KV (artifact label Q16, \texttt{KVQuantMode.NONE}); page size 128; batch size 1. \\",
        r"Data & PG-19 train only; eight books (one per rank); 4,095-token document window; per-rank bank of 32 distinct, non-overlapping 32-token queries. \\",
        r"Generation & Eight argmax tokens; 39 KV tokens appended per request because the final generated token is not fed back; document tail is 127 tokens. \\",
        r"Fan-out & $N\in\{1,2,4,8,16,32\}$; execution order $1,32,2,16,4,8$. \\",
        r"Residency & All request objects remain strongly referenced; round-major serial execution on one CUDA stream (not concurrent kernels). \\",
        r"Hardware & Eight independent H20-3e GPUs (143,771 MiB each); one rank and one PG-19 book per GPU. \\",
        r"Software & " + version_text.replace("_", r"\_") + r". \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def render_capacity_table(metrics: dict[str, Any], output: Path) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\caption{Memory scaling under three explicitly named estimands. The controlled pool retains the common source in both arms and is therefore a controlled contrast, not an optimized production full-copy baseline. Post-pack and full recorded-lifecycle PyTorch peaks are allocator counters, not NVML usage or service capacity.}",
        r"\label{tab:capacity}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\begin{tabular}{@{}r rr rr rr@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Controlled pool (MiB)} & \multicolumn{2}{c}{Post-pack prod. peak (GiB)} & \multicolumn{2}{c}{Full lifecycle peak (GiB)} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"$N$ & Materialize & Shared & Materialize & Shared & Materialize & Shared \\",
        r"\midrule",
    ]
    for row in metrics["capacity"]:
        lines.append(
            f"{row['resident_count']} & {row['fresh_pool_bytes']/MIB:.0f} & {row['reuse_pool_bytes']/MIB:.0f} & "
            f"{row['fresh_post_pack_production_peak_allocated_bytes']/GIB:.3f} & "
            f"{row['reuse_post_pack_production_peak_allocated_bytes']/GIB:.3f} & "
            f"{row['fresh_full_recorded_lifecycle_peak_allocated_bytes']/GIB:.3f} & "
            f"{row['reuse_full_recorded_lifecycle_peak_allocated_bytes']/GIB:.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{1mm}",
            r"\parbox{0.97\textwidth}{\footnotesize The source-retaining analytic pool is exactly $80+90N$ MiB for Source+Materialize and $80+5N$ MiB for Shared-Document. The post-pack production peak is the maximum of resident setup and generation after the phase reset. The full recorded-lifecycle peak is the maximum over prefill, Q16 pack, resident setup, and generation. Values were identical across all eight ranks; this deterministic run does not support a stochastic confidence interval.}",
            r"\end{table}",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="Raw result directory")
    parser.add_argument("--output-root", type=Path, default=ROOT, help="Paper workspace root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = args.results.resolve()
    output_root = args.output_root.resolve()
    summary_path = results / "multifork-resident-summary.json"
    shard_paths = sorted((results / "resident-shards").glob("multifork-resident-shard-*.json"))
    figures = output_root / "figures"
    tables = output_root / "tables"
    generated = output_root / "generated"
    for directory in (figures, tables, generated):
        directory.mkdir(parents=True, exist_ok=True)

    metrics = extract_metrics(summary_path, shard_paths)
    metrics_path = generated / "artifact_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    style_matplotlib()
    render_ownership_contract(figures / "ownership_contract.pdf")
    render_memory_denominators(metrics, figures / "memory_denominators.pdf")
    render_exactness_lattice(metrics, figures / "exactness_lattice.pdf")
    render_protocol_table(metrics, tables / "protocol_table.tex")
    render_capacity_table(metrics, tables / "capacity_table.tex")

    outputs = [
        metrics_path,
        figures / "ownership_contract.pdf",
        figures / "memory_denominators.pdf",
        figures / "exactness_lattice.pdf",
        tables / "protocol_table.tex",
        tables / "capacity_table.tex",
    ]
    print(json.dumps({str(path.relative_to(output_root)): sha256(path) for path in outputs}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
