#!/usr/bin/env python3
"""Pre-output/terminal static authority for affected-only HYPIC RW-D5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from rwd5_model_asset_snapshot import snapshot as model_asset_snapshot
from rwd5_model_asset_snapshot import validate_snapshot as validate_model_asset_snapshot

from build_hypic_formal_static import (
    DATA_SHA256,
    HYPIC_COMMIT,
    MODEL_ARTIFACT_LEDGER_SHA256,
    MODEL_LEDGER_SHA256,
    StaticError,
    atomic_json,
    build_environment_ledger,
    build_source_ledger,
    canonical_bytes,
    parse_model_ledger,
    require,
    sha256_file,
)


MODES = ["prefix_cache", "transition_rope_recompute"]
PLATFORM_CONFIGURATION_SHA256 = "c8fd81cf248587d98c880ada006c393aed37ef31de154a88488bf31e7f33da80"
LIVE_DEBUG_REMOTE_ROOT = (
    "/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/"
    "runs/qcomem/hypic-component-dtype-debug-trial1879097-20260822j/"
)
LIVE_DEBUG_MIRROR_MANIFEST_SHA256 = (
    "59530c0c8bc10cedbf4b0bde51d04e5490adeaf369e8738d9df363fc83941026"
)
LIVE_DEBUG_FILES = {
    "COMPLETED_DEBUG_ONLY",
    "all-debug-artifacts.sha256",
    "commands/prefix_cache.txt",
    "commands/transition_rope_recompute.txt",
    "debug-receipts/prefix_cache-rank-0.json",
    "debug-receipts/prefix_cache-server-info.json",
    "debug-receipts/prefix_cache-validation.json",
    "debug-receipts/transition_rope_recompute-rank-0.json",
    "debug-receipts/transition_rope_recompute-server-info.json",
    "debug-receipts/transition_rope_recompute-validation.json",
    "logs/prefix_cache.log",
    "logs/transition_rope_recompute.log",
    "nvidia-smi-after.txt",
    "nvidia-smi-before.txt",
    "run-summaries/prefix_cache-rank-0.json",
    "run-summaries/transition_rope_recompute-rank-0.json",
    "targets/prefix_cache-rank-0.json",
    "targets/transition_rope_recompute-rank-0.json",
}
ALLOCATOR_DEBUG_REMOTE_ROOT = (
    "/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/"
    "runs/qcomem/hypic-mamba-allocator-debug-20260822d/"
)
ALLOCATOR_DEBUG_MIRROR_MANIFEST_SHA256 = (
    "d57e3e5436f9b7b586a3788a1f3205d9ee3e4f6403496edc3205c2927a842f7e"
)
ALLOCATOR_DEBUG_FILES = {
    "COMPLETED_DEBUG_ONLY",
    "all-debug-artifacts.sha256",
    "commands/server.txt",
    "debug-receipts/server-info.json",
    "debug-receipts/terminal-binding.json",
    "debug-receipts/transition_rope_recompute-rank-0.json",
    "debug-receipts/transition_rope_recompute-validation.json",
    "instrumentation-overlay.diff",
    "logs/server.log",
    "nvidia-smi-after.txt",
    "nvidia-smi-before.txt",
    "run-summaries/transition_rope_recompute-rank-0.json",
    "targets/transition_rope_recompute-rank-0.json",
}


def _sha_manifest(path: Path, *, relative: bool,
                  remote_root: str = LIVE_DEBUG_REMOTE_ROOT) -> dict[str, str]:
    require(path.is_file(), f"manifest exists: {path}")
    rows: dict[str, str] = {}
    for line in path.read_text().splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2 and len(fields[0]) == 64, "manifest row")
        digest, raw_name = fields
        require(all(value in "0123456789abcdef" for value in digest), "manifest digest")
        name = raw_name.strip()
        if relative:
            require(name.startswith("./") and not name.startswith("../"), "relative manifest path")
            name = name[2:]
            candidate = Path(name)
            require(not candidate.is_absolute() and ".." not in candidate.parts, "confined manifest path")
        else:
            require(name.startswith(remote_root), "remote debug root")
            name = name[len(remote_root):]
            require(name and not Path(name).is_absolute() and ".." not in Path(name).parts, "remote manifest path")
        require(name not in rows, "duplicate manifest path")
        rows[name] = digest
    return rows


def _is_c_contiguous(shape: list[int], stride: list[int]) -> bool:
    expected = 1
    for size, observed in zip(reversed(shape), reversed(stride)):
        if size != 1 and observed != expected:
            return False
        expected *= size
    return True


def validate_live_component_debug(
    root: Path, *, expected_manifest_sha256: str, recurrent_layers: int
) -> dict[str, Any]:
    """Bind the successful debug-only GPU inventory into formal preregistration."""
    require(
        expected_manifest_sha256 == LIVE_DEBUG_MIRROR_MANIFEST_SHA256,
        "frozen live debug manifest identity",
    )
    manifest = root / "mirror-files.sha256"
    require(sha256_file(manifest) == expected_manifest_sha256, "live debug mirror manifest SHA")
    mirror_rows = _sha_manifest(manifest, relative=True)
    require(set(mirror_rows) == LIVE_DEBUG_FILES, "exact live debug mirror file set")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "mirror-files.sha256"
    }
    require(actual == LIVE_DEBUG_FILES, "live debug mirror tree")
    for name, digest in mirror_rows.items():
        require(sha256_file(root / name) == digest, f"live debug mirror digest: {name}")
    require((root / "COMPLETED_DEBUG_ONLY").read_bytes() == b"", "debug terminal marker")
    require(not (root / "FAILED_DEBUG_ONLY").exists(), "debug failed marker absent")

    remote_ledger = root / "all-debug-artifacts.sha256"
    remote_rows = _sha_manifest(remote_ledger, relative=False)
    expected_remote = LIVE_DEBUG_FILES - {"COMPLETED_DEBUG_ONLY", "all-debug-artifacts.sha256"}
    require(set(remote_rows) == expected_remote, "remote debug artifact file set")
    for name, digest in remote_rows.items():
        require(digest == mirror_rows[name] == sha256_file(root / name), f"remote/local debug digest: {name}")

    expected_components = {
        "prefix_cache": {
            "conv[0]": ("torch.bfloat16", 2, 4),
            "temporal": ("torch.float32", 4, 5),
        },
        "transition_rope_recompute": {
            "conv[0]": ("torch.bfloat16", 2, 4),
            "temporal": ("torch.float32", 4, 5),
            "transition": ("torch.float32", 4, 5),
            "conv_tails[0]": ("torch.bfloat16", 2, 4),
        },
    }
    mode_summaries = {}
    for mode in MODES:
        raw_path = root / f"debug-receipts/{mode}-rank-0.json"
        validation_path = root / f"debug-receipts/{mode}-validation.json"
        run_path = root / f"run-summaries/{mode}-rank-0.json"
        target_path = root / f"targets/{mode}-rank-0.json"
        raw = json.loads(raw_path.read_text())
        validation = json.loads(validation_path.read_text())
        run = json.loads(run_path.read_text())
        expected_cache = "MambaRadixCache" if mode == "prefix_cache" else "PICache"
        require(
            raw.get("schema") == "hypic-rwd5-component-dtype-debug-v1"
            and raw.get("status") == "debug_only_not_formal_evidence"
            and raw.get("official_commit") == HYPIC_COMMIT
            and raw.get("mode") == mode
            and raw.get("tree_cache_class") == expected_cache
            and raw.get("mamba_pool_class") == "MambaPool"
            and raw.get("formal_receipt_emitted") is False,
            f"live debug raw identity: {mode}",
        )
        require(
            raw.get("runtime_environment") == {
                "SGLANG_MAMBA_CONV_DTYPE": "bfloat16",
                "SGLANG_MAMBA_SSM_DTYPE": "float32",
            },
            f"live debug environment: {mode}",
        )
        capacity = int(raw["mamba_capacity_axis"])
        require(capacity == int(raw["mamba_allocator_size"]) + 1, f"live debug capacity: {mode}")
        components = raw.get("components")
        require(isinstance(components, dict) and set(components) == set(expected_components[mode]), f"live component keys: {mode}")
        for name, (dtype, element_size, rank) in expected_components[mode].items():
            row = components[name]
            require(set(row) == {"dtype", "element_size", "shape", "stride", "device", "c_contiguous"}, f"live component fields: {mode}/{name}")
            shape = row["shape"]
            stride = row["stride"]
            require(
                row["dtype"] == dtype and int(row["element_size"]) == element_size
                and row["device"] == "cuda:0" and row["c_contiguous"] is True
                and isinstance(shape, list) and isinstance(stride, list)
                and len(shape) == len(stride) == rank
                and all(isinstance(value, int) and value > 0 for value in shape + stride)
                and shape[0] == recurrent_layers and shape[1] == capacity
                and _is_c_contiguous(shape, stride),
                f"live component contract: {mode}/{name}",
            )
        if mode == "transition_rope_recompute":
            require(components["conv_tails[0]"] == components["conv[0]"], "live conv tail topology")
            require(components["transition"] == components["temporal"], "live transition topology")

        raw_sha = sha256_file(raw_path)
        require(
            validation.get("schema") == "hypic-rwd5-component-dtype-debug-validation-v1"
            and validation.get("status") == "passed_exact_live_component_contract"
            and validation.get("official_commit") == HYPIC_COMMIT
            and validation.get("mode") == mode
            and validation.get("paper_evidence") is False
            and validation.get("debug_receipt_sha256") == raw_sha
            and validation.get("expected_recurrent_layers") == recurrent_layers
            and validation.get("mamba_capacity_axis") == capacity
            and validation.get("components") == components,
            f"live validation binding: {mode}",
        )
        expected_run_validation = dict(validation)
        expected_run_validation.pop("debug_receipt_sha256")
        require(
            run.get("schema") == "hypic-rwd5-component-dtype-debug-run-v1"
            and run.get("status") == "completed_debug_only_not_formal_evidence"
            and run.get("official_commit") == HYPIC_COMMIT
            and run.get("mode") == mode and run.get("rank") == 0
            and run.get("workload_id") == "qasper-6"
            and run.get("paper_evidence") is False
            and run.get("raw_formal_receipt_emitted") is False
            and run.get("store_formal_receipt_emitted") is False
            and run.get("debug_receipt_sha256") == raw_sha
            and run.get("target_sha256") == sha256_file(target_path)
            and run.get("validation") == expected_run_validation,
            f"live debug run binding: {mode}",
        )
        mode_summaries[mode] = {
            "raw_receipt_sha256": raw_sha,
            "validation_receipt_sha256": sha256_file(validation_path),
            "run_summary_sha256": sha256_file(run_path),
            "target_sha256": sha256_file(target_path),
            "cache_class": expected_cache,
            "mamba_capacity_axis": capacity,
            "components": {
                name: {
                    "dtype": row["dtype"], "element_size": row["element_size"],
                    "shape": row["shape"], "stride": row["stride"],
                }
                for name, row in components.items()
            },
        }
    return {
        "schema": "hypic-rwd5-live-component-debug-binding-v1",
        "status": "passed_debug_only_not_paper_evidence",
        "paper_evidence": False,
        "platform_job_id": 247512,
        "platform_trial_id": 1879097,
        "official_commit": HYPIC_COMMIT,
        "mirror_manifest_sha256": expected_manifest_sha256,
        "remote_artifact_ledger_sha256": sha256_file(remote_ledger),
        "terminal_completed_debug_only": True,
        "formal_receipts_emitted": 0,
        "modes": mode_summaries,
    }


def validate_live_allocator_debug(
    root: Path, *, expected_manifest_sha256: str,
    provenance_path: Path | None = None,
    launch_plan_path: Path | None = None,
    frozen_debug_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Bind the read-only D allocator observation without treating it as a result."""
    require(
        expected_manifest_sha256 == ALLOCATOR_DEBUG_MIRROR_MANIFEST_SHA256,
        "frozen allocator debug manifest identity",
    )
    manifest = root / "mirror-files.sha256"
    require(sha256_file(manifest) == expected_manifest_sha256,
            "allocator debug mirror manifest SHA")
    mirror_rows = _sha_manifest(manifest, relative=True)
    require(set(mirror_rows) == ALLOCATOR_DEBUG_FILES,
            "exact allocator debug mirror file set")
    actual = {
        str(path.relative_to(root)) for path in root.rglob("*")
        if path.is_file() and path.name != "mirror-files.sha256"
    }
    require(actual == ALLOCATOR_DEBUG_FILES, "allocator debug mirror tree")
    for name, digest in mirror_rows.items():
        require(sha256_file(root / name) == digest,
                f"allocator debug mirror digest: {name}")
    require((root / "COMPLETED_DEBUG_ONLY").read_bytes() == b"",
            "allocator debug terminal marker")
    require(not (root / "FAILED_DEBUG_ONLY").exists(),
            "allocator debug failure marker absent")
    formal = root / "formal-receipts-disabled"
    require(formal.is_dir() and not any(formal.iterdir()),
            "allocator debug formal directory truly empty")

    remote_ledger = root / "all-debug-artifacts.sha256"
    remote_rows = _sha_manifest(
        remote_ledger, relative=False, remote_root=ALLOCATOR_DEBUG_REMOTE_ROOT
    )
    expected_remote = ALLOCATOR_DEBUG_FILES - {
        "COMPLETED_DEBUG_ONLY", "all-debug-artifacts.sha256"
    }
    require(set(remote_rows) == expected_remote,
            "allocator remote debug artifact file set")
    for name, digest in remote_rows.items():
        require(digest == mirror_rows[name] == sha256_file(root / name),
                f"allocator remote/local debug digest: {name}")

    raw_path = root / "debug-receipts/transition_rope_recompute-rank-0.json"
    validation_path = root / "debug-receipts/transition_rope_recompute-validation.json"
    run_path = root / "run-summaries/transition_rope_recompute-rank-0.json"
    target_path = root / "targets/transition_rope_recompute-rank-0.json"
    terminal_path = root / "debug-receipts/terminal-binding.json"
    raw = json.loads(raw_path.read_text())
    validation = json.loads(validation_path.read_text())
    run = json.loads(run_path.read_text())
    terminal = json.loads(terminal_path.read_text())
    target = json.loads(target_path.read_text())
    require(
        raw.get("schema") == "hypic-rwd5-mamba-allocator-debug-v1"
        and raw.get("status") == "read_only_post_prime_allocator_captured"
        and raw.get("official_commit") == HYPIC_COMMIT
        and raw.get("paper_evidence") is False
        and raw.get("formal_receipt_emitted") is False
        and raw.get("mutation_performed") is False,
        "allocator debug raw identity",
    )
    observed_target = raw["target"]
    require(
        observed_target["mode"] == "transition_rope_recompute"
        and int(observed_target["rank"]) == 0
        and observed_target["snapshot_id"]
        == "allocator-debug-transition_rope_recompute-rank-0"
        and observed_target["workload_id"] == "qasper-6"
        and observed_target["target_file_sha256"] == sha256_file(target_path)
        and observed_target["document_tokens"] == 3997
        and observed_target["document_token_sha256"]
        == "92a411191dc079487c025bad42156794dd25cd0eb79af5dd9303f238b0fad136"
        and observed_target["segment_token_counts"] == [85, 3912]
        and observed_target["segment_hash_hex"]
        == ["a2a3b5997b8e671fcaaed8b7343a63e9", "6ca2a71d3acca268b1d08113dea93f9a"],
        "allocator debug exact target",
    )
    require(
        target.get("mode") == "transition_rope_recompute"
        and int(target.get("rank")) == 0
        and target.get("workload_id") == "qasper-6"
        and target.get("document_token_sha256") == observed_target["document_token_sha256"]
        and len(target.get("document_token_ids", [])) == 3997
        and [len(segment) for segment in target.get("segment_token_ids", [])] == [85, 3912],
        "allocator target file semantics",
    )

    allocator = raw["allocator"]
    raw_slots = allocator["raw_free_slots"]
    require(
        allocator["class"] == "MambaSlotAllocator"
        and allocator["module"] == "sglang.srt.mem_cache.allocator.mamba"
        and allocator["device"] == "cuda"
        and allocator["tree_cache_alias"] is True
        and allocator["req_pool_alias"] is True
        and allocator["alloc_iter_is_none"] is True
        and allocator["dict_keys"] == ["_alloc_iter", "device", "free_slots", "size"]
        and int(allocator["size"]) == 183
        and int(allocator["available_size"]) == len(raw_slots) == 182
        and all(isinstance(slot, int) and not isinstance(slot, bool) for slot in raw_slots)
        and set(raw_slots).issubset(set(range(1, 184))),
        "allocator exact live structure",
    )
    counts = Counter(raw_slots)
    positions: dict[int, list[int]] = {}
    for index, slot in enumerate(raw_slots):
        positions.setdefault(slot, []).append(index)
    duplicates = [
        {"slot": slot, "count": counts[slot], "positions": positions[slot]}
        for slot in sorted(counts) if counts[slot] > 1
    ]
    missing = sorted(set(range(1, 184)) - set(raw_slots))
    require(
        duplicates == allocator["duplicates"]
        == [{"slot": 3, "count": 2, "positions": [168, 177]}]
        and allocator["duplicate_excess_count"] == 1
        and allocator["raw_count"] == 182
        and allocator["unique_count"] == len(counts) == 181
        and allocator["out_of_domain"] == []
        and allocator["missing_from_raw_unique_domain"] == missing == [14, 15],
        "allocator exact duplicate multiset and unique domain",
    )
    tensor = allocator["free_slots_tensor"]
    require(
        tensor["tensor_name"] == "mamba_allocator.free_slots"
        and tensor["dtype"] == "torch.int64" and tensor["element_size"] == 8
        and tensor["shape"] == [182] and tensor["stride"] == [1]
        and tensor["device"] == "cuda:0"
        and tensor["selection"] == {"kind": "whole_tensor"}
        and tensor["storage_offset_elements"] == 0
        and tensor["range_bytes"] == tensor["storage_nbytes"] == 1456
        and tensor["byte_start"] == 0 and tensor["byte_end"] == 1456
        and tensor["tensor_data_ptr"] == tensor["storage_data_ptr"]
        and tensor["absolute_byte_start"] == tensor["storage_data_ptr"]
        and tensor["absolute_byte_end"] == tensor["storage_data_ptr"] + 1456
        and tensor["storage_id"] == hashlib.sha256(
            f"cuda:0:{tensor['storage_data_ptr']}:1456".encode()
        ).hexdigest(),
        "allocator free tensor identity",
    )
    cache = raw["cache"]
    selected = [int(slot) for slot in cache["target_mamba_state_slots"]]
    require(
        cache["class"] == "PICache"
        and cache["module"] == "sglang.srt.pic.picache"
        and cache["entry_count"] == len(cache["entries"]) == 2
        and all(entry["is_target_segment"] is True and entry["lock_ref"] == 0
                for entry in cache["entries"])
        and sorted(selected) == [14, 15]
        and cache["target_slots_distinct"] is True
        and cache["target_slots_present_in_raw_free_slots"] == {"14": [], "15": []}
        and set(selected).isdisjoint(raw_slots)
        and sorted(entry["mamba_state_slot"] for entry in cache["entries"]) == [14, 15]
        and {entry["segment_hash_hex"] for entry in cache["entries"]}
        == set(observed_target["segment_hash_hex"]),
        "allocator exact target cache ownership",
    )
    raw_sha = sha256_file(raw_path)
    expected_validation = {
        "schema": "hypic-rwd5-mamba-allocator-debug-validation-v1",
        "status": "passed_exact_duplicate_representation_capture",
        "official_commit": HYPIC_COMMIT,
        "paper_evidence": False,
        "workload_id": "qasper-6",
        "allocator_debug_receipt_sha256": raw_sha,
        "allocator_size": 183,
        "raw_count": 182,
        "unique_count": 181,
        "duplicates": duplicates,
        "duplicate_excess_count": 1,
        "target_mamba_state_slots": [14, 15],
        "target_slots_present_in_raw_free_slots": {"14": [], "15": []},
    }
    require(validation == expected_validation, "allocator validation receipt binding")
    require(
        run["schema"] == "hypic-rwd5-mamba-allocator-debug-run-v1"
        and run["status"] == "completed_debug_only_not_formal_evidence"
        and run["official_commit"] == HYPIC_COMMIT
        and run["mode"] == "transition_rope_recompute"
        and run["rank"] == 0 and run["workload_id"] == "qasper-6"
        and run["paper_evidence"] is False and run["formal_receipts_emitted"] == 0
        and run["allocator_debug_receipt_sha256"] == raw_sha
        and run["target_sha256"] == sha256_file(target_path)
        and run["validation"] == {key: value for key, value in expected_validation.items()
                                  if key != "allocator_debug_receipt_sha256"},
        "allocator debug run binding",
    )
    require(
        terminal == {
            "schema": "hypic-rwd5-mamba-allocator-debug-terminal-v1",
            "status": "passed_exact_debug_only_terminal_binding",
            "mode": "transition_rope_recompute", "rank": 0,
            "workload_id": "qasper-6", "paper_evidence": False,
            "raw_receipt_sha256": raw_sha,
            "validation_receipt_sha256": sha256_file(validation_path),
            "run_summary_sha256": sha256_file(run_path),
            "target_sha256": sha256_file(target_path),
        },
        "allocator terminal binding",
    )
    provenance_summary = {}
    if provenance_path is not None:
        require(
            launch_plan_path is not None and frozen_debug_manifest_path is not None,
            "allocator debug provenance companions",
        )
        provenance = json.loads(provenance_path.read_text())
        launch_plan = json.loads(launch_plan_path.read_text())
        require(
            sha256_file(launch_plan_path)
            == provenance["frozen_d_debug_launch_plan_sha256"]
            == "3cf1a833be25e6428c007c60dc753c014387c7b61383abcaa56ef1a2c54482e1"
            and sha256_file(frozen_debug_manifest_path)
            == provenance["frozen_d_debug_manifest_sha256"]
            == "c2fea8fabd51e1a97064d3efe871370884604341e1749fa57bfde653597cbd9e"
            and launch_plan["debug_platform_trial_id"] == 1879456
            and launch_plan["fresh_output_root"] == ALLOCATOR_DEBUG_REMOTE_ROOT.rstrip("/")
            and "platform_job_id" not in launch_plan
            and provenance["schema"] == "hypic-rwd5-allocator-debug-d-provenance-v1"
            and provenance["status"] == "bound_debug_only_execution_authority"
            and provenance["paper_evidence"] is False
            and provenance["formal_receipts_emitted"] == 0
            and provenance["platform_job_id"] == 247574
            and provenance["platform_trial_id"] == 1879456
            and provenance["remote_run_dir"] == ALLOCATOR_DEBUG_REMOTE_ROOT.rstrip("/")
            and provenance["platform_job_authority"]
            == "external execution/submission receipt and post-run platform query supplied by the run coordinator"
            and provenance["launch_plan_job_id_limitation"]
            == "the frozen D debug launch plan records Trial 1879456 and the exact run directory but does not itself contain Job 247574"
            and provenance["live_mirror_manifest_sha256"] == expected_manifest_sha256
            and provenance["remote_artifact_ledger_sha256"] == sha256_file(remote_ledger)
            and provenance["raw_allocator_receipt_sha256"] == raw_sha
            and provenance["validation_receipt_sha256"] == sha256_file(validation_path)
            and provenance["terminal_binding_sha256"] == sha256_file(terminal_path)
            and provenance["global_allocator_correctness_claimed"] is False,
            "allocator debug execution/freeze provenance",
        )
        provenance_summary = {
            "provenance_receipt_sha256": sha256_file(provenance_path),
            "frozen_d_debug_manifest_sha256": sha256_file(frozen_debug_manifest_path),
            "frozen_d_debug_launch_plan_sha256": sha256_file(launch_plan_path),
            "launch_plan_contains_platform_job_id": False,
            "platform_job_authority": provenance["platform_job_authority"],
        }
    return {
        "schema": "hypic-rwd5-live-allocator-debug-binding-v1",
        "status": "passed_debug_only_local_physical_ownership_authority",
        "paper_evidence": False,
        "platform_job_id": 247574,
        "platform_trial_id": 1879456,
        "remote_run_dir": ALLOCATOR_DEBUG_REMOTE_ROOT.rstrip("/"),
        "official_commit": HYPIC_COMMIT,
        "mirror_manifest_sha256": expected_manifest_sha256,
        "remote_artifact_ledger_sha256": sha256_file(remote_ledger),
        "allocator_size": 183,
        "raw_free_count": 182,
        "unique_free_count": 181,
        "duplicates": duplicates,
        "duplicate_excess_count": 1,
        "unique_allocated_domain": [14, 15],
        "target_mamba_state_slots": [14, 15],
        "formal_receipts_emitted": 0,
        "global_allocator_correctness_claimed": False,
        **provenance_summary,
    }


def model_storage_contract(
    model: Path, live_debug_validation: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Derive the exact, model-specific RW-D5 tensor topology."""
    config_path = model / "config.json"
    require(config_path.is_file(), "model config.json")
    raw = json.loads(config_path.read_text())
    text = raw.get("text_config", raw)
    require(isinstance(text, dict), "text model config")
    model_type = str(text.get("model_type", ""))
    require(model_type in {"qwen3_5_text", "qwen3_5_moe_text"}, "Qwen3.5 text config")
    num_layers = int(text["num_hidden_layers"])
    require(num_layers > 0, "num_hidden_layers")
    layer_types = text.get("layer_types")
    if layer_types is None:
        interval = int(text["full_attention_interval"])
        require(interval > 0, "full_attention_interval")
        layer_types = [
            "attention" if (index + 1) % interval == 0 else "linear_attention"
            for index in range(num_layers)
        ]
    require(isinstance(layer_types, list) and len(layer_types) == num_layers, "layer_types")
    normalized = [str(value) for value in layer_types]
    full_ids = [
        index
        for index, value in enumerate(normalized)
        if value in {"attention", "full_attention"}
    ]
    recurrent_ids = [
        index for index, value in enumerate(normalized) if value == "linear_attention"
    ]
    require(full_ids and recurrent_ids, "hybrid layer partition")
    require(len(full_ids) + len(recurrent_ids) == num_layers, "known layer types only")
    return {
        "schema": "hypic-rwd5-model-storage-contract-v3",
        "config_sha256": sha256_file(config_path),
        "model_type": model_type,
        "num_hidden_layers": num_layers,
        "full_attention_layer_ids": full_ids,
        "recurrent_layer_ids": recurrent_ids,
        "full_attention_layer_count": len(full_ids),
        "recurrent_layer_count": len(recurrent_ids),
        "conv_tensor_count": 1,
        "temporal_tensor_count": 1,
        "kv_dtype": "torch.bfloat16",
        "mamba_component_dtypes": {
            "conv": "torch.bfloat16",
            "temporal": "torch.float32",
            "transition": "torch.float32",
            "conv_tails": "torch.bfloat16",
        },
        "dtype_authority": {
            "runtime_environment": {
                "SGLANG_MAMBA_CONV_DTYPE": "bfloat16",
                "SGLANG_MAMBA_SSM_DTYPE": "float32",
            },
            "official_pool_rule": "conv and conv_tails allocate cache_params.dtype.conv; temporal and transition allocate cache_params.dtype.temporal",
            "legacy_unified_dtype_forbidden": True,
            **({"live_debug_validation": live_debug_validation} if live_debug_validation else {}),
        },
        "kv_layout": "nhd",
        "kv_slot_axis": 0,
        "mamba_layer_axis": 0,
        "mamba_slot_axis": 1,
        "page_size": 1,
        "enable_int8_mamba_checkpoint": False,
        "mode_components": {
            "prefix_cache": {
                "transition_tensor_count": 0,
                "conv_tails_tensor_count": 0,
            },
            "transition_rope_recompute": {
                "transition_tensor_count": 1,
                "conv_tails_tensor_count": 1,
            },
        },
    }


def overlay_material(args: argparse.Namespace) -> tuple[dict[str, Any], bytes]:
    expected_status = [
        " M python/sglang/srt/managers/scheduler.py",
        " M python/sglang/srt/mem_cache/common.py",
        "?? python/sglang/srt/retained_state_receipt.py",
    ]
    status = subprocess.check_output(
        [
            "git", "-C", str(args.instrumented_repo), "status",
            "--porcelain=v1", "--untracked-files=all",
        ],
        text=True,
    ).splitlines()
    require(sorted(status) == sorted(expected_status), f"instrumented overlay status: {status!r}")
    diff = subprocess.check_output(
        [
            "git", "-C", str(args.instrumented_repo), "diff", "--binary",
            "--no-ext-diff", "--full-index", "--",
            "python/sglang/srt/managers/scheduler.py",
            "python/sglang/srt/mem_cache/common.py",
        ]
    )
    require(diff, "canonical tracked overlay diff")
    module = args.instrumented_repo / "python/sglang/srt/retained_state_receipt.py"
    ledger = {
        "schema": "hypic-rwd5-instrumentation-overlay-v2",
        "base_commit": HYPIC_COMMIT,
        "porcelain_v1": sorted(status),
        "tracked_files": [
            "python/sglang/srt/managers/scheduler.py",
            "python/sglang/srt/mem_cache/common.py",
        ],
        "untracked_files": ["python/sglang/srt/retained_state_receipt.py"],
        "canonical_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "receipt_module_sha256": sha256_file(module),
        "no_other_tracked_or_untracked": True,
    }
    return ledger, diff


def code_row(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing code file: {path}")
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def parse_canonical_freeze_manifest(raw: bytes) -> list[tuple[str, str]]:
    require(raw.endswith(b"\n"), "freeze manifest terminal newline")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise StaticError("freeze manifest UTF-8") from exc
    seen: set[str] = set()
    rows: list[tuple[str, str]] = []
    for line in text.splitlines():
        parts = line.split("  ")
        require(len(parts) == 2, "freeze manifest row shape")
        digest, marked_path = parts
        require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            "freeze manifest digest",
        )
        require(marked_path.startswith("./"), "freeze manifest relative marker")
        relative = marked_path[2:]
        components = relative.split("/")
        require(
            relative
            and not relative.startswith("/")
            and "//" not in relative
            and all(component not in ("", ".", "..") for component in components)
            and relative not in seen,
            "freeze manifest canonical unique path",
        )
        seen.add(relative)
        rows.append((digest, relative))
    require(seen, "freeze manifest nonempty")
    return rows


def _after_freeze_manifest_capture_for_test() -> None:
    """Deterministic race hook; immutable production code leaves it a no-op."""


def _hash_open_regular_file(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"manifest member regular file: {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        stable = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_uid,
            value.st_gid,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        require(
            stable(before) == stable(after) == stable(named),
            f"manifest member identity stable while hashing: {path}",
        )
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def capture_and_verify_freeze_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), "freeze manifest regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_mode, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "freeze manifest FD stable while capturing",
        )
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    capture_sha256 = hashlib.sha256(raw).hexdigest()
    require(capture_sha256 == expected_sha256, "external frozen manifest SHA")
    rows = parse_canonical_freeze_manifest(raw)
    _after_freeze_manifest_capture_for_test()
    for member_sha256, relative in rows:
        require(
            _hash_open_regular_file(path.parent / relative) == member_sha256,
            f"frozen manifest member SHA: {relative}",
        )
    return {
        "capture_sha256": capture_sha256,
        "member_count": len(rows),
        "all_entries_verified_from_same_capture": True,
    }


def launch_authority_description(manifest_member_count: int) -> str:
    require(manifest_member_count > 0, "positive freeze manifest member count")
    return (
        "the only authorized external entry point is the frozen safe wrapper; "
        "it captures exact manifest/STOP bytes once per checkpoint, verifies "
        "external hashes, the exact canonical "
        f"{manifest_member_count}-member set derived from the verified manifest, "
        "and every row before PRE, again after import probes before the "
        "captured-manifest-bound POST helper FD, and once more after POST before "
        "a captured-manifest-bound internal launcher FD exec; it also verifies "
        "all 14+9 model payload bytes and stable asset semantics, fixes a "
        "root-owned non-writable cwd, proves exact frozen unittest and SGLang "
        "import origins, and pins every launcher input under env -i"
    )


def validate_platform_execution_authority(receipt_path: Path, pid1_environ_path: Path) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text())
    require(
        receipt.get("schema") == "hypic-rwd5-platform-execution-authority-v1"
        and receipt.get("freeze") == "Z"
        and receipt.get("platform_job_id") == 247699
        and receipt.get("platform_trial_id") == 1880085
        and receipt.get("trial_name") == "liuhanzuo-qcomem-hypic-store-rwd5-node-recovery-20260822x"
        and receipt.get("status_at_preregistration") == "Uncommit"
        and receipt.get("queue_id") == 408
        and receipt.get("queue_name") == "RL_main"
        and receipt.get("cloud_id") == "6"
        and receipt.get("cluster_id") == 53
        and receipt.get("resource_package_id") == 183
        and receipt.get("worker_num") == 1
        and receipt.get("worker_gpu") == 8
        and receipt.get("gpu_type") == "NVIDIA H20-3e"
        and receipt.get("user_env") == "QCOMEM_DEBUG_NODE=1,QCOMEM_DEBUG_SCOPE=ROUND27_HYPIC_STORE_FORMAL_W"
        and receipt.get("runtime_status_required_before_staging") == "Running"
        and receipt.get("paper_evidence") is False
        and receipt.get("formal_cells_before_z_green") == 0,
        "exact Z platform execution authority receipt",
    )
    configuration = {
        "cloud_id": receipt["cloud_id"],
        "cluster_id": receipt["cluster_id"],
        "command": receipt["command"],
        "image": receipt["image"],
        "resource_package_id": receipt["resource_package_id"],
        "user_env": receipt["user_env"],
        "worker_gpu": receipt["worker_gpu"],
        "worker_num": receipt["worker_num"],
    }
    configuration_sha256 = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    require(
        configuration_sha256 == PLATFORM_CONFIGURATION_SHA256
        and receipt.get("configuration_canonical_sha256") == PLATFORM_CONFIGURATION_SHA256,
        "Z platform configuration hash",
    )
    raw = pid1_environ_path.read_bytes()
    require(raw.endswith(b"\0"), "pid1 environment terminal NUL")
    relevant: dict[str, list[str]] = {
        "QS_JOB_ID": [], "QS_TRIAL_ID": [], "QCOMEM_DEBUG_SCOPE": []
    }
    for item in raw.split(b"\0"):
        if not item:
            continue
        require(b"=" in item, "pid1 environment entry shape")
        key_raw, value_raw = item.split(b"=", 1)
        key = key_raw.decode("utf-8", errors="strict")
        if key in relevant:
            relevant[key].append(value_raw.decode("utf-8", errors="strict"))
    expected = receipt["runtime_pid1_environ_required"]
    require(
        all(relevant[key] == [expected[key]] for key in relevant)
        and expected == {
            "QS_JOB_ID": "247699",
            "QS_TRIAL_ID": "1880085",
            "QCOMEM_DEBUG_SCOPE": "ROUND27_HYPIC_STORE_FORMAL_W",
        },
        "platform-owned pid1 execution identity",
    )
    require(
        os.environ.get("RWD5_PLATFORM_JOB_ID") == "247699"
        and os.environ.get("RWD5_PLATFORM_TRIAL_ID") == "1880085",
        "safe-wrapper pinned platform identity",
    )
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "configuration_sha256": configuration_sha256,
        "platform_job_id": 247699,
        "platform_trial_id": 1880085,
        "pid1_environ_path": str(pid1_environ_path),
        "pid1_identity_verified": True,
    }


def materialize(args: argparse.Namespace, verify_model_bytes: bool) -> dict[str, bytes]:
    require(sha256_file(args.data) == DATA_SHA256, "data SHA")
    parse_model_ledger(
        args.model_weight_ledger,
        args.model,
        expected_raw_sha256=MODEL_LEDGER_SHA256,
        expected_count=14,
        label="model weight",
        verify_bytes=verify_model_bytes,
    )
    require(args.asset_observation.is_file(), "asset observation exists")
    asset_observation = validate_model_asset_snapshot(
        json.loads(args.asset_observation.read_text()), model_root=args.model
    )
    require(
        asset_observation == model_asset_snapshot(args.model),
        "asset observation still matches exact current same-preflight identity",
    )
    platform_authority = validate_platform_execution_authority(
        args.platform_authority_receipt, args.pid1_environ
    )
    retired_w = json.loads(args.retired_w_trial_receipt.read_text())
    require(
        retired_w.get("schema") == "hypic-rwd5-external-manual-stop-v1"
        and retired_w.get("platform_job_id") == 247699
        and retired_w.get("platform_trial_id") == 1879843
        and retired_w.get("scheduler_condition") == "manual_stop"
        and retired_w.get("stop_actor") is None
        and retired_w.get("stop_issued_by_root_agent") is False
        and retired_w.get("stop_issued_by_r27_hypic_store_bytes_agent") is False
        and retired_w.get("w_bundle_staged") is False
        and retired_w.get("safe_entry_invoked") is False
        and retired_w.get("formal_cells_started") == 0
        and retired_w.get("paper_evidence") is False,
        "exact externally stopped unfrozen W trial",
    )
    invalid_t = json.loads(args.invalid_t_receipt.read_text())
    require(
        invalid_t.get("schema") == "hypic-rwd5-invalid-pre-science-attempt-v1"
        and invalid_t.get("freeze") == "T"
        and invalid_t.get("platform_job_id") == 247574
        and invalid_t.get("platform_trial_id") == 1879456
        and invalid_t.get("status") == "invalid_pre_science_asset_authority_rejection"
        and invalid_t.get("paper_evidence") is False
        and invalid_t.get("store_result") is False
        and invalid_t.get("completed_cells") == 0
        and invalid_t.get("gpu_server_started") is False
        and invalid_t.get("exit_code") == 1
        and invalid_t.get("later_platform_event", {}).get("shell_exit_code") == 137
        and invalid_t.get("z_recovery_boundary", {}).get("platform_job_id") == 247699
        and invalid_t.get("z_recovery_boundary", {}).get("platform_trial_id") == 1880085
        and invalid_t.get("z_recovery_boundary", {}).get("queue_id") == 408
        and invalid_t.get("z_recovery_boundary", {}).get("configuration_canonical_sha256") == PLATFORM_CONFIGURATION_SHA256
        and invalid_t.get("retired_unfrozen_w_trial", {}).get("platform_trial_id") == 1879843
        and invalid_t.get("retired_unfrozen_w_trial", {}).get("stop_actor_known") is False
        and invalid_t.get("retired_unfrozen_w_trial", {}).get("bundle_staged") is False
        and invalid_t.get("retired_unfrozen_w_trial", {}).get("paper_evidence") is False
        and invalid_t.get("retired_v_execution_node", {}).get("platform_job_id") == 247668
        and invalid_t.get("retired_v_execution_node", {}).get("platform_trial_id") == 1879689
        and invalid_t.get("retired_v_execution_node", {}).get("pod_created") is False
        and invalid_t.get("retired_v_execution_node", {}).get("gpu_or_science_execution") is False
        and invalid_t.get("retired_v_freeze", {}).get("manifest_sha256") == "19ea81e461ec00e9e2412fc67a8523a86f805cd3faff7b213438901af2a38108"
        and invalid_t.get("retired_v_freeze", {}).get("gpu_or_science_execution") is False
        and invalid_t.get("retired_w_freeze", {}).get("manifest_sha256") == "227380b6e6bf3103f6698090629fcb9cf3db1d344e3f7dfdac858ea58880654e"
        and invalid_t.get("retired_w_freeze", {}).get("stop_sha256") == "109bad7529a73775dee4abf338e667d3cd7835f2abc3a1d2864cdf98da86e2e1"
        and invalid_t.get("retired_w_freeze", {}).get("tree_sha256") == "fdf7e5bb1961eee838cd197fe46cce143994923981b2dd2967759dba7a616764"
        and invalid_t.get("retired_w_freeze", {}).get("gpu_or_science_execution") is False
        and invalid_t.get("retired_x_freeze", {}).get("manifest_sha256") == "f42eb1b394b588fc1cf22753b100a25e6a03147924ed3fad39b2d88832d003a0"
        and invalid_t.get("retired_x_freeze", {}).get("stop_sha256") == "f6894a398ab1744500a1e82cbb2c44be78b59fb1781affdf930ca9be25ce2130"
        and invalid_t.get("retired_x_freeze", {}).get("tree_sha256") == "9e8395c5667320fc6e03a46387276e63d954c751c9b4a218eda5c653fb1504c1"
        and invalid_t.get("retired_x_freeze", {}).get("gpu_or_science_execution") is False
        and invalid_t.get("retired_y_freeze", {}).get("manifest_sha256") == "a36734ad22953df8d9b126f19535fe39532f47ec03a011a75059fe91fec33cb9"
        and invalid_t.get("retired_y_freeze", {}).get("stop_sha256") == "82b48f5ae571c81c6fed4d71e49b9ff15a1e94e26b5c8a1c600ee7e66a52935d"
        and invalid_t.get("retired_y_freeze", {}).get("tree_sha256") == "40cd97df272764c330d6ca9a7bca14f23197d84df0e46ba2c871792c1c333fd2"
        and invalid_t.get("retired_y_freeze", {}).get("gpu_or_science_execution") is False
        and invalid_t.get("retired_y_freeze", {}).get("remote_staging") is False
        and invalid_t.get("retired_empty_u_node", {}).get("platform_trial_id") == 1879665
        and invalid_t.get("retired_empty_u_node", {}).get("gpu_or_science_execution") is False
        and invalid_t.get("retired_u_freeze", {}).get("gpu_or_science_execution") is False,
        "exact invalid T attempt identity and recovery separation",
    )
    parse_model_ledger(
        args.model_artifact_ledger,
        args.model,
        expected_raw_sha256=MODEL_ARTIFACT_LEDGER_SHA256,
        expected_count=9,
        label="model artifact",
        verify_bytes=verify_model_bytes,
    )
    source = build_source_ledger(args.official_repo)
    environment = build_environment_ledger()
    subprocess.check_call(
        ["git", "-C", str(args.official_repo), "apply", "--check", str(args.patch)]
    )
    subprocess.check_call(
        ["git", "-C", str(args.instrumented_repo), "apply", "--reverse", "--check", str(args.patch)]
    )
    installed_module = args.instrumented_repo / "python/sglang/srt/retained_state_receipt.py"
    require(
        installed_module.is_file()
        and sha256_file(installed_module) == sha256_file(args.receipt_module),
        "installed receipt module",
    )
    require(args.freeze_manifest.is_file(), "external frozen manifest")
    freeze_manifest_capture = capture_and_verify_freeze_manifest(
        args.freeze_manifest, args.expected_freeze_manifest_sha256
    )
    freeze_manifest_member_count = int(freeze_manifest_capture["member_count"])
    storage_contract = model_storage_contract(args.model)
    live_debug_validation = validate_live_component_debug(
        args.live_debug_root,
        expected_manifest_sha256=args.expected_live_debug_manifest_sha256,
        recurrent_layers=int(storage_contract["recurrent_layer_count"]),
    )
    allocator_debug_validation = validate_live_allocator_debug(
        args.allocator_debug_root,
        expected_manifest_sha256=args.expected_allocator_debug_manifest_sha256,
        provenance_path=args.allocator_debug_provenance,
        launch_plan_path=args.allocator_debug_launch_plan,
        frozen_debug_manifest_path=args.allocator_debug_freeze_manifest,
    )
    storage_contract = model_storage_contract(args.model, live_debug_validation)
    overlay, overlay_diff = overlay_material(args)
    code = {
        name: code_row(getattr(args, name))
        for name in (
            "client",
            "formal_helper",
            "formal_static_helper",
            "serving_helper",
            "receipt_module",
            "patch",
            "receipt_test",
            "inherited_test",
            "inherited_launcher",
            "replay",
            "launcher",
            "safe_wrapper",
            "safe_cwd_guard",
            "model_asset_snapshot",
            "static_builder",
        )
    }
    prereg = {
        "schema": "hypic-rwd5-retained-state-preregistration-v2",
        "status": "frozen_before_outputs",
        "research_question": "What physical tensor payload is owned by the cached document in Prefix Cache and HYPIC under the completed same-protocol run?",
        "official_repository": "https://github.com/redai-infra/HYPIC",
        "official_commit": HYPIC_COMMIT,
        "official_source_ledger_sha256": hashlib.sha256(canonical_bytes(source)).hexdigest(),
        "environment_ledger_sha256": hashlib.sha256(canonical_bytes(environment)).hexdigest(),
        "instrumentation": {
            "read_only": True,
            "official_checkout_remains_clean": True,
            "temporary_copy_base_commit": HYPIC_COMMIT,
            "patch_sha256": sha256_file(args.patch),
            "module_sha256": sha256_file(args.receipt_module),
            "forbidden": ["NVML", "process allocation delta", "pool capacity delta"],
            "overlay": overlay,
        },
        "external_freeze": {
            "manifest_path": str(args.freeze_manifest),
            "manifest_sha256": freeze_manifest_capture["capture_sha256"],
            "manifest_member_count": freeze_manifest_member_count,
            "all_entries_verified": freeze_manifest_capture["all_entries_verified_from_same_capture"],
            "single_captured_byte_stream": True,
        },
        "invalid_t_attempt_authority": {
            "path": str(args.invalid_t_receipt),
            "sha256": sha256_file(args.invalid_t_receipt),
            "receipt": invalid_t,
        },
        "platform_execution_authority": platform_authority,
        "retired_unfrozen_w_trial_authority": {
            "path": str(args.retired_w_trial_receipt),
            "sha256": sha256_file(args.retired_w_trial_receipt),
            "receipt": retired_w,
        },
        "code": code,
        "model": {
            "path": str(args.model),
            "weight_ledger_raw_sha256": MODEL_LEDGER_SHA256,
            "artifact_ledger_raw_sha256": MODEL_ARTIFACT_LEDGER_SHA256,
            "config_sha256": storage_contract["config_sha256"],
            "storage_contract": storage_contract,
            "storage_contract_sha256": hashlib.sha256(canonical_bytes(storage_contract)).hexdigest(),
            "asset_identity": {
                "observation_path": str(args.asset_observation),
                "observation_sha256": sha256_file(args.asset_observation),
                "observation": asset_observation,
                "cross_node_authority": "regular non-symlink, exact SHA and size, mode 0444, uid/gid 0, plus verified 14+9 ledger bytes",
                "same_preflight_gate": "exact O_NOFOLLOW open/fstat/hash/lstat observation equality before and after all wrapper authority probes, and rechecked during static materialization",
                "old_node_inode_device_and_timestamps_are_not_authority": True,
            },
        },
        "data": {"path": str(args.data), "sha256": DATA_SHA256, "frozen_rows": 8},
        "design": {
            "modes": MODES,
            "cells": 16,
            "excluded_modes": [
                "full_recompute",
                "CoMem",
                "RR2",
                "GDN",
                "vLLM",
                "SGLang serving controls",
            ],
            "model": "Qwen3.5-35B-A3B",
            "hardware": "one H20-3e per frozen row",
            "tp_size": 1,
            "input_cap": 4096,
            "decode": "greedy max 32 tokens",
            "hypic_seam_tokens": 8,
            "server_readiness": "each exact mode/rank/base-URL/server-PID cell binds model_info success to evidence-bearing short /server_info polls; 300 second total deadline",
            "failure_lifecycle": "set -E plus idempotent EXIT/ERR/INT/TERM process-group cleanup; TERM is followed by bounded liveness polling, KILL escalation, final reap, and an explicit PID plus process-group absence assertion before the next mode or COMPLETED; unsuccessful exit removes COMPLETED and writes FAILED before cleanup",
            "launch_authority": launch_authority_description(freeze_manifest_member_count),
            "frozen_preflight_tests": "the formal launcher executes the frozen focused receipt/replay suite and the frozen inherited same-protocol suite before preregistration or GPU servers",
            "terminal_resource_gate": "after bounded PID/PGID cleanup, all eight frozen GPU UUIDs must report zero compute applications and zero MiB used, and no matching SGLang server/scheduler or formal client process may remain; this is checked before the artifact ledger and again immediately before COMPLETED",
            "snapshot_timing": "after formal prime completes and before measured query begins",
            "payload_denominator": "exact target-entry-owned physical tensor-range union, overlap-aware by backing storage; allocator duplicate bookkeeping is excluded from the byte denominator and reported separately",
            "payload_components": [
                "full-attention key",
                "full-attention value",
                "conv",
                "temporal",
                "transition exactly once for HYPIC and absent for Prefix Cache",
                "conv_tails exactly once for HYPIC and absent for Prefix Cache",
            ],
            "metadata": "reported separately and excluded from Store MiB",
            "allocator_claim_boundary": "HYPIC raw free-slot multiset and duplicate fingerprint are preserved; Store validity is local to exact target entries and selected physical slots; global allocator correctness and runtime safety are not claimed",
            "hypic_pre_free_gate": "unique allocated physical Mamba domain equals the exact two target-entry slots; raw duplicate multiset is retained as an anomaly",
            "terminal_gate": "flush removes exact target entries, returns selected slots into the unique full physical domain, and preserves the original duplicate fingerprint without migration or growth",
        },
        "acceptance": {
            "all_16_cells": True,
            "cache_hit_coverage": True,
            "range_inside_backing_storage": True,
            "blind_union_replay_exact": True,
            "terminal_ownership_removal": True,
            "median_only_after_all_rows_pass": True,
            "server_info_ready_before_server_receipt": True,
            "failure_terminal_has_no_completed_marker": True,
            "all_server_pid_and_process_groups_absent_before_completion": True,
            "all_eight_gpus_zero_compute_and_zero_mib_before_completion": True,
            "frozen_focused_and_inherited_tests_pass_before_gpu": True,
        },
        "invalid_attempts": [
            {
                "trial_id": 1876986,
                "freeze": "C",
                "status": "invalid_before_outputs",
                "completed_cells": 0,
                "reason": "one-shot /server_info 30 second timeout during approximately 46 second scheduler internal warmup after model_info readiness",
                "paper_evidence": False,
            },
            {
                "trial_id": None,
                "freeze": "D",
                "status": "retired_before_gpu",
                "completed_cells": 0,
                "reason": "independent audit found an unbounded TERM-to-wait ordering and insufficient mode/rank/endpoint readiness binding",
                "paper_evidence": False,
            },
            {
                "trial_id": None,
                "freeze": "E",
                "status": "retired_before_gpu",
                "completed_cells": 0,
                "reason": "independent audit proved a completely re-signed rank-1 receipt chain could be placed in the rank-0 file slot because blind replay lacked an external expected-cell anchor",
                "paper_evidence": False,
            },
            {
                "trial_id": 1876986,
                "freeze": "F",
                "status": "invalid_before_raw_outputs",
                "completed_cells": 0,
                "reason": "unified Mamba dtype contract rejected the live FP32 temporal state on all Prefix servers",
                "paper_evidence": False,
            },
            {
                "trial_id": 1879097,
                "freeze": "P",
                "status": "invalid_partial_before_any_hypic_raw_or_store_receipt",
                "completed_cells": 8,
                "excluded_completed_prefix_cells": 8,
                "hypic_completed_cells": 0,
                "reason": "all HYPIC scheduler hooks failed closed on a duplicate pre-snapshot Mamba free-list entry; Prefix-only partial output is excluded and no Store ordering claim is permitted",
                "paper_evidence": False,
            },
            {
                "job_id": 247574,
                "trial_id": 1879456,
                "freeze": "T",
                "status": "invalid_pre_science_asset_authority_rejection",
                "completed_cells": 0,
                "reason": "the old-node inode/device/time authority rejected a byte-identical, root-owned, read-only model view rematerialized under /tmp on another node",
                "later_platform_state": "Terminated after the wrapper had already exited 1; not the invalidation cause",
                "paper_evidence": False,
            },
        ],
        "debug_validation": live_debug_validation,
        "allocator_debug_validation": allocator_debug_validation,
    }
    return {
        "official-source-ledger.json": canonical_bytes(source),
        "environment-ledger.json": canonical_bytes(environment),
        "model-storage-contract.json": canonical_bytes(storage_contract),
        "instrumentation-overlay.json": canonical_bytes(overlay),
        "instrumentation-overlay.diff": overlay_diff,
        "preregistration.json": canonical_bytes(prereg),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("build", "verify"), required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--instrumented-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-weight-ledger", type=Path, required=True)
    parser.add_argument("--model-artifact-ledger", type=Path, required=True)
    parser.add_argument("--asset-observation", type=Path, required=True)
    parser.add_argument("--invalid-t-receipt", type=Path, required=True)
    parser.add_argument("--platform-authority-receipt", type=Path, required=True)
    parser.add_argument("--retired-w-trial-receipt", type=Path, required=True)
    parser.add_argument("--pid1-environ", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--expected-freeze-manifest-sha256", required=True)
    parser.add_argument("--live-debug-root", type=Path, required=True)
    parser.add_argument("--expected-live-debug-manifest-sha256", required=True)
    parser.add_argument("--allocator-debug-root", type=Path, required=True)
    parser.add_argument("--expected-allocator-debug-manifest-sha256", required=True)
    parser.add_argument("--allocator-debug-provenance", type=Path, required=True)
    parser.add_argument("--allocator-debug-launch-plan", type=Path, required=True)
    parser.add_argument("--allocator-debug-freeze-manifest", type=Path, required=True)
    for name in (
        "client",
        "formal_helper",
        "formal_static_helper",
        "serving_helper",
        "receipt_module",
        "patch",
        "receipt_test",
        "inherited_test",
        "inherited_launcher",
        "replay",
        "launcher",
        "safe_wrapper",
        "safe_cwd_guard",
        "model_asset_snapshot",
        "static_builder",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--verify-model-bytes", action="store_true")
    args = parser.parse_args()
    outputs = materialize(args, verify_model_bytes=args.stage == "build" or args.verify_model_bytes)
    if args.stage == "build":
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for name, data in outputs.items():
            (args.output_dir / name).write_bytes(data)
        atomic_json(
            args.output_dir / "preoutput-validation.json",
            {
                "schema": "hypic-rwd5-preoutput-validation-v1",
                "passed": True,
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(outputs.items())},
            },
        )
    else:
        require(args.validation_output is not None, "validation output")
        for name, expected in outputs.items():
            require((args.output_dir / name).read_bytes() == expected, f"terminal static drift: {name}")
        atomic_json(
            args.validation_output,
            {
                "schema": "hypic-rwd5-terminal-static-verification-v1",
                "passed": True,
                "model_bytes_rehashed": bool(args.verify_model_bytes),
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(outputs.items())},
            },
        )


if __name__ == "__main__":
    main()
