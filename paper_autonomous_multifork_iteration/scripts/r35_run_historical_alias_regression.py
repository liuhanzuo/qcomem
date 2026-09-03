from __future__ import annotations

"""Run one rank of the R35 historical clean-run alias regression.

The producer intentionally does not import the R29 executor or its fault
suite.  It consumes an R35-owned, byte-bound execution input which records the
R29 input only as upstream provenance.  One model load serves three strictly
fresh cache cases: the archived pre-fix borrowed path, the repaired borrowed
path, and a materialized-state control.
"""

import argparse
import gc
import hashlib
import importlib.abc
import json
import math
import os
import platform
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence


RUN_ID = "R35-HISTORICAL-ALIAS-20260826A"
UPSTREAM_R29_RUN_ID = "R29-HELDOUT-FAULTS-20260825C"
UPSTREAM_R29_RAW_SHA256 = "5a522e48650e3010621e6e06c7c8bbab67c074bca1b6a6a0c70aa50133b4e98d"
INPUT_SCHEMA = "forkaudit-r35-historical-alias-execution-input-v1"
RESOURCE_SCHEMA = "forkaudit-r35-resource-amendment-v1"
PROTOCOL_SCHEMA = "forkaudit-r35-historical-alias-protocol-v1"
RANK_SCHEMA = "forkaudit-r35-historical-alias-rank-v1"
LANES = ("historical_pre_fix", "repaired_borrowed", "materialized_control")
EVEN_ORDER = LANES
ODD_ORDER = tuple(reversed(LANES))
RECEIPT_ORDER = (
    "frozen_input_and_request_provenance",
    "live_kv_ownership_and_construction_binding",
    "gdn_phase_storage_snapshot_and_pointer_free_replay",
    "advertised_scheduler_action_sequence_replay",
    "persistent_kv_and_gdn_immutability",
    "fresh_case_disposal_pending",
)
HISTORICAL_RECEIPT = "gdn_phase_storage_snapshot_and_pointer_free_replay"
HISTORICAL_GATE = "gdn_completed_binding_rebound"
SIDE_CAR_SHAPE = (1, 248320)
SIDE_CAR_NBYTES = 993280
SHA256_RE = re.compile(r"[0-9a-f]{64}")
FORBIDDEN_IMPORTS = frozenset(
    {"r29_execute_heldout_faults", "r29_heldout_fault_suite"}
)


class R35Error(RuntimeError):
    pass


class ReceiptPredicateRejection(RuntimeError):
    def __init__(self, predicate_id: str, message: str) -> None:
        self.gate_id = predicate_id
        super().__init__(f"{predicate_id}: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise R35Error(message)


class _ForbiddenImportBlocker(importlib.abc.MetaPathFinder):
    """Fail closed if a transitive import tries to load R29 fault code."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: Any = None,
    ) -> None:
        if fullname in FORBIDDEN_IMPORTS:
            raise ImportError(f"R35 forbids importing {fullname}")
        return None


require(not (FORBIDDEN_IMPORTS & set(sys.modules)), "R29 fault code was loaded before R35")
_IMPORT_BLOCKER = _ForbiddenImportBlocker()
sys.meta_path.insert(0, _IMPORT_BLOCKER)

# These are the no-fault RR2/runtime primitives named by the R35 execution
# input.  The import census is checked again immediately below and at output.
import torch

import qcomem_forkaudit_storage_witness as storage_witness
import qcomem_joint_policy as joint_policy
import qcomem_single_token_gdn_ownership as repair
import qcomem_vllm_paged_multifork_resident as resident
import qcomem_vllm_paged_fair_control as fair_control
import run_qcomem_qwen35_forkaudit_review_revision as rr2
import run_qcomem_qwen35_vllm_paged_multifork_resident as resident_runner

SHARED_REUSE = fair_control.SHARED_REUSE
require(
    "qcomem_forkaudit_mutants" in sys.modules,
    "frozen RR2 generic-mutant import census drift",
)


def assert_fault_isolation(stage: str) -> dict[str, Any]:
    observed = sorted(FORBIDDEN_IMPORTS & set(sys.modules))
    require(not observed, f"forbidden R29 import at {stage}: {observed}")
    return {
        "stage": stage,
        "r29_heldout_fault_suite_import_blocked": True,
        "r29_heldout_fault_suite_in_sys_modules": False,
        "r29_executor_in_sys_modules": False,
        "generic_mutant_definition_module_passively_loaded": (
            "qcomem_forkaudit_mutants" in sys.modules
        ),
        "mutation_requested": False,
        "mutation_applied": False,
        "observed_forbidden_modules": observed,
    }


assert_fault_isolation("after_runtime_imports")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def tensor_sha(tensor: torch.Tensor) -> str:
    return sha256_bytes(tensor_bytes(tensor))


def read_bound_file(path: Path, expected_sha256: str, label: str) -> bytes:
    require(SHA256_RE.fullmatch(expected_sha256 or "") is not None, f"{label} SHA format")
    payload = path.read_bytes()
    require(sha256_bytes(payload) == expected_sha256, f"{label} raw SHA drift")
    return payload


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    require(not path.exists() and not pending.exists(), f"refusing to overwrite {path}")
    with pending.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    pending.replace(path)
    _fsync_directory(path.parent)


def write_json_atomic(path: Path, value: Any) -> None:
    write_bytes_atomic(path, canonical_bytes(value) + b"\n")


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    require(set(value) == expected, f"{label} schema drift: {sorted(set(value) ^ expected)}")


def validate_protocol(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "protocol must be an object")
    _exact_keys(
        value,
        {
            "schema_version",
            "run_id",
            "rank_count",
            "lanes",
            "lane_order_by_rank",
            "lane_audit_mode",
            "sidecar",
            "receipt_order",
            "historical_first_failure",
            "expected",
            "content_digest_formula",
            "coordinate_classes",
            "comparison_matrix",
            "source_bindings",
            "resource_amendment_binding",
        },
        "protocol",
    )
    require(value["schema_version"] == PROTOCOL_SCHEMA, "protocol schema")
    require(value["run_id"] == RUN_ID and value["rank_count"] == 8, "protocol run/rank count")
    require(value["lanes"] == list(LANES), "protocol lanes")
    require(
        value["lane_order_by_rank"]
        == {"even": list(EVEN_ORDER), "odd": list(ODD_ORDER)},
        "protocol lane order",
    )
    require(
        value["lane_audit_mode"]
        == {
            "historical_pre_fix": "unified_storage_and_binding",
            "repaired_borrowed": "unified_storage_and_binding",
            "materialized_control": "policy_aware_storage_only",
        },
        "protocol audit modes",
    )
    require(
        value["sidecar"]
        == {
            "dtype": "float32-little-endian",
            "shape": list(SIDE_CAR_SHAPE),
            "nbytes": SIDE_CAR_NBYTES,
        },
        "protocol sidecar",
    )
    require(value["receipt_order"] == list(RECEIPT_ORDER), "protocol receipt order")
    require(
        value["historical_first_failure"]
        == {
            "model_step_index": 0,
            "receipt_id": HISTORICAL_RECEIPT,
            "predicate_id": HISTORICAL_GATE,
        },
        "protocol historical gate",
    )
    require(
        value["expected"] == {"resident_count": 2, "linear_state_count": 60},
        "protocol expected geometry",
    )
    require(
        value["content_digest_formula"]
        == "sha256_json(ordered_content_digests)",
        "protocol content digest formula",
    )
    require(
        value["coordinate_classes"]
        == {
            "archived_coordinate_ranks": [0, 1, 2],
            "additional_frozen_input_ranks": [3, 4, 5, 6, 7],
            "statistical_independence_claimed": False,
        },
        "protocol coordinate classes",
    )
    require(
        value["comparison_matrix"]
        == {
            "pair_mappings": [
                "historical_pre_fix_vs_materialized_control",
                "historical_pre_fix_vs_repaired_borrowed",
                "repaired_borrowed_vs_materialized_control",
            ],
            "output_only": ["greedy_token_exact", "full_fp32_logits_exact"],
            "state_differential": [
                "request0_terminal_gdn_content_exact",
                "logical_kv_content_exact",
            ],
            "state_invariant": ["persistent_base_content_only_invariant"],
            "forkaudit": [
                "lane_local_storage_intervals",
                "owner_relations",
                "setup_to_transition_binding",
            ],
            "normalized_storage_ids_comparable_across_lanes": False,
        },
        "protocol comparison matrix",
    )
    require(isinstance(value["source_bindings"], dict) and value["source_bindings"], "protocol source bindings")
    for key, digest in value["source_bindings"].items():
        require(key.endswith("_sha256") and SHA256_RE.fullmatch(digest), f"protocol source {key}")
    require(
        value["source_bindings"].get("upstream_r29_execution_input_raw_sha256")
        == UPSTREAM_R29_RAW_SHA256,
        "protocol historical R29 input binding",
    )
    require(value["resource_amendment_binding"] == "external_preexecution", "protocol resource binding")
    return dict(value)


def validate_execution_input(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "execution input must be an object")
    _exact_keys(
        value,
        {
            "schema_version",
            "status",
            "created_at_utc",
            "run_id",
            "candidate_output_seen_when_frozen",
            "rank_count",
            "protocol",
            "upstream_r29_execution_input",
            "code",
            "output",
            "claim_boundary",
        },
        "execution input",
    )
    require(value["schema_version"] == INPUT_SCHEMA, "execution input schema")
    require(value["status"] == "frozen_before_candidate_outputs", "execution input status")
    require(value["run_id"] == RUN_ID and value["rank_count"] == 8, "execution run/rank count")
    require(value["candidate_output_seen_when_frozen"] is False, "execution input outcome leak")
    value["protocol"] = validate_protocol(value["protocol"])
    upstream = value["upstream_r29_execution_input"]
    _exact_keys(
        upstream,
        {
            "raw_path",
            "raw_sha256",
            "run_id",
            "model",
            "data",
            "environment",
            "imported_rr2_code",
        },
        "upstream R29 input",
    )
    require(upstream["raw_sha256"] == UPSTREAM_R29_RAW_SHA256, "upstream R29 SHA")
    require(upstream["run_id"] == UPSTREAM_R29_RUN_ID, "upstream R29 run id")
    imported = upstream["imported_rr2_code"]
    _exact_keys(imported, {"code_dir", "ledger_path", "ledger_raw_sha256"}, "RR2 code")
    require(SHA256_RE.fullmatch(imported["ledger_raw_sha256"]) is not None, "RR2 ledger SHA")
    code = value["code"]
    _exact_keys(
        code,
        {
            "runner_path",
            "runner_sha256",
            "repair_path",
            "repair_sha256",
            "storage_witness_path",
            "storage_witness_sha256",
            "resident_path",
            "resident_sha256",
            "resident_runner_path",
            "resident_runner_sha256",
            "transformers_qwen35_source_sha256",
        },
        "R35 code",
    )
    for key, digest in code.items():
        if key.endswith("_sha256"):
            require(SHA256_RE.fullmatch(digest) is not None, f"R35 code {key}")
    _exact_keys(value["output"], {"run_root"}, "output")
    require(isinstance(value["claim_boundary"], dict), "claim boundary")
    return dict(value)


def validate_upstream_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    require(value["raw_sha256"] == UPSTREAM_R29_RAW_SHA256, "historical R29 input SHA drift")
    require(value["run_id"] == UPSTREAM_R29_RUN_ID, "historical R29 run drift")
    raw = read_bound_file(Path(value["raw_path"]), value["raw_sha256"], "upstream R29 input")
    original = json.loads(raw)
    require(original.get("run_id") == value["run_id"], "upstream run id copy drift")
    for key in ("model", "data", "environment"):
        require(original.get(key) == value[key], f"upstream {key} copy drift")
    original_code = original.get("code", {})
    imported = value["imported_rr2_code"]
    require(
        imported
        == {
            "code_dir": original_code.get("imported_rr2_code_dir"),
            "ledger_path": original_code.get("imported_rr2_code_ledger_path"),
            "ledger_raw_sha256": original_code.get("imported_rr2_code_ledger_raw_sha256"),
        },
        "upstream RR2 code copy drift",
    )
    return {
        "raw_sha256": value["raw_sha256"],
        "run_id": value["run_id"],
        "copied_fields_exact": True,
    }


def validate_resource_amendment(
    value: Any,
    *,
    rank: int,
    physical_gpu_index: int,
    expected_gpu_uuid: str,
    expected_execution_input_sha256: str,
) -> dict[str, Any]:
    require(isinstance(value, dict), "resource amendment must be an object")
    _exact_keys(
        value,
        {
            "schema_version",
            "status",
            "created_at_utc",
            "run_id",
            "preregistration_raw_sha256",
            "execution_input_raw_sha256",
            "source_ledger_raw_sha256",
            "execution_package_sha256",
            "science_design_changed",
            "candidate_output_seen_when_frozen",
            "job_id",
            "trial_id",
            "pod",
            "gpu_assignments",
        },
        "resource amendment",
    )
    require(value["schema_version"] == RESOURCE_SCHEMA, "resource amendment schema")
    require(value["status"] == "frozen_after_resource_creation_before_candidate_outputs", "resource status")
    require(value["run_id"] == RUN_ID, "resource run id")
    require(value["science_design_changed"] is False, "resource amendment changed science")
    require(value["candidate_output_seen_when_frozen"] is False, "resource amendment outcome leak")
    for key in (
        "preregistration_raw_sha256",
        "execution_input_raw_sha256",
        "source_ledger_raw_sha256",
        "execution_package_sha256",
    ):
        require(SHA256_RE.fullmatch(value[key]) is not None, f"resource {key}")
    require(
        value["execution_input_raw_sha256"] == expected_execution_input_sha256,
        "resource amendment execution-input binding drift",
    )
    require(
        (type(value["job_id"]) is int and value["job_id"] >= 0)
        or (isinstance(value["job_id"], str) and bool(value["job_id"])),
        "resource job",
    )
    require(
        (type(value["trial_id"]) is int and value["trial_id"] >= 0)
        or (isinstance(value["trial_id"], str) and bool(value["trial_id"])),
        "resource trial",
    )
    require(isinstance(value["pod"], str) and value["pod"], "resource pod")
    assignments = value["gpu_assignments"]
    require(isinstance(assignments, dict) and set(assignments) == {str(i) for i in range(8)}, "GPU assignments")
    observed_uuids: list[str] = []
    for candidate_rank in range(8):
        row = assignments[str(candidate_rank)]
        _exact_keys(row, {"physical_index", "uuid"}, f"GPU assignment row {candidate_rank}")
        require(
            type(row["physical_index"]) is int and row["physical_index"] == candidate_rank,
            f"physical GPU assignment {candidate_rank}",
        )
        require(
            isinstance(row["uuid"], str) and row["uuid"].startswith("GPU-"),
            f"GPU UUID assignment {candidate_rank}",
        )
        observed_uuids.append(row["uuid"])
    require(len(set(observed_uuids)) == 8, "GPU UUID assignments are not unique")
    row = assignments[str(rank)]
    require(row["physical_index"] == physical_gpu_index, "physical GPU assignment drift")
    require(row["uuid"] == expected_gpu_uuid, "GPU UUID assignment drift")
    return dict(value)


def validate_source_bindings(execution_input: Mapping[str, Any]) -> dict[str, str]:
    code = execution_input["code"]
    observed_paths = {
        "runner": Path(__file__).resolve(),
        "repair": Path(repair.__file__).resolve(),
        "storage_witness": Path(storage_witness.__file__).resolve(),
        "resident": Path(resident.__file__).resolve(),
        "resident_runner": Path(resident_runner.__file__).resolve(),
    }
    output: dict[str, str] = {}
    for name, path in observed_paths.items():
        expected_path = Path(code[f"{name}_path"]).resolve()
        expected_sha = code[f"{name}_sha256"]
        require(path == expected_path, f"{name} source path drift")
        require(sha256_file(path) == expected_sha, f"{name} source SHA drift")
        output[f"{name}_sha256"] = expected_sha
    from transformers.models.qwen3_5 import modeling_qwen3_5

    transformers_source = Path(modeling_qwen3_5.__file__).resolve()
    expected_transformers_sha = code["transformers_qwen35_source_sha256"]
    require(
        sha256_file(transformers_source) == expected_transformers_sha,
        "Transformers Qwen3.5 source SHA drift",
    )
    output["transformers_qwen35_source_sha256"] = expected_transformers_sha
    protocol_bindings = execution_input["protocol"]["source_bindings"]
    expected_subset = {
        "runner_sha256": output["runner_sha256"],
        "repair_sha256": output["repair_sha256"],
        "storage_witness_sha256": output["storage_witness_sha256"],
        "resident_sha256": output["resident_sha256"],
        "resident_runner_sha256": output["resident_runner_sha256"],
        "transformers_qwen35_source_sha256": output[
            "transformers_qwen35_source_sha256"
        ],
    }
    require(
        all(protocol_bindings.get(key) == digest for key, digest in expected_subset.items()),
        "protocol/runtime source binding drift",
    )
    return dict(protocol_bindings)


def _verify_sha256_ledger(path: Path, root: Path, expected_sha256: str) -> dict[str, Any]:
    payload = read_bound_file(path, expected_sha256, "RR2 code ledger")
    rows: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for line_number, raw in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split(None, 1)
        require(len(parts) == 2 and SHA256_RE.fullmatch(parts[0]) is not None, f"ledger line {line_number}")
        relative = parts[1].lstrip("*")
        relative_path = Path(relative)
        require(not relative_path.is_absolute() and ".." not in relative_path.parts, "unsafe ledger path")
        canonical_relative = relative_path.as_posix().removeprefix("./")
        require(canonical_relative not in seen_paths, f"duplicate ledger path: {canonical_relative}")
        seen_paths.add(canonical_relative)
        target = root / relative_path
        require(target.is_file() and sha256_file(target) == parts[0], f"RR2 source drift: {relative}")
        rows.append({"path": canonical_relative, "sha256": parts[0]})
    require(bool(rows), "empty RR2 code ledger")
    return {
        "raw_sha256": expected_sha256,
        "file_count": len(rows),
        "rows_sha256": sha256_json(rows),
        "file_sha256": {row["path"]: row["sha256"] for row in rows},
    }


def validate_loaded_science_module_closure(
    *,
    rr2_root: Path,
    rr2_code_receipt: Mapping[str, Any],
    execution_input: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind every loaded project science module to the package or RR2 ledger."""

    code = execution_input["code"]
    package_overrides = {
        "qcomem_single_token_gdn_ownership": (
            Path(code["repair_path"]).resolve(),
            code["repair_sha256"],
        ),
        "qcomem_forkaudit_storage_witness": (
            Path(code["storage_witness_path"]).resolve(),
            code["storage_witness_sha256"],
        ),
        "qcomem_vllm_paged_multifork_resident": (
            Path(code["resident_path"]).resolve(),
            code["resident_sha256"],
        ),
        "run_qcomem_qwen35_vllm_paged_multifork_resident": (
            Path(code["resident_runner_path"]).resolve(),
            code["resident_runner_sha256"],
        ),
    }
    ledger = rr2_code_receipt["file_sha256"]
    require(isinstance(ledger, dict) and ledger, "RR2 module ledger map")
    rows: list[dict[str, Any]] = []
    for module_name, module in sorted(sys.modules.items()):
        if not (
            module_name.startswith("qcomem_")
            or module_name.startswith("run_qcomem_")
            or module_name.startswith("build_qcomem_")
            or module_name == "run_downstream"
        ):
            continue
        module_file = getattr(module, "__file__", None)
        require(isinstance(module_file, str) and module_file, f"science module file missing: {module_name}")
        observed_path = Path(module_file).resolve()
        observed_sha256 = sha256_file(observed_path)
        if module_name in package_overrides:
            expected_path, expected_sha256 = package_overrides[module_name]
            require(observed_path == expected_path, f"package module path drift: {module_name}")
            require(observed_sha256 == expected_sha256, f"package module SHA drift: {module_name}")
            source_class = "r35_package_override"
            relative_path = str(observed_path)
        else:
            require(observed_path.is_relative_to(rr2_root), f"RR2 module shadowed: {module_name}")
            relative_path = str(observed_path.relative_to(rr2_root))
            require(ledger.get(relative_path) == observed_sha256, f"RR2 module ledger drift: {module_name}")
            source_class = "imported_rr2_ledger"
        rows.append(
            {
                "module": module_name,
                "source_class": source_class,
                "path": relative_path,
                "sha256": observed_sha256,
            }
        )
    required_modules = {
        "build_qcomem_forkaudit_rr2_input_manifest",
        "qcomem_forkaudit_storage_witness",
        "qcomem_joint_policy",
        "qcomem_qwen35_functional_stack",
        "qcomem_single_token_gdn_ownership",
        "qcomem_vllm_paged_fair_control",
        "qcomem_vllm_paged_kernel",
        "qcomem_vllm_paged_multifork_resident",
        "run_qcomem_qwen35_forkaudit_review_revision",
        "run_qcomem_qwen35_vllm_paged_multifork_resident",
    }
    observed_modules = {row["module"] for row in rows}
    require(required_modules.issubset(observed_modules), "required science module absent from closure")
    return {
        "module_count": len(rows),
        "modules": rows,
        "modules_sha256": sha256_json(rows),
        "shadowed_module_count": 0,
    }


def validate_python_environment_identity(env_dir_value: str) -> dict[str, Any]:
    lexical = lambda value: Path(os.path.abspath(os.fspath(value)))
    env_dir = lexical(env_dir_value)
    invoked = lexical(sys.executable)
    prefix = lexical(sys.prefix)
    expected = env_dir / "bin" / "python"
    require(prefix == env_dir, "sys.prefix differs from frozen environment")
    require(invoked == expected, "Python was not invoked through frozen env/bin/python")
    require(invoked.resolve(strict=True) == expected.resolve(strict=True), "Python target drift")
    return {
        "frozen_env_dir": str(env_dir),
        "sys_prefix": str(prefix),
        "sys_executable": str(invoked),
        "exact": True,
    }


@dataclass
class Runtime:
    model: Any
    backbone: Any
    plan: Any
    kernel: Any
    document: torch.Tensor
    queries: tuple[torch.Tensor, ...]
    boundary_token: torch.Tensor
    hardware: dict[str, Any]
    input_receipt: dict[str, Any]
    allocator_baseline: dict[str, int] | None = None


def numeric_gpu_receipt(
    *,
    physical_gpu_index: int,
    expected_gpu_uuid: str,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == expected_gpu_uuid,
        "UUID CUDA selector drift",
    )
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "rank requires one visible GPU")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    capability = list(torch.cuda.get_device_capability(0))
    require(capability == environment["compute_capability"] and "H20" in properties.name, "GPU environment drift")
    output = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_gpu_index}",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    columns = [item.strip() for item in output.split(",")]
    require(
        len(columns) == 4
        and int(columns[0]) == physical_gpu_index
        and columns[1] == expected_gpu_uuid
        and columns[2] == environment["gpu_name"],
        "numeric GPU/UUID binding drift",
    )
    return {
        "physical_index": physical_gpu_index,
        "cuda_visible_devices": expected_gpu_uuid,
        "uuid": columns[1],
        "name": columns[2],
        "memory_mib": int(columns[3]),
        "compute_capability": capability,
        "torch_version": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
    }


def load_runtime(
    *,
    rank: int,
    physical_gpu_index: int,
    expected_gpu_uuid: str,
    execution_input: Mapping[str, Any],
) -> Runtime:
    import build_qcomem_forkaudit_rr2_input_manifest as rr2_builder
    from importlib.metadata import version as package_version
    import qcomem_qwen35_functional_stack as functional_stack
    import qcomem_vllm_paged_kernel as paged_kernel
    from transformers import AutoModelForImageTextToText, __version__ as transformers_version

    audit_qwen35_functional_stack_plan = functional_stack.audit_qwen35_functional_stack_plan
    _resolve_vllm_unified_attention = paged_kernel._resolve_vllm_unified_attention
    audit_frozen_kernel_environment = paged_kernel.audit_frozen_kernel_environment
    _audit_model_config_geometry = resident_runner._audit_model_config_geometry
    _resolve_backbone = resident_runner._resolve_backbone

    require(not torch.cuda.is_initialized(), "input rebuild must precede CUDA initialization")
    upstream = execution_input["upstream_r29_execution_input"]
    model_input = upstream["model"]
    data_input = upstream["data"]
    environment = upstream["environment"]
    imported = upstream["imported_rr2_code"]
    require(platform.python_version() == environment["python"], "Python version drift")
    require(str(torch.__version__) == environment["torch"], "Torch version drift")
    require(str(torch.version.cuda) == environment["torch_cuda"], "Torch CUDA drift")
    require(transformers_version == environment["transformers"], "Transformers version drift")
    require(package_version("vllm") == environment["vllm"], "vLLM version drift")
    python_identity = validate_python_environment_identity(environment["env_dir"])
    rr2_root = Path(imported["code_dir"]).resolve()
    require(rr2_root in [Path(item).resolve() for item in sys.path if item], "RR2 code absent from PYTHONPATH")
    code_receipt = _verify_sha256_ledger(Path(imported["ledger_path"]), rr2_root, imported["ledger_raw_sha256"])
    module_closure = validate_loaded_science_module_closure(
        rr2_root=rr2_root,
        rr2_code_receipt=code_receipt,
        execution_input=execution_input,
    )

    pg19_raw = read_bound_file(Path(data_input["pg19_data_path"]), data_input["pg19_data_raw_sha256"], "PG19 data")
    manifest_raw = read_bound_file(Path(data_input["pg19_manifest_path"]), data_input["pg19_manifest_raw_sha256"], "PG19 manifest")
    query_raw = read_bound_file(Path(data_input["frozen_query_banks_path"]), data_input["frozen_query_banks_raw_sha256"], "query banks")
    read_bound_file(Path(model_input["weight_ledger_path"]), model_input["weight_ledger_raw_sha256"], "model weights")
    read_bound_file(Path(model_input["artifact_ledger_path"]), model_input["artifact_ledger_raw_sha256"], "model artifacts")
    banks = json.loads(query_raw)
    require(isinstance(banks, list) and len(banks) == 8, "query bank rank coverage")
    bank = banks[rank]
    model_dir = Path(model_input["model_dir"])
    tokenizer = rr2_builder.load_local_tokenizer(model_dir)
    records, _ = rr2_builder._audit_pg19_train64_bytes(
        pg19_raw, manifest_raw, expectations=rr2_builder.FORMAL_EXPECTATIONS
    )
    windows, windows_sha = joint_policy.build_pg19_calibration_windows(
        records,
        tokenizer,
        books=rr2.FORMAL_BOOKS,
        document_tokens=rr2.FORMAL_DOCUMENT_TOKENS,
        query_tokens=rr2.FORMAL_QUERY_TOKENS,
        stride=rr2.FORMAL_WINDOW_STRIDE,
        candidate_windows_per_book=8,
        seed=20260817,
    )
    require(windows_sha == data_input["pg19_windows_canonical_sha256"], "PG19 windows drift")
    window = windows[rank]
    queries, query_audit = resident.build_pg19_train_query_bank(
        records,
        tokenizer,
        window,
        document_tokens=rr2.FORMAL_DOCUMENT_TOKENS,
        query_tokens=rr2.FORMAL_QUERY_TOKENS,
        count=max(rr2.FORMAL_RESIDENT_COUNTS),
        query_stride=rr2.FORMAL_QUERY_BANK_STRIDE,
    )
    document_cpu = window.document_ids.detach().contiguous().unsqueeze(0)
    require(tensor_sha(document_cpu) == bank["document_token_ids_sha256"], "document token drift")
    require([tensor_sha(item) for item in queries] == [row["query_token_ids_sha256"] for row in bank["rows"]], "query token drift")
    require([int(row["source_token_offset"]) for row in query_audit["rows"]] == [int(row["source_token_offset"]) for row in bank["rows"]], "query coordinates drift")
    hardware = numeric_gpu_receipt(
        physical_gpu_index=physical_gpu_index,
        expected_gpu_uuid=expected_gpu_uuid,
        environment=environment,
    )
    weight_rows = rr2._parse_sha256_ledger(Path(model_input["weight_ledger_path"]).read_bytes(), label="R35 model weights")
    artifact_rows = rr2._parse_sha256_ledger(Path(model_input["artifact_ledger_path"]).read_bytes(), label="R35 model artifacts")
    rr2._verify_weight_ledger_structure(weight_rows, model_dir=model_dir)
    rr2._verify_model_ledger(artifact_rows, model_dir=model_dir, label="R35 model artifacts")
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir),
        revision=model_input["revision"],
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
    )
    outer = getattr(model, "model", None)
    if outer is not None and hasattr(outer, "visual"):
        outer.visual = None
    model.eval()
    _audit_model_config_geometry(model_dir)
    plan = audit_qwen35_functional_stack_plan(model)
    require(tuple(plan.full_attention_layer_indices) == rr2.FORMAL_FULL_LAYERS, "full layer plan drift")
    require(tuple(plan.linear_layer_indices) == rr2.FORMAL_LINEAR_LAYERS, "linear layer plan drift")
    kernel_environment = audit_frozen_kernel_environment()
    require(kernel_environment.get("matches_frozen_environment") is True, "kernel environment drift")
    model = model.to(device="cuda:0", dtype=torch.bfloat16)
    backbone = _resolve_backbone(model)
    kernel = _resolve_vllm_unified_attention()
    document = document_cpu.to(device="cuda:0", non_blocking=False)
    live_queries = tuple(item.to(device="cuda:0", non_blocking=False) for item in queries)
    boundary = live_queries[0][:, 31:32].detach().clone()
    require(tuple(boundary.shape) == (1, 1) and boundary.dtype == torch.long, "boundary token geometry")
    return Runtime(
        model=model,
        backbone=backbone,
        plan=plan,
        kernel=kernel,
        document=document,
        queries=live_queries,
        boundary_token=boundary,
        hardware=hardware,
        input_receipt={
            "rank": rank,
            "coordinate_class": "archived" if rank < 3 else "additional_frozen",
            "model_revision": model_input["revision"],
            "pg19_windows_canonical_sha256": windows_sha,
            "query_bank_manifest_sha256": bank["manifest_sha256"],
            "document_token_ids_sha256": tensor_sha(document),
            "query_token_ids_sha256": [tensor_sha(item) for item in live_queries],
            "boundary_token_coordinate": "frozen_query_bank[rank][0][31]",
            "boundary_token_id": int(boundary.item()),
            "boundary_token_sha256": tensor_sha(boundary),
            "imported_rr2_code": code_receipt,
            "loaded_science_module_closure": module_closure,
            "kernel_environment": kernel_environment,
            "python_environment_identity": python_identity,
        },
    )


def allocator_snapshot() -> dict[str, int]:
    torch.cuda.synchronize()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
    }


def cleanup_allocator() -> dict[str, int]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return allocator_snapshot()


def byte_interval(tensor: torch.Tensor) -> tuple[int, int]:
    shape = tuple(int(value) for value in tensor.shape)
    stride = tuple(int(value) for value in tensor.stride())
    require(bool(shape) and len(shape) == len(stride) and all(size > 0 for size in shape), "tensor geometry")
    minimum = int(tensor.storage_offset())
    maximum = minimum
    for size, step in zip(shape, stride):
        displacement = (size - 1) * step
        minimum += min(displacement, 0)
        maximum += max(displacement, 0)
    element_size = int(tensor.element_size())
    return minimum * element_size, (maximum + 1) * element_size


@dataclass
class IdentityRegistry:
    storage_labels: dict[tuple[str, int, int], str] = field(default_factory=dict)
    object_labels: dict[int, str] = field(default_factory=dict)

    def storage_label(self, tensor: torch.Tensor) -> str:
        storage = tensor.untyped_storage()
        key = (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()))
        if key not in self.storage_labels:
            self.storage_labels[key] = f"storage-{len(self.storage_labels):04d}"
        return self.storage_labels[key]

    def object_label(self, tensor: torch.Tensor) -> str:
        key = id(tensor)
        if key not in self.object_labels:
            self.object_labels[key] = f"tensor-{len(self.object_labels):04d}"
        return self.object_labels[key]


def tensor_descriptor(
    tensor: torch.Tensor,
    *,
    registry: IdentityRegistry,
    owner: str,
    layer_index: int,
    family: str,
    state_index: int,
) -> dict[str, Any]:
    start, end = byte_interval(tensor)
    storage_nbytes = int(tensor.untyped_storage().nbytes())
    require(0 <= start < end <= storage_nbytes, "tensor byte interval")
    return {
        "owner": owner,
        "layer_index": int(layer_index),
        "state_family": family,
        "state_index": int(state_index),
        "coordinate": f"layer:{int(layer_index)}/{family}/state:{int(state_index)}",
        "tensor_id": registry.object_label(tensor),
        "storage_id": registry.storage_label(tensor),
        "storage_nbytes": storage_nbytes,
        "byte_start": start,
        "byte_end_exclusive": end,
        "tensor_nbytes": int(tensor.numel()) * int(tensor.element_size()),
        "shape": [int(value) for value in tensor.shape],
        "stride": [int(value) for value in tensor.stride()],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "dense_contiguous": bool(tensor.is_contiguous()),
        "content_sha256": tensor_sha(tensor),
        "contains_absolute_pointer": False,
        "contains_python_object_id": False,
    }


def capture_snapshot(
    persistent: Any,
    group: Any,
    layer_indices: Sequence[int],
    registry: IdentityRegistry,
) -> dict[str, Any]:
    owners = {"persistent": persistent, "request_0": group.requests[0], "request_1": group.requests[1]}
    rows: list[dict[str, Any]] = []
    for owner in ("persistent", "request_0", "request_1"):
        cache = owners[owner]
        for layer_index in layer_indices:
            layer = cache.layers[int(layer_index)]
            for family in ("conv_states", "recurrent_states"):
                states = getattr(layer, family)
                require(isinstance(states, dict) and sorted(states) == [0], "state schema drift")
                rows.append(
                    tensor_descriptor(
                        states[0],
                        registry=registry,
                        owner=owner,
                        layer_index=int(layer_index),
                        family=family,
                        state_index=0,
                    )
                )
    require(len(rows) == 180, "diagnostic snapshot row count")
    return {
        "row_count": len(rows),
        "rows": rows,
        "rows_sha256": sha256_json(rows),
        "absolute_pointers_persisted": False,
        "python_object_ids_persisted": False,
    }


def _row_map(snapshot: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {(row["owner"], row["coordinate"]): row for row in snapshot["rows"]}


def relation(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    same_storage = left["storage_id"] == right["storage_id"]
    overlap = same_storage and max(left["byte_start"], right["byte_start"]) < min(
        left["byte_end_exclusive"], right["byte_end_exclusive"]
    )
    return {
        "same_tensor_object": left["tensor_id"] == right["tensor_id"],
        "same_storage": same_storage,
        "ranges_overlap": overlap,
        "exact_byte_range_alias": same_storage
        and left["byte_start"] == right["byte_start"]
        and left["byte_end_exclusive"] == right["byte_end_exclusive"],
        "storage_disjoint": not overlap,
        "content_equal": left["content_sha256"] == right["content_sha256"],
    }


def ownership_relations(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    rows = _row_map(snapshot)
    coordinates = sorted(coordinate for owner, coordinate in rows if owner == "persistent")
    output = []
    for coordinate in coordinates:
        base = rows[("persistent", coordinate)]
        request_0 = rows[("request_0", coordinate)]
        request_1 = rows[("request_1", coordinate)]
        output.append(
            {
                "coordinate": coordinate,
                "state_family": base["state_family"],
                "request_0_vs_base": relation(request_0, base),
                "request_0_vs_peer": relation(request_0, request_1),
                "base_vs_peer": relation(base, request_1),
            }
        )
    return {"row_count": len(output), "rows": output, "rows_sha256": sha256_json(output)}


def transition_relations(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    left, right = _row_map(before), _row_map(after)
    require(set(left) == set(right), "snapshot coordinate drift")
    rows = []
    for owner, coordinate in sorted(left):
        pre, post = left[(owner, coordinate)], right[(owner, coordinate)]
        rows.append(
            {
                "owner": owner,
                "coordinate": coordinate,
                "state_family": pre["state_family"],
                "binding_changed": any(
                    pre[key] != post[key]
                    for key in ("tensor_id", "storage_id", "byte_start", "byte_end_exclusive")
                ),
                "content_changed": pre["content_sha256"] != post["content_sha256"],
                "pre_content_sha256": pre["content_sha256"],
                "post_content_sha256": post["content_sha256"],
            }
        )
    return {"row_count": len(rows), "rows": rows, "rows_sha256": sha256_json(rows)}


def owner_content(snapshot: Mapping[str, Any], owner: str) -> dict[str, Any]:
    values = [row["content_sha256"] for row in snapshot["rows"] if row["owner"] == owner]
    require(len(values) == 60, f"{owner} content count")
    return {
        "sha256": sha256_json(values),
        "tensor_count": len(values),
        "ordered_content_digests": values,
    }


def logical_kv_content(group: Any, layer_indices: Sequence[int]) -> list[dict[str, Any]]:
    rows = resident_runner._request_logical_kv_digests(group, layer_indices)
    require(len(rows) == 2, "logical KV request count")
    return json.loads(json.dumps(rows))


def build_case(runtime: Runtime, lane: str) -> tuple[Any, Any, Any, Any, Any, dict[str, str], str]:
    persistent, _ = rr2._convert_persistent(runtime.backbone, runtime.plan, runtime.document, resident_count=2)
    source = resident.source_document_physical_digests(persistent, runtime.plan.full_attention_layer_indices)
    persistent_guard = storage_witness.capture_persistent_gdn_guard(persistent, runtime.plan.linear_layer_indices)
    gdn_policy = (
        resident.GDN_MATERIALIZE_REQUEST_BASE
        if lane == "materialized_control"
        else resident.GDN_BORROW_IMMUTABLE_BASE
    )
    group = resident.build_resident_request_group(
        persistent,
        runtime.plan,
        resident_count=2,
        policy=SHARED_REUSE,
        gdn_base_policy=gdn_policy,
    )
    resident_runner._set_production_no_mask(group, runtime.plan.full_attention_layer_indices)
    request_guard = storage_witness.capture_request_gdn_binding_guard(
        group.requests, runtime.plan.linear_layer_indices, policy=gdn_policy
    )
    live_kv_guard = rr2.capture_live_kv_identity_guard(group, runtime.plan)
    return persistent, group, persistent_guard, request_guard, live_kv_guard, source, gdn_policy


def make_backend(runtime: Runtime, group: Any) -> tuple[Any, str]:
    ledger = resident.MultiForkHitLedger(
        runtime.plan,
        group.requests[0],
        request_index=0,
        resident_count=2,
        request_policy=group.policy,
        expected_calls_per_layer=1,
        initial_query_tokens=1,
        kernel=runtime.kernel,
        strict_position_values=True,
    )
    return ledger, resident.register_multifork_backend(ledger)


def model_step(runtime: Runtime, group: Any, backend: str) -> tuple[torch.Tensor, dict[str, Any]]:
    original = runtime.backbone.config._attn_implementation
    output = None
    try:
        runtime.backbone.config._attn_implementation = backend
        output = runtime.backbone(
            input_ids=runtime.boundary_token,
            past_key_values=group.requests[0],
            use_cache=True,
        )
        logits = resident_runner._last_logits(runtime.model, output).detach().cpu().float().contiguous()
        require(tuple(logits.shape) == SIDE_CAR_SHAPE, "full-logit shape drift")
        require(bool(torch.isfinite(logits).all()), "non-finite logits")
        return logits, {
            "step_index": 0,
            "request_index": 0,
            "semantic_horizon_reached": True,
            "input_token_coordinate": "frozen_query_bank[rank][0][31]",
            "input_token_id": int(runtime.boundary_token.item()),
            "input_token_sha256": tensor_sha(runtime.boundary_token),
            "full_logit_sha256": tensor_sha(logits),
            "greedy_token_id": int(logits.argmax(dim=-1).item()),
        }
    finally:
        output = None
        runtime.backbone.config._attn_implementation = original


def sidecar_reference(path: Path, raw_root: Path, logits: torch.Tensor) -> dict[str, Any]:
    payload = tensor_bytes(logits)
    require(len(payload) == SIDE_CAR_NBYTES, "logit sidecar byte count")
    write_bytes_atomic(path, payload)
    return {
        "path": path.resolve().relative_to(raw_root.resolve()).as_posix(),
        "sha256": sha256_bytes(payload),
        "dtype": "float32-little-endian",
        "shape": list(SIDE_CAR_SHAPE),
        "nbytes": len(payload),
        "finite": True,
    }


def exception_record(exc: BaseException) -> dict[str, Any]:
    return {
        "module": type(exc).__module__,
        "type": type(exc).__qualname__,
        "message": str(exc),
        "gate_id": getattr(exc, "gate_id", None),
        "stack": [
            {"filename": Path(frame.filename).name, "line": int(frame.lineno), "function": frame.name}
            for frame in traceback.extract_tb(exc.__traceback__)
        ],
    }


def clear_exception(exc: BaseException) -> None:
    if exc.__traceback__ is not None:
        traceback.clear_frames(exc.__traceback__)
    exc.__traceback__ = None


def classify_authenticated_rejection(exc: BaseException, receipt_id: str) -> dict[str, Any] | None:
    if not isinstance(exc, (resident.RuntimeInvariantError, storage_witness.GDNStorageWitnessError, ReceiptPredicateRejection)):
        return None
    record = exception_record(exc)
    require(isinstance(record["gate_id"], str) and record["gate_id"], "authenticated gate id missing")
    return {
        "authenticated": True,
        "receipt_id": receipt_id,
        "predicate_id": record["gate_id"],
        "exception": record,
    }


def validate_completed_kv(runtime: Runtime, persistent: Any, group: Any) -> dict[str, Any]:
    ownership = resident.validate_runtime_kv_ownership(
        persistent, group, runtime.plan, require_appended_tail_cow=False
    )
    rows = []
    for request_index, request in enumerate(group.requests):
        for layer_index in runtime.plan.full_attention_layer_indices:
            appended = int(request.layers[layer_index].sequence.appended_tokens)
            if request_index == 0 and appended <= 0:
                raise ReceiptPredicateRejection("KV_TAIL_COW", "completed request did not append")
            rows.append(
                {
                    "request_index": request_index,
                    "layer_index": int(layer_index),
                    "append_required": request_index == 0,
                    "appended_tokens": appended,
                }
            )
    return {"group_ownership": ownership, "completed_request_indices": [0], "appended_rows": rows, "rows_sha256": sha256_json(rows)}


def schedule_receipt() -> dict[str, Any]:
    observed = [{"event_index": 0, "phase": "advertised-model-boundary", "slot_id": 0, "round_index": 0, "request_id": "request-0"}]
    return {"event_count": 1, "schedule_exact": True, "events_sha256": sha256_json(observed)}


def pointer_free_guard(guard: Any) -> dict[str, Any]:
    return {
        "guard_id": guard.guard_id,
        "layer_indices": list(guard.layer_indices),
        "state_index": guard.state_index,
        "baseline_binding_sha256": guard.baseline_binding_sha256,
        "baseline_content_sha256": guard.baseline_content_sha256,
        "absolute_pointers_persisted": False,
    }


def run_audit_battery(
    *,
    runtime: Runtime,
    lane: str,
    persistent: Any,
    group: Any,
    persistent_guard: Any,
    request_guard: Any,
    live_kv_guard: Any,
    source_guard: Mapping[str, str],
    gdn_policy: str,
) -> dict[str, Any]:
    audit_mode = (
        "policy_aware_storage_only"
        if lane == "materialized_control"
        else "unified_storage_and_binding"
    )

    def provenance() -> dict[str, Any]:
        require(runtime.input_receipt["boundary_token_coordinate"] == "frozen_query_bank[rank][0][31]", "boundary coordinate")
        require(group.resident_count == 2 and group.policy == SHARED_REUSE, "request provenance")
        return {"rank": runtime.input_receipt["rank"], "resident_count": 2, "kv_policy": group.policy, "gdn_policy": gdn_policy}

    def live_kv() -> dict[str, Any]:
        ownership = validate_completed_kv(runtime, persistent, group)
        witness = rr2.capture_live_kv_witness(
            persistent,
            group,
            runtime.plan,
            live_kv_guard,
            phase=storage_witness.PHASE_POST_TRANSITION,
            capture_id=os.urandom(16).hex(),
            completed_request_indices=[0],
        )
        return {"ownership": ownership, "witness": witness, "witness_sha256": sha256_json(witness)}

    def gdn_phase() -> dict[str, Any]:
        if audit_mode == "unified_storage_and_binding":
            phase = storage_witness.capture_gdn_phase_witness(
                persistent,
                group.requests,
                runtime.plan.linear_layer_indices,
                run_id=RUN_ID,
                cell_id=f"rank-{runtime.input_receipt['rank']}-{lane}",
                kv_policy=SHARED_REUSE,
                phase=storage_witness.PHASE_POST_TRANSITION,
                policy=gdn_policy,
                persistent_guard=persistent_guard,
                request_guard=request_guard,
                completed_request_indices=[0],
            )
            pointer_free = json.loads(json.dumps(phase))
            storage_replay = storage_witness.replay_gdn_storage_witness(pointer_free["storage_witness"])
            binding_replay = storage_witness.replay_request_gdn_binding_witness(pointer_free["binding_witness"])
            return {
                "capture_protocol": "unified-live-gdn-phase-v1",
                "phase_witness": pointer_free,
                "storage_replay": storage_replay,
                "binding_replay": binding_replay,
            }
        capture_id = os.urandom(16).hex()
        snapshot = storage_witness.capture_gdn_storage_snapshot(
            persistent,
            group.requests,
            runtime.plan.linear_layer_indices,
            phase=storage_witness.PHASE_POST_TRANSITION,
            policy=gdn_policy,
            persistent_guard=persistent_guard,
            completed_request_indices=[0],
            capture_id=capture_id,
            request_guard_id=request_guard.guard_id,
        )
        pointer_free = json.loads(json.dumps(snapshot))
        replay = storage_witness.replay_gdn_storage_witness(pointer_free)
        return {
            "capture_protocol": "policy-aware-storage-only-v1",
            "storage_witness": pointer_free,
            "storage_replay": replay,
        }

    def persistent_immutable() -> dict[str, Any]:
        gdn = storage_witness.verify_persistent_gdn_guard(persistent_guard, persistent)
        observed_kv = resident.source_document_physical_digests(
            persistent, runtime.plan.full_attention_layer_indices
        )
        if observed_kv != source_guard:
            raise ReceiptPredicateRejection("PERSISTENT_KV_IMMUTABLE", "persistent KV changed")
        return {"persistent_gdn": gdn, "persistent_kv_exact": True, "persistent_kv_sha256": sha256_json(observed_kv)}

    operations: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        (RECEIPT_ORDER[0], provenance),
        (RECEIPT_ORDER[1], live_kv),
        (RECEIPT_ORDER[2], gdn_phase),
        (RECEIPT_ORDER[3], schedule_receipt),
        (RECEIPT_ORDER[4], persistent_immutable),
        (RECEIPT_ORDER[5], lambda: {"required": True, "mode": "fresh_case_cleanup_follows_audit"}),
    ]
    completed = []
    rejection = None
    for receipt_id, operation in operations:
        try:
            payload = operation()
        except BaseException as exc:
            rejection = classify_authenticated_rejection(exc, receipt_id)
            if rejection is None:
                raise
            clear_exception(exc)
            break
        completed.append(
            {
                "receipt_id": receipt_id,
                "status": "passed",
                "payload": payload,
                "payload_sha256": sha256_json(payload),
            }
        )
    storage_passed = any(row["receipt_id"] == RECEIPT_ORDER[2] for row in completed)
    return {
        "audit_mode": audit_mode,
        "completed_receipts": completed,
        "first_authenticated_rejection": rejection,
        "unified_witness_passed": (
            storage_passed if audit_mode == "unified_storage_and_binding" else None
        ),
        "storage_witness_passed": storage_passed,
        "expected_historical_rejection_observed": lane == "historical_pre_fix"
        and rejection is not None
        and rejection["receipt_id"] == HISTORICAL_RECEIPT
        and rejection["predicate_id"] == HISTORICAL_GATE,
    }


def persistent_guard_diagnostic(guard: Any, persistent: Any) -> dict[str, Any]:
    try:
        receipt = storage_witness.verify_persistent_gdn_guard(guard, persistent)
        return {"status": "passed", "receipt": receipt, "rejection": None}
    except BaseException as exc:
        rejection = classify_authenticated_rejection(exc, "persistent_guard_secondary_diagnostic")
        if rejection is None:
            raise
        clear_exception(exc)
        return {"status": "authenticated_rejection", "receipt": None, "rejection": rejection}


def run_lane(
    *,
    runtime: Runtime,
    lane: str,
    process_instance_id: str,
    raw_root: Path,
) -> dict[str, Any]:
    before = allocator_snapshot()
    require(before == runtime.allocator_baseline, f"{lane} allocator before fresh case")
    persistent = group = persistent_guard = request_guard = live_kv_guard = None
    source_guard = None
    registry = None
    ledger = None
    backend = ""
    logits = None
    result: dict[str, Any] | None = None
    case_nonce = hashlib.sha256(
        f"{RUN_ID}:{runtime.input_receipt['rank']}:{lane}:{os.urandom(16).hex()}".encode()
    ).hexdigest()[:32]
    try:
        (
            persistent,
            group,
            persistent_guard,
            request_guard,
            live_kv_guard,
            source_guard,
            gdn_policy,
        ) = build_case(runtime, lane)
        registry = IdentityRegistry()
        setup = capture_snapshot(persistent, group, runtime.plan.linear_layer_indices, registry)
        setup_relations = ownership_relations(setup)
        transition_receipt = None
        if lane == "repaired_borrowed":
            transition_receipt = repair.prepare_borrowed_single_token_conv_transition(
                persistent,
                group.requests,
                runtime.plan.linear_layer_indices,
                request_index=0,
            )
        ledger, backend = make_backend(runtime, group)
        logits, step = model_step(runtime, group, backend)
        kernel_ledger = json.loads(json.dumps(rr2._pointer_free_kernel_ledger(ledger.verify_complete())))
        sidecar = sidecar_reference(raw_root / f"{lane}-full-fp32-logits.bin", raw_root, logits)
        post = capture_snapshot(persistent, group, runtime.plan.linear_layer_indices, registry)
        post_relations = ownership_relations(post)
        transitions = transition_relations(setup, post)
        logical_kv = logical_kv_content(group, runtime.plan.full_attention_layer_indices)
        terminal = {
            "request_gdn": [
                {"request_index": index, **owner_content(post, f"request_{index}")}
                for index in range(2)
            ],
            "logical_kv": logical_kv,
            "logical_kv_sha256": sha256_json(logical_kv),
            "persistent_gdn": {
                "setup": owner_content(setup, "persistent"),
                "post": owner_content(post, "persistent"),
            },
            "storage_or_pointer_fields_persisted": False,
        }
        audit = run_audit_battery(
            runtime=runtime,
            lane=lane,
            persistent=persistent,
            group=group,
            persistent_guard=persistent_guard,
            request_guard=request_guard,
            live_kv_guard=live_kv_guard,
            source_guard=source_guard,
            gdn_policy=gdn_policy,
        )
        guard_result = persistent_guard_diagnostic(persistent_guard, persistent)
        if audit["first_authenticated_rejection"] is not None:
            status = "authenticated_forkaudit_rejection_after_model_step"
        else:
            status = "completed_clean"
        result = {
            "lane": lane,
            "rank": runtime.input_receipt["rank"],
            "status": status,
            "fresh_case": True,
            "case_nonce": case_nonce,
            "process_instance_id": process_instance_id,
            "state_reused_from_prior_lane": False,
            "allocator_before": before,
            "allocator_baseline": dict(runtime.allocator_baseline),
            "policy": {"kv": SHARED_REUSE, "gdn": gdn_policy},
            "mutation_receipt": {
                "r29_heldout_fault_module_loaded": False,
                "generic_mutant_definition_module_passively_loaded": (
                    "qcomem_forkaudit_mutants" in sys.modules
                ),
                "mutation_requested": False,
                "mutation_applied": False,
                "mutation_event_count": 0,
            },
            "repair_transition_receipt": transition_receipt,
            "model_step": step,
            "full_logits": sidecar,
            "kernel_ledger": kernel_ledger,
            "setup_snapshot": setup,
            "post_snapshot": post,
            "setup_ownership_relations": setup_relations,
            "post_ownership_relations": post_relations,
            "transition_relations": transitions,
            "terminal_content": terminal,
            "persistent_guard_baseline": pointer_free_guard(persistent_guard),
            "persistent_guard_result": guard_result,
            "source_kv": {
                "setup": source_guard,
                "post": resident.source_document_physical_digests(
                    persistent, runtime.plan.full_attention_layer_indices
                ),
            },
            "audit": audit,
            "operational_invalid": None,
            "cleanup": None,
        }
    finally:
        cleanup_error = None
        if backend:
            try:
                rr2._unregister_backends([backend])
            except BaseException as exc:
                cleanup_error = exception_record(exc)
                clear_exception(exc)
        backend = ""
        persistent = group = persistent_guard = request_guard = live_kv_guard = None
        source_guard = registry = ledger = logits = None
        after = cleanup_allocator()
        exact = after == runtime.allocator_baseline
        cleanup = {
            "fresh_case_disposed": True,
            "registered_backend_restored": cleanup_error is None,
            "strong_references_released": True,
            "gc_collect_completed": True,
            "cuda_empty_cache_completed": True,
            "cuda_synchronize_completed": True,
            "allocator_after": after,
            "allocator_baseline_exact": exact,
            "cleanup_passed": cleanup_error is None and exact,
            "cleanup_error": cleanup_error,
        }
        if result is not None:
            result["cleanup"] = cleanup
        require(cleanup["cleanup_passed"], f"{lane} lifecycle cleanup failed")
    require(result is not None, f"{lane} result missing")
    return result


def discarded_warmup(runtime: Runtime) -> dict[str, Any]:
    persistent = group = persistent_guard = request_guard = live_kv_guard = source = None
    ledger = None
    backend = ""
    logits = None
    try:
        persistent, group, persistent_guard, request_guard, live_kv_guard, source, _ = build_case(
            runtime, "materialized_control"
        )
        ledger, backend = make_backend(runtime, group)
        logits, step = model_step(runtime, group, backend)
        kernel = json.loads(json.dumps(rr2._pointer_free_kernel_ledger(ledger.verify_complete())))
        return {"performed": True, "discarded": True, "model_step": step, "kernel_ledger": kernel}
    finally:
        if backend:
            rr2._unregister_backends([backend])
        persistent = group = persistent_guard = request_guard = live_kv_guard = source = None
        ledger = logits = None
        runtime.allocator_baseline = cleanup_allocator()


def comparison(candidate: Mapping[str, Any], control: Mapping[str, Any], raw_root: Path) -> dict[str, bool]:
    candidate_logits = (raw_root / candidate["full_logits"]["path"]).read_bytes()
    control_logits = (raw_root / control["full_logits"]["path"]).read_bytes()
    return {
        "greedy_token_exact": candidate["model_step"]["greedy_token_id"]
        == control["model_step"]["greedy_token_id"],
        "full_fp32_logits_exact": candidate_logits == control_logits,
        "request0_terminal_gdn_content_exact": candidate["terminal_content"]["request_gdn"][0]["sha256"]
        == control["terminal_content"]["request_gdn"][0]["sha256"],
        "logical_kv_content_exact": candidate["terminal_content"]["logical_kv"]
        == control["terminal_content"]["logical_kv"],
        "persistent_base_content_only_invariant": candidate["terminal_content"]["persistent_gdn"]["setup"]["sha256"]
        == candidate["terminal_content"]["persistent_gdn"]["post"]["sha256"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(0 <= args.rank < 8, "rank must be in [0,7]")
    require(args.physical_gpu_index == args.rank, "fixed rank/physical GPU mapping")
    require(not args.run_dir.exists(), "rank run directory already exists")
    execution_raw = read_bound_file(args.execution_input, args.expected_execution_input_sha256, "R35 execution input")
    execution_input = validate_execution_input(json.loads(execution_raw))
    upstream_receipt = validate_upstream_copy(execution_input["upstream_r29_execution_input"])
    source_bindings = validate_source_bindings(execution_input)
    require(source_bindings == execution_input["protocol"]["source_bindings"], "protocol/source binding drift")
    expected_run_dir = Path(execution_input["output"]["run_root"]) / f"rank-{args.rank}"
    require(args.run_dir.resolve() == expected_run_dir.resolve(), "rank output path drift")
    resource_raw = read_bound_file(args.resource_amendment, args.expected_resource_amendment_sha256, "resource amendment")
    resource = validate_resource_amendment(
        json.loads(resource_raw),
        rank=args.rank,
        physical_gpu_index=args.physical_gpu_index,
        expected_gpu_uuid=args.expected_gpu_uuid,
        expected_execution_input_sha256=args.expected_execution_input_sha256,
    )
    runtime = load_runtime(
        rank=args.rank,
        physical_gpu_index=args.physical_gpu_index,
        expected_gpu_uuid=args.expected_gpu_uuid,
        execution_input=execution_input,
    )
    args.run_dir.mkdir(parents=True)
    raw_root = args.run_dir / "raw"
    raw_root.mkdir()
    process_instance_id = hashlib.sha256(
        f"{RUN_ID}:{args.rank}:{os.getpid()}:{os.urandom(32).hex()}".encode()
    ).hexdigest()
    with torch.inference_mode():
        warmup = discarded_warmup(runtime)
        require(runtime.allocator_baseline is not None, "warmup allocator baseline")
        lane_order = EVEN_ORDER if args.rank % 2 == 0 else ODD_ORDER
        lane_rows = [
            run_lane(
                runtime=runtime,
                lane=lane,
                process_instance_id=process_instance_id,
                raw_root=raw_root,
            )
            for lane in lane_order
        ]
    by_lane = {row["lane"]: row for row in lane_rows}
    control = by_lane["materialized_control"]
    comparisons = {
        "historical_pre_fix_vs_materialized_control": comparison(
            by_lane["historical_pre_fix"], control, raw_root
        ),
        "repaired_borrowed_vs_materialized_control": comparison(
            by_lane["repaired_borrowed"], control, raw_root
        ),
        "historical_pre_fix_vs_repaired_borrowed": comparison(
            by_lane["historical_pre_fix"], by_lane["repaired_borrowed"], raw_root
        ),
    }
    runtime.input_receipt["loaded_science_module_closure"] = validate_loaded_science_module_closure(
        rr2_root=Path(
            execution_input["upstream_r29_execution_input"]["imported_rr2_code"]["code_dir"]
        ).resolve(),
        rr2_code_receipt=runtime.input_receipt["imported_rr2_code"],
        execution_input=execution_input,
    )
    assert_fault_isolation("before_rank_artifact")
    final_isolation = {
        "r29_heldout_fault_suite_import_blocked": True,
        "r29_heldout_fault_suite_in_sys_modules": False,
        "generic_mutant_definition_module_passively_loaded": (
            "qcomem_forkaudit_mutants" in sys.modules
        ),
        "mutation_requested": False,
        "mutation_applied": False,
    }
    result = {
        "schema_version": RANK_SCHEMA,
        "run_id": RUN_ID,
        "status": "rank_completed",
        "rank": args.rank,
        "operational_invalid": None,
        "process_instance_id": process_instance_id,
        "protocol": execution_input["protocol"],
        "execution_input_raw_sha256": args.expected_execution_input_sha256,
        "preregistration_raw_sha256": resource["preregistration_raw_sha256"],
        "amendment_raw_sha256": args.expected_resource_amendment_sha256,
        "resource": {
            "job_id": resource["job_id"],
            "trial_id": resource["trial_id"],
            "pod": resource["pod"],
            "gpu_assignment": resource["gpu_assignments"][str(args.rank)],
            "preregistration_raw_sha256": resource["preregistration_raw_sha256"],
            "execution_package_sha256": resource["execution_package_sha256"],
        },
        "upstream_r29_execution_input": upstream_receipt,
        "source_bindings": source_bindings,
        "fault_isolation": final_isolation,
        "hardware": runtime.hardware,
        "input_receipt": runtime.input_receipt,
        "discarded_warmup": warmup,
        "lane_order": list(lane_order),
        "lanes": by_lane,
        "comparisons": comparisons,
        "scientific_outcome_does_not_control_operational_validity": True,
    }
    write_json_atomic(raw_root / "rank-result.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--execution-input", type=Path, required=True)
    value.add_argument("--expected-execution-input-sha256", required=True)
    value.add_argument("--resource-amendment", type=Path, required=True)
    value.add_argument("--expected-resource-amendment-sha256", required=True)
    value.add_argument("--rank", type=int, required=True)
    value.add_argument("--physical-gpu-index", type=int, required=True)
    value.add_argument("--expected-gpu-uuid", required=True)
    value.add_argument("--run-dir", type=Path, required=True)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
