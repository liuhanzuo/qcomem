from __future__ import annotations

"""Cross-execute one outcome-blind Round-29 held-out fault.

The fault author owns ``r29_heldout_fault_suite.py``.  This executor owns only
the generic adapter from that frozen public interface to the frozen RR2
Qwen3.5/H20 stack.  Detector decisions below never branch on a fault id or an
expected gate.  Every rank runs the same clean/conventional/ForkAudit lanes.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

import qcomem_forkaudit_storage_witness as storage_witness
import qcomem_joint_policy as joint_policy
import qcomem_vllm_paged_multifork_resident as resident
import r29_heldout_fault_suite as fault_suite
import run_qcomem_qwen35_forkaudit_review_revision as rr2
from qcomem_vllm_paged_fair_control import SHARED_REUSE
from qcomem_vllm_paged_multifork_resident import (
    GDN_BORROW_IMMUTABLE_BASE,
    MultiForkHitLedger,
    build_pg19_train_query_bank,
    build_resident_request_group,
    register_multifork_backend,
)
from run_qcomem_qwen35_vllm_paged_multifork_resident import (
    _last_logits,
    _set_production_no_mask,
)


INPUT_SCHEMA = "forkaudit-r29-heldout-execution-input-v1"
RANK_SCHEMA = "forkaudit-r29-heldout-fault-rank-v1"
LANE_ORDER = ("clean", "fault_conventional", "fault_forkaudit")
SIDE_CAR_SHAPE = (1, 248320)
SIDE_CAR_DTYPE = "float32"
SIDE_CAR_NBYTES = 993280
SHA256_RE = re.compile(r"[0-9a-f]{64}")
PRODUCTION_ASSERTION_ALLOWLIST = (
    {
        "allowlist_id": "PA-Q16-CANONICAL-MASK-v1",
        "message": "vLLM fused backend cannot replace this non-canonical attention mask",
        "function": "validate_canonical_tail_causal_mask",
    },
    {
        "allowlist_id": "PA-Q16-PAIRED-VIEWS-v1",
        "message": "fused backend requires paired Q16 paged views",
        "function": "_paired_sequence",
    },
)


class HeldOutExecutionError(RuntimeError):
    """A preflight, lifecycle, or artifact-contract failure."""


class ReceiptPredicateRejection(RuntimeError):
    """A uniform executor-owned receipt predicate rejection.

    This class is used only for predicates computed by the frozen executor
    itself (for example, immutable source-KV bytes).  Candidate audit modules
    retain their own exception classes and gate identifiers.
    """

    def __init__(self, predicate_id: str, message: str) -> None:
        self.gate_id = predicate_id
        super().__init__(f"{predicate_id}: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HeldOutExecutionError(message)


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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    require(not pending.exists(), f"stale pending artifact: {pending}")
    with pending.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    pending.replace(path)
    _fsync_directory(path.parent)


def write_json_atomic(path: Path, value: Any) -> None:
    write_bytes_atomic(path, canonical_bytes(value) + b"\n")


def read_bound_file(path: Path, expected_sha256: str, label: str) -> bytes:
    require(SHA256_RE.fullmatch(expected_sha256 or "") is not None, f"{label} SHA format")
    payload = path.read_bytes()
    require(sha256_bytes(payload) == expected_sha256, f"{label} raw SHA drift")
    return payload


def _expect_exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    require(set(value) == keys, f"{label} schema drift")


def validate_execution_input(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "execution input must be an object")
    _expect_exact_keys(
        value,
        {
            "schema_version",
            "status",
            "created_at_utc",
            "run_id",
            "suite_binding",
            "fixed_protocol",
            "code",
            "model",
            "data",
            "environment",
            "output",
            "claim_boundary",
        },
        "execution input",
    )
    require(value["schema_version"] == INPUT_SCHEMA, "execution input schema")
    require(value["status"] == "frozen_before_candidate_outputs", "execution input status")
    require(isinstance(value["run_id"], str) and value["run_id"], "run id")
    suite = value["suite_binding"]
    _expect_exact_keys(suite, {"raw_sha256", "canonical_sha256", "fault_module_sha256", "launcher_sha256", "author_test_sha256"}, "suite binding")
    for field in suite:
        require(SHA256_RE.fullmatch(suite[field]) is not None, f"suite binding {field}")
    protocol = value["fixed_protocol"]
    _expect_exact_keys(
        protocol,
        {
            "rank_assignment",
            "lane_order",
            "document_tokens",
            "page_size",
            "resident_count",
            "input_token_coordinate",
            "advertised_horizon_tokens",
            "kv_policy",
            "gdn_policy",
            "sidecar_shape",
            "sidecar_dtype",
            "sidecar_nbytes",
        },
        "fixed protocol",
    )
    require(protocol["rank_assignment"] == {"0": "H01", "1": "H02", "2": "H03"}, "rank assignment")
    require(protocol["lane_order"] == list(LANE_ORDER), "lane order")
    require(protocol["document_tokens"] == 4095 and protocol["page_size"] == 128, "document/page geometry")
    require(protocol["resident_count"] == 2, "resident count")
    require(protocol["input_token_coordinate"] == "frozen_query_bank[rank][0][31]", "token coordinate")
    require(protocol["advertised_horizon_tokens"] == 1, "advertised horizon")
    require(protocol["kv_policy"] == SHARED_REUSE, "KV policy")
    require(protocol["gdn_policy"] == GDN_BORROW_IMMUTABLE_BASE, "GDN policy")
    require(protocol["sidecar_shape"] == list(SIDE_CAR_SHAPE), "sidecar shape")
    require(protocol["sidecar_dtype"] == SIDE_CAR_DTYPE, "sidecar dtype")
    require(protocol["sidecar_nbytes"] == SIDE_CAR_NBYTES, "sidecar bytes")
    code = value["code"]
    _expect_exact_keys(
        code,
        {
            "code_dir",
            "executor_path",
            "executor_sha256",
            "aggregator_path",
            "aggregator_sha256",
            "fault_module_path",
            "imported_rr2_code_dir",
            "imported_rr2_code_ledger_path",
            "imported_rr2_code_ledger_raw_sha256",
        },
        "code input",
    )
    for field in ("executor_sha256", "aggregator_sha256", "imported_rr2_code_ledger_raw_sha256"):
        require(SHA256_RE.fullmatch(code[field]) is not None, f"code {field}")
    model = value["model"]
    _expect_exact_keys(
        model,
        {
            "model_dir",
            "model_id",
            "revision",
            "dtype",
            "weight_ledger_path",
            "weight_ledger_raw_sha256",
            "artifact_ledger_path",
            "artifact_ledger_raw_sha256",
        },
        "model input",
    )
    require(model["model_id"] == "Qwen/Qwen3.5-35B-A3B", "model id")
    require(model["revision"] == "59d61f3ce65a6d9863b86d2e96597125219dc754", "model revision")
    require(model["dtype"] == "bfloat16", "model dtype")
    for field in ("weight_ledger_raw_sha256", "artifact_ledger_raw_sha256"):
        require(SHA256_RE.fullmatch(model[field]) is not None, f"model {field}")
    data = value["data"]
    _expect_exact_keys(
        data,
        {
            "split",
            "pg19_data_path",
            "pg19_data_raw_sha256",
            "pg19_manifest_path",
            "pg19_manifest_raw_sha256",
            "pg19_windows_canonical_sha256",
            "frozen_query_banks_path",
            "frozen_query_banks_raw_sha256",
        },
        "data input",
    )
    require(data["split"] == "PG19 train only", "data split")
    for field in ("pg19_data_raw_sha256", "pg19_manifest_raw_sha256", "pg19_windows_canonical_sha256", "frozen_query_banks_raw_sha256"):
        require(SHA256_RE.fullmatch(data[field]) is not None, f"data {field}")
    environment = value["environment"]
    _expect_exact_keys(environment, {"env_dir", "python", "torch", "torch_cuda", "transformers", "vllm", "gpu_name", "compute_capability"}, "environment input")
    require(environment["gpu_name"] == "NVIDIA H20-3e", "GPU name")
    require(environment["compute_capability"] == [9, 0], "compute capability")
    output = value["output"]
    _expect_exact_keys(output, {"run_root", "raw_root"}, "output input")
    require(Path(output["raw_root"]) == Path(output["run_root"]) / "raw", "output roots")
    claim = value["claim_boundary"]
    _expect_exact_keys(claim, {"historical_pattern_inspired_only", "naturally_occurring_claimed", "upstream_implementation_evaluated", "detection_rate_reported"}, "claim boundary")
    require(claim == {"historical_pattern_inspired_only": True, "naturally_occurring_claimed": False, "upstream_implementation_evaluated": False, "detection_rate_reported": False}, "claim boundary values")
    return dict(value)


def _verify_sha256_ledger(ledger_path: Path, root: Path, expected_sha256: str) -> dict[str, Any]:
    payload = read_bound_file(ledger_path, expected_sha256, "imported RR2 code ledger")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split(None, 1)
        require(len(parts) == 2 and SHA256_RE.fullmatch(parts[0]) is not None, f"code ledger line {line_number}")
        relative = parts[1].lstrip("*")
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts, f"unsafe code ledger path {relative}")
        target = root / path
        require(target.is_file(), f"missing imported source {relative}")
        observed = sha256_file(target)
        require(observed == parts[0], f"imported source SHA drift: {relative}")
        rows.append({"path": relative, "sha256": observed})
    require(bool(rows), "empty imported code ledger")
    return {"raw_sha256": expected_sha256, "file_count": len(rows), "rows_sha256": sha256_json(rows)}


def _exception_record(exc: BaseException) -> dict[str, Any]:
    frames = traceback.extract_tb(exc.__traceback__)
    return {
        "module": type(exc).__module__,
        "type": type(exc).__qualname__,
        "message": str(exc),
        "gate_id": getattr(exc, "gate_id", None),
        "stack": [
            {"filename": Path(frame.filename).name, "line": int(frame.lineno), "function": frame.name}
            for frame in frames
        ],
    }


def _clear_exception(exc: BaseException) -> None:
    tb = exc.__traceback__
    if tb is not None:
        traceback.clear_frames(tb)
    exc.__traceback__ = None


def classify_production_assertion(exc: BaseException) -> dict[str, Any] | None:
    """Recognize only an assertion emitted by the production paged kernel.

    The rule is source/module based and independent of fault identity or
    message text.  Everything else remains operationally invalid unless it is
    an authenticated ForkAudit rejection in the enabled lane.
    """

    module = type(exc).__module__
    name = type(exc).__qualname__
    record = _exception_record(exc)
    if module == "qcomem_vllm_paged_kernel" and name == "QComemPagedKernelError":
        for allowlisted in PRODUCTION_ASSERTION_ALLOWLIST:
            exact_frame = any(
                row["filename"] == "qcomem_vllm_paged_kernel.py"
                and row["function"] == allowlisted["function"]
                for row in record["stack"]
            )
            if str(exc) == allowlisted["message"] and exact_frame:
                return {
                    "classification": "exact_production_assertion",
                    "production_assertion_allowlist_id": allowlisted["allowlist_id"],
                    "exception": record,
                }
    return None


AUTHENTICATED_AUDIT_EXCEPTIONS = (
    resident.RuntimeInvariantError,
    storage_witness.GDNStorageWitnessError,
    ReceiptPredicateRejection,
)


def classify_authenticated_rejection(exc: BaseException, receipt_id: str) -> dict[str, Any] | None:
    if not isinstance(exc, AUTHENTICATED_AUDIT_EXCEPTIONS):
        return None
    record = _exception_record(exc)
    gate = record.get("gate_id")
    require(isinstance(gate, str) and gate, "authenticated audit rejection lacks predicate id")
    return {
        "authenticated": True,
        "receipt_id": receipt_id,
        "predicate_id": gate,
        "exception": record,
    }


def detector_cell(*, status: str, caught: bool | None, reason: str, evidence: Any = None) -> dict[str, Any]:
    require(status in ("evaluated", "not_evaluated"), "detector status")
    require((status == "evaluated") == isinstance(caught, bool), "detector caught semantics")
    return {"status": status, "caught": caught, "reason": reason, "evidence": evidence}


def compare_semantic_outputs(clean: Mapping[str, Any], faulty: Mapping[str, Any], *, raw_root: Path) -> dict[str, Any]:
    """Apply token and full-logit baselines without any fault-specific rule."""

    token_ready = clean.get("semantic_horizon_reached") is True and faulty.get("semantic_horizon_reached") is True and isinstance(clean.get("greedy_token_id"), int) and isinstance(faulty.get("greedy_token_id"), int)
    if token_ready:
        same = clean["greedy_token_id"] == faulty["greedy_token_id"]
        token_cell = detector_cell(status="evaluated", caught=not same, reason="greedy_token_comparison", evidence={"clean": clean["greedy_token_id"], "fault": faulty["greedy_token_id"], "equal": same})
    else:
        token_cell = detector_cell(status="not_evaluated", caught=None, reason="clean_or_fault_semantic_horizon_missing")

    def load(row: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, Any]] | None:
        sidecar = row.get("full_logits")
        if not isinstance(sidecar, dict) or sidecar.get("finite") is not True:
            return None
        relative = sidecar.get("path")
        if not isinstance(relative, str):
            return None
        relative_path = Path(relative)
        require(not relative_path.is_absolute() and ".." not in relative_path.parts, "unsafe sidecar path")
        path = raw_root / relative_path
        payload = path.read_bytes()
        require(len(payload) == SIDE_CAR_NBYTES, "sidecar byte length")
        require(sha256_bytes(payload) == sidecar.get("sha256"), "sidecar SHA drift")
        tensor = torch.frombuffer(bytearray(payload), dtype=torch.float32).clone().reshape(SIDE_CAR_SHAPE)
        require(bool(torch.isfinite(tensor).all()), "non-finite sidecar")
        return tensor, dict(sidecar)

    clean_loaded = load(clean)
    fault_loaded = load(faulty)
    if clean_loaded is None or fault_loaded is None:
        logit_cell = detector_cell(status="not_evaluated", caught=None, reason="clean_or_fault_finite_full_logits_missing")
    else:
        clean_logits, clean_ref = clean_loaded
        fault_logits, fault_ref = fault_loaded
        delta = fault_logits - clean_logits
        max_abs = float(delta.abs().max().item())
        denominator = max(float(torch.linalg.vector_norm(clean_logits).item()), 1e-30)
        relative_l2 = float(torch.linalg.vector_norm(delta).item()) / denominator
        exact = clean_ref["sha256"] == fault_ref["sha256"]
        argmax_clean = int(clean_logits.argmax(dim=-1).item())
        argmax_fault = int(fault_logits.argmax(dim=-1).item())
        logit_cell = detector_cell(
            status="evaluated",
            caught=not exact,
            reason="full_fp32_logit_byte_sha_comparison",
            evidence={
                "exact": exact,
                "clean_sha256": clean_ref["sha256"],
                "fault_sha256": fault_ref["sha256"],
                "argmax_equal": argmax_clean == argmax_fault,
                "max_absolute_difference": max_abs,
                "relative_l2": relative_l2,
            },
        )
    return {"greedy_token": token_cell, "full_fp32_logits": logit_cell}


def _snapshot_allocator() -> dict[str, int]:
    torch.cuda.synchronize()
    return {"allocated_bytes": int(torch.cuda.memory_allocated()), "reserved_bytes": int(torch.cuda.memory_reserved())}


def _cleanup_allocator() -> dict[str, int]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return _snapshot_allocator()


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


def _gpu_receipt(expected_uuid: str, execution_input: Mapping[str, Any]) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    require(visible == expected_uuid, "CUDA_VISIBLE_DEVICES/assignment mismatch")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "rank requires one visible GPU")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    capability = list(torch.cuda.get_device_capability(0))
    expected = execution_input["environment"]
    require(capability == expected["compute_capability"] and "H20" in properties.name, "assigned GPU environment drift")
    output = subprocess.run(
        ["nvidia-smi", f"--id={expected_uuid}", "--query-gpu=uuid,name,memory.total", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    columns = [item.strip() for item in output.split(",")]
    require(len(columns) == 3 and columns[0] == expected_uuid and columns[1] == expected["gpu_name"], "nvidia-smi receipt drift")
    return {
        "uuid": columns[0],
        "name": columns[1],
        "memory_mib": int(columns[2]),
        "compute_capability": capability,
        "torch_version": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
    }


def validate_python_environment_identity(
    env_dir_value: str | os.PathLike[str],
    *,
    executable_value: str | os.PathLike[str] | None = None,
    prefix_value: str | os.PathLike[str] | None = None,
    base_prefix_value: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Authenticate a symlink-based virtual environment without false rejection.

    Virtual-environment launchers commonly resolve to a base interpreter that
    lives outside the environment directory.  Environment membership is
    therefore proved by the lexical invocation path and ``sys.prefix``.  The
    resolved executable is separately required to equal the frozen
    ``env/bin/python`` target, but is not required to remain below ``env``.
    Exact package versions are checked independently by ``_load_runtime``.
    """

    def lexical_absolute(value: str | os.PathLike[str]) -> Path:
        return Path(os.path.abspath(os.fspath(value)))

    env_dir = lexical_absolute(env_dir_value)
    invoked = lexical_absolute(sys.executable if executable_value is None else executable_value)
    prefix = lexical_absolute(sys.prefix if prefix_value is None else prefix_value)
    base_prefix = lexical_absolute(
        sys.base_prefix if base_prefix_value is None else base_prefix_value
    )
    expected_invocation = env_dir / "bin" / "python"
    require(env_dir.is_absolute(), "frozen environment directory must be absolute")
    require(prefix == env_dir, "sys.prefix differs from frozen environment directory")
    require(invoked == expected_invocation, "Python was not invoked through frozen env/bin/python")
    invoked_realpath = invoked.resolve(strict=True)
    expected_realpath = expected_invocation.resolve(strict=True)
    require(
        invoked_realpath == expected_realpath,
        "invoked Python resolves to a different interpreter than frozen env/bin/python",
    )
    return {
        "schema_version": "forkaudit-r29-python-environment-identity-v2",
        "frozen_env_dir": str(env_dir),
        "sys_prefix": str(prefix),
        "sys_base_prefix": str(base_prefix),
        "sys_executable": str(invoked),
        "frozen_env_python": str(expected_invocation),
        "sys_executable_realpath": str(invoked_realpath),
        "frozen_env_python_realpath": str(expected_realpath),
        "sys_prefix_exact": True,
        "lexical_invocation_exact": True,
        "resolved_interpreter_target_exact": True,
        "resolved_interpreter_required_below_env": False,
    }


def _load_runtime(args: argparse.Namespace, execution_input: Mapping[str, Any]) -> Runtime:
    import build_qcomem_forkaudit_rr2_input_manifest as rr2_builder
    from qcomem_qwen35_functional_stack import audit_qwen35_functional_stack_plan
    from qcomem_vllm_paged_kernel import _resolve_vllm_unified_attention, audit_frozen_kernel_environment
    from run_qcomem_qwen35_vllm_paged_multifork_resident import _audit_model_config_geometry, _resolve_backbone
    from transformers import AutoModelForImageTextToText, __version__ as transformers_version

    require(not torch.cuda.is_initialized(), "input rebuild must precede CUDA initialization")
    code = execution_input["code"]
    model_input = execution_input["model"]
    data_input = execution_input["data"]
    environment = execution_input["environment"]
    require(platform.python_version().startswith(environment["python"]), "Python version drift")
    require(str(torch.__version__) == environment["torch"], "Torch version drift")
    require(str(torch.version.cuda) == environment["torch_cuda"], "Torch CUDA version drift")
    require(transformers_version == environment["transformers"], "Transformers version drift")
    try:
        from importlib.metadata import version as package_version

        vllm_version = package_version("vllm")
    except Exception as exc:
        raise HeldOutExecutionError(f"cannot resolve vLLM version: {exc}") from exc
    require(vllm_version == environment["vllm"], "vLLM version drift")
    python_environment_identity = validate_python_environment_identity(
        environment["env_dir"]
    )
    require(Path(code["imported_rr2_code_dir"]).resolve() in [Path(item).resolve() for item in sys.path if item], "imported RR2 directory absent from PYTHONPATH")
    code_receipt = _verify_sha256_ledger(Path(code["imported_rr2_code_ledger_path"]), Path(code["imported_rr2_code_dir"]), code["imported_rr2_code_ledger_raw_sha256"])

    pg19_raw = read_bound_file(Path(data_input["pg19_data_path"]), data_input["pg19_data_raw_sha256"], "PG19 data")
    manifest_raw = read_bound_file(Path(data_input["pg19_manifest_path"]), data_input["pg19_manifest_raw_sha256"], "PG19 manifest")
    read_bound_file(Path(data_input["frozen_query_banks_path"]), data_input["frozen_query_banks_raw_sha256"], "frozen query banks")
    read_bound_file(Path(model_input["weight_ledger_path"]), model_input["weight_ledger_raw_sha256"], "model weight ledger")
    read_bound_file(Path(model_input["artifact_ledger_path"]), model_input["artifact_ledger_raw_sha256"], "model artifact ledger")
    banks = json.loads(Path(data_input["frozen_query_banks_path"]).read_text(encoding="utf-8"))
    require(isinstance(banks, list) and len(banks) == 8, "frozen query-bank rank coverage")
    bank = banks[args.rank]
    model_dir = Path(model_input["model_dir"])
    tokenizer = rr2_builder.load_local_tokenizer(model_dir)
    records, _audit = rr2_builder._audit_pg19_train64_bytes(pg19_raw, manifest_raw, expectations=rr2_builder.FORMAL_EXPECTATIONS)
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
    require(windows_sha == data_input["pg19_windows_canonical_sha256"], "PG19 windows digest drift")
    window = windows[args.rank]
    queries, query_audit = build_pg19_train_query_bank(
        records,
        tokenizer,
        window,
        document_tokens=rr2.FORMAL_DOCUMENT_TOKENS,
        query_tokens=rr2.FORMAL_QUERY_TOKENS,
        count=max(rr2.FORMAL_RESIDENT_COUNTS),
        query_stride=rr2.FORMAL_QUERY_BANK_STRIDE,
    )
    document_cpu = window.document_ids.detach().contiguous().unsqueeze(0)
    require(tensor_sha(document_cpu) == bank["document_token_ids_sha256"], "document token digest drift")
    require([tensor_sha(query) for query in queries] == [row["query_token_ids_sha256"] for row in bank["rows"]], "query-bank digest drift")
    require([int(row["source_token_offset"]) for row in query_audit["rows"]] == [int(row["source_token_offset"]) for row in bank["rows"]], "query-bank coordinate drift")
    hardware = _gpu_receipt(args.expected_gpu_uuid, execution_input)
    weight_rows = rr2._parse_sha256_ledger(Path(model_input["weight_ledger_path"]).read_bytes(), label="R29 held-out model weight ledger")
    artifact_rows = rr2._parse_sha256_ledger(Path(model_input["artifact_ledger_path"]).read_bytes(), label="R29 held-out model artifact ledger")
    rr2._verify_weight_ledger_structure(weight_rows, model_dir=model_dir)
    rr2._verify_model_ledger(artifact_rows, model_dir=model_dir, label="R29 held-out model artifact ledger")
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
    require(tuple(plan.full_attention_layer_indices) == rr2.FORMAL_FULL_LAYERS, "full-attention plan drift")
    require(tuple(plan.linear_layer_indices) == rr2.FORMAL_LINEAR_LAYERS, "linear-attention plan drift")
    kernel_environment = audit_frozen_kernel_environment()
    require(kernel_environment.get("matches_frozen_environment") is True, "kernel environment drift")
    model = model.to(device="cuda:0", dtype=torch.bfloat16)
    backbone = _resolve_backbone(model)
    kernel = _resolve_vllm_unified_attention()
    document = document_cpu.to(device="cuda:0", non_blocking=False)
    live_queries = tuple(query.to(device="cuda:0", non_blocking=False) for query in queries)
    boundary_token = live_queries[0][:, 31:32].detach().clone()
    require(tuple(boundary_token.shape) == (1, 1) and boundary_token.dtype == torch.long, "boundary token geometry")
    return Runtime(
        model=model,
        backbone=backbone,
        plan=plan,
        kernel=kernel,
        document=document,
        queries=live_queries,
        boundary_token=boundary_token,
        hardware=hardware,
        input_receipt={
            "rank": args.rank,
            "model_revision": model_input["revision"],
            "model_weight_ledger_raw_sha256": model_input["weight_ledger_raw_sha256"],
            "model_artifact_ledger_raw_sha256": model_input["artifact_ledger_raw_sha256"],
            "pg19_data_raw_sha256": data_input["pg19_data_raw_sha256"],
            "pg19_manifest_raw_sha256": data_input["pg19_manifest_raw_sha256"],
            "pg19_windows_canonical_sha256": windows_sha,
            "frozen_query_banks_raw_sha256": data_input["frozen_query_banks_raw_sha256"],
            "query_bank_manifest_sha256": bank["manifest_sha256"],
            "document_token_ids_sha256": tensor_sha(document),
            "query_token_ids_sha256": [tensor_sha(query) for query in live_queries],
            "boundary_token_coordinate": "frozen_query_bank[rank][0][31]",
            "boundary_token_id": int(boundary_token.item()),
            "boundary_token_sha256": tensor_sha(boundary_token),
            "imported_rr2_code": code_receipt,
            "kernel_environment": kernel_environment,
            "python_environment_identity": python_environment_identity,
        },
    )


def _default_action_sequence(rank: int) -> dict[str, Any]:
    events = [
        {
            "event_index": 0,
            "request_index": 0,
            "role": "advertised-boundary-token",
            "input_coordinate": "frozen_query_bank[rank][0][31]",
            "externally_advertised": True,
        }
    ]
    return {
        "schema_version": "forkaudit-r29-executor-action-sequence-v1",
        "fault_id": None,
        "rank": rank,
        "advertised_logical_advance_tokens": 1,
        "actual_model_invocations": 1,
        "events": events,
        "events_sha256": sha256_json(events),
        "fresh_case_disposal_required": True,
    }


def _validate_action_sequence(value: Any, *, expected_fault_id: str | None) -> dict[str, Any]:
    require(isinstance(value, dict), "action sequence must be an object")
    required = {
        "schema_version",
        "fault_id",
        "advertised_logical_advance_tokens",
        "actual_model_invocations",
        "events",
        "events_sha256",
        "fresh_case_disposal_required",
    }
    optional = {"rank"}
    require(set(value) == required or set(value) == required | optional, "action sequence schema")
    require(value["fault_id"] == expected_fault_id, "action sequence fault binding")
    require(value["advertised_logical_advance_tokens"] == 1, "advertised action horizon")
    events = value["events"]
    require(isinstance(events, list) and len(events) == value["actual_model_invocations"] and len(events) >= 1, "action event count")
    require(sha256_json(events) == value["events_sha256"], "action sequence digest")
    for index, event in enumerate(events):
        require(isinstance(event, dict), "action event object")
        _expect_exact_keys(event, {"event_index", "request_index", "role", "input_coordinate", "externally_advertised"}, "action event")
        require(event["event_index"] == index and event["request_index"] == 0, "action event index/request")
        require(event["input_coordinate"] == "frozen_query_bank[rank][0][31]", "action input coordinate")
        require(type(event["externally_advertised"]) is bool, "action advertised flag")
    require(sum(int(event["externally_advertised"]) for event in events) == 1, "exactly one advertised action")
    require(value["fresh_case_disposal_required"] is True, "fresh action case required")
    return dict(value)


def _action_sequence_for_lane(fault_id: str, lane: str, rank: int) -> dict[str, Any]:
    fault_active = lane != "clean"
    if fault_active and fault_id in fault_suite.ACTION_SEQUENCE_FAULT_IDS:
        return _validate_action_sequence(fault_suite.h02_action_sequence(request_index=0), expected_fault_id=fault_id)
    return _validate_action_sequence(_default_action_sequence(rank), expected_fault_id=None)


def _build_fresh_case(runtime: Runtime) -> tuple[Any, Any, Any, Any, Any, dict[str, str]]:
    persistent, _conversion = rr2._convert_persistent(runtime.backbone, runtime.plan, runtime.document, resident_count=2)
    source_kv_guard = resident.source_document_physical_digests(persistent, runtime.plan.full_attention_layer_indices)
    persistent_gdn_guard = storage_witness.capture_persistent_gdn_guard(persistent, runtime.plan.linear_layer_indices)
    group = build_resident_request_group(
        persistent,
        runtime.plan,
        resident_count=2,
        policy=SHARED_REUSE,
        gdn_base_policy=GDN_BORROW_IMMUTABLE_BASE,
    )
    _set_production_no_mask(group, runtime.plan.full_attention_layer_indices)
    request_gdn_guard = storage_witness.capture_request_gdn_binding_guard(
        group.requests,
        runtime.plan.linear_layer_indices,
        policy=GDN_BORROW_IMMUTABLE_BASE,
    )
    live_kv_guard = rr2.capture_live_kv_identity_guard(group, runtime.plan)
    return persistent, group, persistent_gdn_guard, request_gdn_guard, live_kv_guard, source_kv_guard


def _make_backend(runtime: Runtime, group: Any, actual_invocations: int) -> tuple[MultiForkHitLedger, str]:
    ledger = MultiForkHitLedger(
        runtime.plan,
        group.requests[0],
        request_index=0,
        resident_count=2,
        request_policy=group.policy,
        expected_calls_per_layer=actual_invocations,
        initial_query_tokens=1,
        kernel=runtime.kernel,
        strict_position_values=True,
    )
    return ledger, register_multifork_backend(ledger)


def _model_step(runtime: Runtime, group: Any, backend: str) -> tuple[torch.Tensor, dict[str, Any]]:
    original = runtime.backbone.config._attn_implementation
    output = None
    try:
        runtime.backbone.config._attn_implementation = backend
        output = runtime.backbone(input_ids=runtime.boundary_token, past_key_values=group.requests[0], use_cache=True)
        logits = _last_logits(runtime.model, output).detach().cpu().float().contiguous()
        require(tuple(logits.shape) == SIDE_CAR_SHAPE, "live full-logit shape drift")
        require(bool(torch.isfinite(logits).all()), "non-finite live full logits")
        return logits, {
            "request_index": 0,
            "input_token_coordinate": "frozen_query_bank[rank][0][31]",
            "input_token_id": int(runtime.boundary_token.item()),
            "input_token_sha256": tensor_sha(runtime.boundary_token),
            "full_logit_sha256": tensor_sha(logits),
            "greedy_token_id": int(logits.argmax(dim=-1).item()),
        }
    finally:
        output = None
        runtime.backbone.config._attn_implementation = original
        require(runtime.backbone.config._attn_implementation == original, "attention backend did not restore")


def _sidecar_reference(path: Path, *, raw_root: Path, logits: torch.Tensor) -> dict[str, Any]:
    payload = tensor_bytes(logits)
    require(len(payload) == SIDE_CAR_NBYTES, "full-logit sidecar byte size")
    relative = path.resolve().relative_to(raw_root.resolve())
    write_bytes_atomic(path, payload)
    return {
        "path": relative.as_posix(),
        "sha256": sha256_bytes(payload),
        "dtype": SIDE_CAR_DTYPE,
        "shape": list(SIDE_CAR_SHAPE),
        "nbytes": len(payload),
        "finite": True,
        "contains_absolute_pointer": False,
    }


def _receipt_require(condition: bool, predicate_id: str, message: str) -> None:
    if not condition:
        raise ReceiptPredicateRejection(predicate_id, message)


def _schedule_receipt(action_sequence: Mapping[str, Any]) -> dict[str, Any]:
    observed = [
        {
            "phase": "advertised-model-boundary" if event["externally_advertised"] else "hidden-model-boundary",
            "slot_id": 0,
            "round_index": index,
            "request_id": "request-0",
        }
        for index, event in enumerate(action_sequence["events"])
    ]
    expected = [
        {
            "event_index": index,
            "phase": "advertised-model-boundary",
            "slot_id": 0,
            "round_index": index,
            "request_id": "request-0",
        }
        for index in range(int(action_sequence["advertised_logical_advance_tokens"]))
    ]
    normalized = [
        {
            "event_index": index,
            "phase": row.get("phase"),
            "slot_id": row.get("slot_id"),
            "round_index": row.get("round_index"),
            "request_id": row.get("request_id"),
        }
        for index, row in enumerate(observed)
    ]
    _receipt_require(
        normalized == expected,
        "ADVERTISED_ACTION_SEQUENCE_EXACT",
        "observed model action sequence differs from the advertised logical horizon",
    )
    return {
        "observed_event_count": len(observed),
        "advertised_event_count": len(expected),
        "observed_events_sha256": sha256_json(observed),
        "expected_events_sha256": sha256_json(expected),
        "replay": {"event_count": len(normalized), "schedule_exact": True},
    }


def validate_completed_request_kv_horizon(
    persistent: Any,
    group: Any,
    plan: Any,
    *,
    completed_request_indices: Sequence[int],
) -> dict[str, Any]:
    """Validate group ownership while scoping the append requirement correctly.

    The resident validator's boolean ``require_appended_tail_cow`` applies to
    every request in the group.  A held-out lane advances only the requests
    named by its frozen action sequence, while the other resident request is a
    deliberately live but unadvanced peer.  We therefore retain the complete
    group ownership validation and its per-appended-sequence COW checks, then
    require a positive append only for the completed request set.
    """

    completed = tuple(int(value) for value in completed_request_indices)
    _receipt_require(bool(completed), "KV_TAIL_COW", "completed request set is empty")
    _receipt_require(
        len(completed) == len(set(completed))
        and all(0 <= value < len(group.requests) for value in completed),
        "KV_TAIL_COW",
        "completed request set is invalid",
    )
    group_ownership = resident.validate_runtime_kv_ownership(
        persistent,
        group,
        plan,
        require_appended_tail_cow=False,
    )
    appended_rows: list[dict[str, int | bool]] = []
    completed_set = set(completed)
    for request_index, request in enumerate(group.requests):
        for layer_index in plan.full_attention_layer_indices:
            appended_tokens = int(request.layers[layer_index].sequence.appended_tokens)
            required = request_index in completed_set
            if required:
                _receipt_require(
                    appended_tokens > 0,
                    "KV_TAIL_COW",
                    "completed request did not append before tail-COW validation",
                )
            appended_rows.append(
                {
                    "request_index": request_index,
                    "layer_index": int(layer_index),
                    "append_required": required,
                    "appended_tokens": appended_tokens,
                }
            )
    return {
        "schema_version": "forkaudit-r29-completed-request-kv-horizon-v1",
        "group_ownership": group_ownership,
        "group_validator_requires_every_request_appended": False,
        "completed_request_indices": list(completed),
        "append_requirement_scoped_to_completed_requests": True,
        "completed_requests_appended": True,
        "appended_rows": appended_rows,
        "appended_rows_sha256": sha256_json(appended_rows),
    }


def run_receipt_battery(
    *,
    execution_input: Mapping[str, Any],
    suite_raw_sha256: str,
    suite_canonical_sha256: str,
    rank: int,
    runtime: Runtime,
    persistent: Any,
    group: Any,
    persistent_gdn_guard: Any,
    request_gdn_guard: Any,
    live_kv_guard: Any,
    source_kv_guard: Mapping[str, str],
    action_sequence: Mapping[str, Any],
    cell_id: str,
    restore_mutation: Callable[[], Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    """Run the one ordered battery; stop at the first authenticated rejection."""

    receipts: list[dict[str, Any]] = []
    restoration: dict[str, Any] | None = None

    def provenance() -> dict[str, Any]:
        _receipt_require(suite_raw_sha256 == execution_input["suite_binding"]["raw_sha256"], "FROZEN_INPUT_PROVENANCE", "suite raw binding drift")
        _receipt_require(suite_canonical_sha256 == execution_input["suite_binding"]["canonical_sha256"], "FROZEN_INPUT_PROVENANCE", "suite canonical binding drift")
        _receipt_require(runtime.input_receipt["rank"] == rank, "FROZEN_INPUT_PROVENANCE", "rank input drift")
        _receipt_require(runtime.input_receipt["boundary_token_coordinate"] == "frozen_query_bank[rank][0][31]", "FROZEN_INPUT_PROVENANCE", "token coordinate drift")
        _receipt_require(group.resident_count == 2 and group.policy == SHARED_REUSE, "FROZEN_REQUEST_PROVENANCE", "resident request construction drift")
        return {
            "suite_raw_sha256": suite_raw_sha256,
            "suite_canonical_sha256": suite_canonical_sha256,
            "rank": rank,
            "boundary_token_sha256": runtime.input_receipt["boundary_token_sha256"],
            "resident_count": group.resident_count,
            "kv_policy": group.policy,
            "gdn_policy": GDN_BORROW_IMMUTABLE_BASE,
        }

    def live_kv() -> dict[str, Any]:
        completed_request_indices = tuple(
            sorted({int(event["request_index"]) for event in action_sequence["events"]})
        )
        ownership = validate_completed_request_kv_horizon(
            persistent,
            group,
            runtime.plan,
            completed_request_indices=completed_request_indices,
        )
        witness = rr2.capture_live_kv_witness(
            persistent,
            group,
            runtime.plan,
            live_kv_guard,
            phase=storage_witness.PHASE_POST_TRANSITION,
            capture_id=os.urandom(16).hex(),
            completed_request_indices=list(completed_request_indices),
        )
        return {"ownership": ownership, "witness": witness, "witness_sha256": sha256_json(witness)}

    def gdn_phase() -> dict[str, Any]:
        phase = storage_witness.capture_gdn_phase_witness(
            persistent,
            group.requests,
            runtime.plan.linear_layer_indices,
            run_id=execution_input["run_id"],
            cell_id=cell_id,
            kv_policy=SHARED_REUSE,
            phase=storage_witness.PHASE_POST_TRANSITION,
            policy=GDN_BORROW_IMMUTABLE_BASE,
            persistent_guard=persistent_gdn_guard,
            request_guard=request_gdn_guard,
            completed_request_indices=list(
                sorted({int(event["request_index"]) for event in action_sequence["events"]})
            ),
        )
        pointer_free = json.loads(json.dumps(phase))
        storage_replay = storage_witness.replay_gdn_storage_witness(pointer_free["storage_witness"])
        binding_replay = storage_witness.replay_request_gdn_binding_witness(pointer_free["binding_witness"])
        return {
            "phase_witness": pointer_free,
            "phase_witness_sha256": sha256_json(pointer_free),
            "storage_replay": storage_replay,
            "binding_replay": binding_replay,
            "pointer_free_round_trip": True,
        }

    def immutable() -> dict[str, Any]:
        gdn = storage_witness.verify_persistent_gdn_guard(persistent_gdn_guard, persistent)
        observed_kv = resident.source_document_physical_digests(persistent, runtime.plan.full_attention_layer_indices)
        _receipt_require(observed_kv == source_kv_guard, "PERSISTENT_KV_IMMUTABLE", "persistent document KV digest changed")
        return {"persistent_gdn": gdn, "persistent_kv_digest_sha256": sha256_json(observed_kv), "persistent_kv_exact": True}

    def restoration_or_disposal() -> dict[str, Any]:
        nonlocal restoration
        if restore_mutation is not None:
            restoration = dict(restore_mutation())
            return {"mode": "state_mutation_restored", "restoration": restoration}
        return {"mode": "fresh_case_disposal_pending", "required": True}

    ordered: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("frozen_input_and_request_provenance", provenance),
        ("live_kv_ownership_and_construction_binding", live_kv),
        ("gdn_phase_storage_snapshot_and_pointer_free_replay", gdn_phase),
        ("advertised_scheduler_action_sequence_replay", lambda: _schedule_receipt(action_sequence)),
        ("persistent_kv_and_gdn_immutability", immutable),
        ("mutation_restoration_or_fresh_case_disposal", restoration_or_disposal),
    ]
    for receipt_id, operation in ordered:
        try:
            payload = operation()
        except BaseException as exc:
            rejection = classify_authenticated_rejection(exc, receipt_id)
            if rejection is None:
                raise
            _clear_exception(exc)
            return receipts, rejection, restoration
        receipts.append({"receipt_id": receipt_id, "status": "passed", "payload": payload})
    return receipts, None, restoration


def _case_nonce(run_id: str, rank: int, fault_id: str, lane: str) -> str:
    return sha256_bytes(f"{run_id}\0{rank}\0{fault_id}\0{lane}".encode("utf-8"))[:32]


def _intervention_for_lane(fault_id: str, lane: str, group: Any, plan: Any, action_sequence: Mapping[str, Any]) -> tuple[Any | None, dict[str, Any]]:
    if lane == "clean":
        return None, {"kind": "none", "fault_active": False, "mutation_observed": False}
    if fault_id in fault_suite.STATE_MUTATION_FAULT_IDS:
        handle = fault_suite.apply_state_fault(fault_id, group, plan)
        return handle, {"kind": "reversible_state_mutation", "fault_active": True, "applied_receipt": handle.applied_receipt()}
    require(fault_id in fault_suite.ACTION_SEQUENCE_FAULT_IDS, "unknown fault intervention type")
    return None, {
        "kind": "immutable_action_sequence",
        "fault_active": True,
        "action_sequence": dict(action_sequence),
        "action_sequence_sha256": sha256_json(action_sequence),
        "fresh_case_disposal_required": True,
    }


_MUTATION_DESCRIPTOR_KEYS = {
    "schema_version",
    "request_index",
    "layer_index",
    "field",
    "shape",
    "stride",
    "dtype",
    "device",
    "values",
    "values_sha256",
    "contains_absolute_pointer",
}
_MUTATION_DESCRIPTOR_GEOMETRY_KEYS = (
    "schema_version",
    "request_index",
    "layer_index",
    "field",
    "shape",
    "stride",
    "dtype",
    "device",
    "contains_absolute_pointer",
)


def _validate_mutation_descriptor(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be a mapping")
    descriptor = dict(value)
    require(set(descriptor) == _MUTATION_DESCRIPTOR_KEYS, f"{label} schema drift")
    values = descriptor["values"]
    require(isinstance(values, list) and bool(values), f"{label} values")
    require(
        all(type(item) is int for item in values),
        f"{label} values must be integer coordinates",
    )
    require(
        descriptor["values_sha256"] == sha256_json(values),
        f"{label} value digest drift",
    )
    require(descriptor["contains_absolute_pointer"] is False, f"{label} pointer flag")
    return descriptor


def restore_mutation_delta_scope(handle: Any) -> dict[str, Any]:
    """Undo only the frozen mutation delta and preserve legal model evolution.

    A frozen state handle may describe a complete routing tensor even though
    its undo changes only a strict coordinate subset.  A model step may also
    make a legal transition at another coordinate.  Comparing the entire
    post-undo tensor to the pre-model descriptor therefore confounds target
    restoration with legitimate transition state.  This generic adapter
    derives the mutation coordinates solely from ``pre != mutated``; it has no
    fault-id or expected-outcome branch.
    """

    pre = _validate_mutation_descriptor(handle.pre_descriptor, "mutation pre descriptor")
    mutated = _validate_mutation_descriptor(
        handle.mutated_descriptor,
        "mutation mutated descriptor",
    )
    require(
        all(pre[key] == mutated[key] for key in _MUTATION_DESCRIPTOR_GEOMETRY_KEYS),
        "mutation descriptor geometry drift",
    )
    require(len(pre["values"]) == len(mutated["values"]), "mutation value length drift")
    changed_indices = [
        index
        for index, (left, right) in enumerate(zip(pre["values"], mutated["values"]))
        if left != right
    ]
    require(bool(changed_indices), "mutation delta is empty")

    pre_restore = _validate_mutation_descriptor(
        handle.capture(),
        "mutation pre-restore descriptor",
    )
    require(
        all(
            pre_restore[key] == pre[key]
            for key in _MUTATION_DESCRIPTOR_GEOMETRY_KEYS
        ),
        "pre-restore descriptor geometry drift",
    )
    require(
        all(
            pre_restore["values"][index] == mutated["values"][index]
            for index in changed_indices
        ),
        "mutation target changed before restoration",
    )

    handle.undo()
    restored = _validate_mutation_descriptor(
        handle.capture(),
        "mutation restored descriptor",
    )
    require(
        all(
            restored[key] == pre[key]
            for key in _MUTATION_DESCRIPTOR_GEOMETRY_KEYS
        ),
        "restored descriptor geometry drift",
    )
    changed_set = set(changed_indices)
    target_restored = all(
        restored["values"][index] == pre["values"][index]
        for index in changed_indices
    )
    non_target_preserved = all(
        restored["values"][index] == pre_restore["values"][index]
        for index in range(len(restored["values"]))
        if index not in changed_set
    )
    require(target_restored, "mutation target did not restore exactly")
    require(non_target_preserved, "mutation undo changed a non-target coordinate")

    target_pre_values = [pre["values"][index] for index in changed_indices]
    target_mutated_values = [mutated["values"][index] for index in changed_indices]
    target_pre_restore_values = [
        pre_restore["values"][index] for index in changed_indices
    ]
    target_restored_values = [
        restored["values"][index] for index in changed_indices
    ]
    return {
        "schema_version": "forkaudit-r29-heldout-restoration-v2",
        "fault_id": str(handle.fault_id),
        "target_kind": str(handle.target_kind),
        "applied_pre_sha256": sha256_json(pre),
        "applied_mutated_sha256": sha256_json(mutated),
        "mutation_coordinate_indices": changed_indices,
        "pre_restore_descriptor": pre_restore,
        "restored_descriptor": restored,
        "pre_restore_sha256": sha256_json(pre_restore),
        "restored_sha256": sha256_json(restored),
        "target_pre_values_sha256": sha256_json(target_pre_values),
        "target_mutated_values_sha256": sha256_json(target_mutated_values),
        "target_pre_restore_values_sha256": sha256_json(
            target_pre_restore_values
        ),
        "target_restored_values_sha256": sha256_json(target_restored_values),
        "target_remained_mutated_through_horizon": True,
        "target_restored_exact": True,
        "non_target_preserved_across_undo": True,
        "restoration_observed": True,
        "contains_absolute_pointer": False,
    }


def _run_lane(
    *,
    runtime: Runtime,
    execution_input: Mapping[str, Any],
    suite_raw_sha256: str,
    suite_canonical_sha256: str,
    rank: int,
    fault_id: str,
    lane: str,
    sidecar_root: Path,
    raw_root: Path,
) -> dict[str, Any]:
    require(runtime.allocator_baseline is not None, "allocator baseline not frozen")
    before = _snapshot_allocator()
    nonce = _case_nonce(execution_input["run_id"], rank, fault_id, lane)
    case: dict[str, Any] = {
        "lane": lane,
        "case_nonce": nonce,
        "fresh_case": True,
        "state_reused_from_prior_lane": False,
        "allocator_before": before,
        "allocator_baseline": dict(runtime.allocator_baseline),
        "action_sequence": None,
        "intervention": None,
        "model_invocations": [],
        "kernel_ledger": None,
        "completion_status": "not_started",
        "semantic_horizon_reached": False,
        "advertised_horizon_tokens": 1,
        "greedy_token_id": None,
        "full_logits": None,
        "production_assertion": detector_cell(status="not_evaluated", caught=None, reason="model_not_run"),
        "forkaudit": detector_cell(status="not_evaluated", caught=None, reason="receipt_verdicts_not_available"),
        "completed_receipts": [],
        "first_authenticated_rejection": None,
        "restoration_receipt": None,
        "operational_invalid": None,
        "cleanup": None,
    }
    persistent = group = persistent_gdn_guard = request_gdn_guard = live_kv_guard = None
    source_kv_guard: Mapping[str, str] | None = None
    mutation_handle = None
    ledger = None
    backend = ""
    logits = None
    restoration_cache: dict[str, Any] | None = None

    def restore_mutation_once() -> dict[str, Any]:
        nonlocal restoration_cache
        require(mutation_handle is not None, "mutation handle missing at restoration")
        if restoration_cache is None:
            restoration_cache = restore_mutation_delta_scope(mutation_handle)
        return dict(restoration_cache)

    try:
        require(before == runtime.allocator_baseline, "allocator baseline not restored before fresh lane")
        action_sequence = _action_sequence_for_lane(fault_id, lane, rank)
        case["action_sequence"] = action_sequence
        persistent, group, persistent_gdn_guard, request_gdn_guard, live_kv_guard, source_kv_guard = _build_fresh_case(runtime)
        mutation_handle, intervention = _intervention_for_lane(fault_id, lane, group, runtime.plan, action_sequence)
        case["intervention"] = intervention
        ledger, backend = _make_backend(runtime, group, int(action_sequence["actual_model_invocations"]))
        execution_exception: BaseException | None = None
        try:
            for _event in action_sequence["events"]:
                logits, step = _model_step(runtime, group, backend)
                case["model_invocations"].append(step)
            case["kernel_ledger"] = rr2._pointer_free_kernel_ledger(ledger.verify_complete())
            case["completion_status"] = "completed"
            case["semantic_horizon_reached"] = True
            case["greedy_token_id"] = int(logits.argmax(dim=-1).item())
            sidecar_path = sidecar_root / f"{lane}-final-fp32-logits.bin"
            case["full_logits"] = _sidecar_reference(sidecar_path, raw_root=raw_root, logits=logits)
            case["production_assertion"] = detector_cell(status="evaluated", caught=False, reason="model_completed_without_production_assertion")
        except BaseException as exc:
            execution_exception = exc
            production = classify_production_assertion(exc)
            if production is not None:
                case["completion_status"] = "production_assertion"
                case["production_assertion"] = detector_cell(status="evaluated", caught=True, reason="exact_production_assertion", evidence=production)
            elif lane == "fault_forkaudit":
                authenticated = classify_authenticated_rejection(exc, "model_execution_registered_runtime_receipt")
                if authenticated is not None:
                    case["completion_status"] = "authenticated_forkaudit_rejection_before_horizon"
                    case["first_authenticated_rejection"] = authenticated
                    case["forkaudit"] = detector_cell(status="evaluated", caught=True, reason="authenticated_registered_runtime_receipt_rejection", evidence=authenticated)
                else:
                    case["completion_status"] = "operational_invalid"
                    case["operational_invalid"] = _exception_record(exc)
            else:
                case["completion_status"] = "operational_invalid"
                case["operational_invalid"] = _exception_record(exc)
            _clear_exception(exc)

        receipts_enabled = lane in ("clean", "fault_forkaudit")
        if execution_exception is None and receipts_enabled:
            require(source_kv_guard is not None, "source KV guard missing")
            receipts, rejection, restoration = run_receipt_battery(
                execution_input=execution_input,
                suite_raw_sha256=suite_raw_sha256,
                suite_canonical_sha256=suite_canonical_sha256,
                rank=rank,
                runtime=runtime,
                persistent=persistent,
                group=group,
                persistent_gdn_guard=persistent_gdn_guard,
                request_gdn_guard=request_gdn_guard,
                live_kv_guard=live_kv_guard,
                source_kv_guard=source_kv_guard,
                action_sequence=action_sequence,
                cell_id=f"rank-{rank}-{lane}-{nonce}",
                restore_mutation=(
                    None if mutation_handle is None else restore_mutation_once
                ),
            )
            case["completed_receipts"] = receipts
            case["first_authenticated_rejection"] = rejection
            if restoration is not None:
                case["restoration_receipt"] = restoration
            if lane == "clean":
                if rejection is None:
                    case["forkaudit"] = detector_cell(status="evaluated", caught=False, reason="clean_uniform_receipt_battery_passed")
                else:
                    case["completion_status"] = "operational_invalid"
                    case["operational_invalid"] = {"classification": "clean_receipt_rejection", "rejection": rejection}
                    case["forkaudit"] = detector_cell(status="not_evaluated", caught=None, reason="clean_case_invalid")
            else:
                case["forkaudit"] = detector_cell(
                    status="evaluated",
                    caught=rejection is not None,
                    reason="first_authenticated_rejection" if rejection is not None else "uniform_receipt_battery_passed_without_rejection",
                    evidence=rejection,
                )
        elif lane == "fault_conventional":
            case["forkaudit"] = detector_cell(status="not_evaluated", caught=None, reason="forkaudit_receipt_verdicts_withheld_by_preregistration")
    except BaseException as exc:
        if case.get("operational_invalid") is None:
            case["completion_status"] = "operational_invalid"
            case["operational_invalid"] = _exception_record(exc)
        _clear_exception(exc)
    finally:
        restoration_error = None
        if mutation_handle is not None:
            try:
                restoration = restore_mutation_once()
                if case.get("restoration_receipt") is None:
                    case["restoration_receipt"] = restoration
            except BaseException as exc:
                restoration_error = _exception_record(exc)
                _clear_exception(exc)
        if backend:
            try:
                rr2._unregister_backends([backend])
            except BaseException as exc:
                restoration_error = restoration_error or _exception_record(exc)
                _clear_exception(exc)
        backend = ""
        ledger = mutation_handle = logits = None
        persistent = group = persistent_gdn_guard = request_gdn_guard = live_kv_guard = source_kv_guard = None
        after = _cleanup_allocator()
        baseline_exact = after == runtime.allocator_baseline
        cleanup_passed = restoration_error is None and baseline_exact
        case["cleanup"] = {
            "fresh_case_disposed": True,
            "registered_backend_restored": restoration_error is None,
            "gc_collect_completed": True,
            "cuda_empty_cache_completed": True,
            "cuda_synchronize_completed": True,
            "allocator_after": after,
            "allocator_baseline_exact": baseline_exact,
            "cleanup_passed": cleanup_passed,
            "cleanup_error": restoration_error,
        }
        if not cleanup_passed and case.get("operational_invalid") is None:
            case["completion_status"] = "operational_invalid"
            case["operational_invalid"] = {"classification": "lifecycle_cleanup_failure", "cleanup": case["cleanup"]}
    return case


def _discarded_warmup(runtime: Runtime) -> dict[str, Any]:
    persistent = group = persistent_guard = request_guard = kv_guard = source_guard = None
    backend = ""
    ledger = None
    logits = None
    try:
        persistent, group, persistent_guard, request_guard, kv_guard, source_guard = _build_fresh_case(runtime)
        ledger, backend = _make_backend(runtime, group, 1)
        logits, step = _model_step(runtime, group, backend)
        receipt = rr2._pointer_free_kernel_ledger(ledger.verify_complete())
        return {"performed": True, "discarded": True, "model_step": step, "kernel_ledger": receipt}
    finally:
        if backend:
            rr2._unregister_backends([backend])
        backend = ""
        persistent = group = persistent_guard = request_guard = kv_guard = source_guard = ledger = logits = None
        baseline = _cleanup_allocator()
        runtime.allocator_baseline = baseline


def _validate_invocation_args(args: argparse.Namespace, suite: Mapping[str, Any], execution_input: Mapping[str, Any]) -> None:
    expected_fault = execution_input["fixed_protocol"]["rank_assignment"][str(args.rank)]
    require(args.fault_id == expected_fault, "rank/fault invocation mismatch")
    require(suite["rank_assignment"][str(args.rank)] == args.fault_id, "suite rank/fault mismatch")
    output = Path(args.output).resolve()
    sidecar = Path(args.sidecar_root).resolve()
    raw_root = Path(execution_input["output"]["raw_root"]).resolve()
    require(output == raw_root / f"heldout-fault-rank-{args.rank}.json", "rank output path drift")
    require(sidecar == raw_root / "sidecars" / f"rank-{args.rank}", "rank sidecar root drift")


def run(args: argparse.Namespace) -> dict[str, Any]:
    suite_raw = read_bound_file(args.suite, args.expected_suite_raw_sha256, "held-out suite")
    suite = json.loads(suite_raw)
    suite_validation = fault_suite.validate_frozen_suite(suite)
    require(suite_validation["suite_sha256"] == args.expected_suite_canonical_sha256, "suite canonical SHA drift")
    input_raw = read_bound_file(args.execution_input, args.expected_execution_input_sha256, "execution input")
    execution_input = validate_execution_input(json.loads(input_raw))
    require(execution_input["suite_binding"]["raw_sha256"] == args.expected_suite_raw_sha256, "execution-input suite raw binding")
    require(execution_input["suite_binding"]["canonical_sha256"] == args.expected_suite_canonical_sha256, "execution-input suite canonical binding")
    code = execution_input["code"]
    require(sha256_file(Path(__file__).resolve()) == code["executor_sha256"], "executor source SHA drift")
    require(Path(__file__).resolve() == Path(code["executor_path"]).resolve(), "executor path binding drift")
    require(sha256_file(Path(code["fault_module_path"])) == execution_input["suite_binding"]["fault_module_sha256"], "fault module SHA drift")
    _validate_invocation_args(args, suite, execution_input)
    runtime = _load_runtime(args, execution_input)
    with torch.inference_mode():
        warmup = _discarded_warmup(runtime)
        require(runtime.allocator_baseline is not None, "warmup did not freeze allocator baseline")
        lanes = [
            _run_lane(
                runtime=runtime,
                execution_input=execution_input,
                suite_raw_sha256=args.expected_suite_raw_sha256,
                suite_canonical_sha256=args.expected_suite_canonical_sha256,
                rank=args.rank,
                fault_id=args.fault_id,
                lane=lane,
                sidecar_root=args.sidecar_root,
                raw_root=Path(execution_input["output"]["raw_root"]),
            )
            for lane in LANE_ORDER
        ]
    require([row["lane"] for row in lanes] == list(LANE_ORDER), "lane execution order")
    clean = lanes[0]
    for faulty in lanes[1:]:
        faulty["semantic_comparisons"] = compare_semantic_outputs(clean, faulty, raw_root=Path(execution_input["output"]["raw_root"]))
    clean["semantic_comparisons"] = {
        "greedy_token": detector_cell(status="not_evaluated", caught=None, reason="clean_reference_lane"),
        "full_fp32_logits": detector_cell(status="not_evaluated", caught=None, reason="clean_reference_lane"),
    }
    operational_invalid_count = sum(row["operational_invalid"] is not None for row in lanes)
    result = {
        "schema_version": RANK_SCHEMA,
        "status": "completed_rank_artifact",
        "run_id": execution_input["run_id"],
        "rank": args.rank,
        "fault_id": args.fault_id,
        "suite_raw_sha256": args.expected_suite_raw_sha256,
        "suite_canonical_sha256": args.expected_suite_canonical_sha256,
        "execution_input_raw_sha256": args.expected_execution_input_sha256,
        "source_bindings": {
            "executor_sha256": code["executor_sha256"],
            "aggregator_sha256": code["aggregator_sha256"],
            "fault_module_sha256": execution_input["suite_binding"]["fault_module_sha256"],
            "launcher_sha256": execution_input["suite_binding"]["launcher_sha256"],
            "imported_rr2_code_ledger_raw_sha256": code["imported_rr2_code_ledger_raw_sha256"],
        },
        "hardware": runtime.hardware,
        "input_receipt": runtime.input_receipt,
        "discarded_warmup": {**warmup, "post_warmup_allocator_baseline": dict(runtime.allocator_baseline)},
        "lanes": lanes,
        "operational_invalid_count": operational_invalid_count,
        "detection_rate_reported": False,
        "naturally_occurring_claimed": False,
        "upstream_implementation_evaluated": False,
    }
    write_json_atomic(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--suite", type=Path, required=True)
    value.add_argument("--expected-suite-raw-sha256", required=True)
    value.add_argument("--expected-suite-canonical-sha256", required=True)
    value.add_argument("--execution-input", type=Path, required=True)
    value.add_argument("--expected-execution-input-sha256", required=True)
    value.add_argument("--fault-id", choices=fault_suite.FAULT_IDS, required=True)
    value.add_argument("--rank", type=int, choices=(0, 1, 2), required=True)
    value.add_argument("--expected-gpu-uuid", required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--sidecar-root", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HeldOutExecutionError",
    "INPUT_SCHEMA",
    "LANE_ORDER",
    "PRODUCTION_ASSERTION_ALLOWLIST",
    "RANK_SCHEMA",
    "ReceiptPredicateRejection",
    "SIDE_CAR_DTYPE",
    "SIDE_CAR_NBYTES",
    "SIDE_CAR_SHAPE",
    "classify_authenticated_rejection",
    "classify_production_assertion",
    "compare_semantic_outputs",
    "detector_cell",
    "restore_mutation_delta_scope",
    "run_receipt_battery",
    "sha256_file",
    "sha256_json",
    "validate_execution_input",
    "validate_completed_request_kv_horizon",
    "validate_python_environment_identity",
]
