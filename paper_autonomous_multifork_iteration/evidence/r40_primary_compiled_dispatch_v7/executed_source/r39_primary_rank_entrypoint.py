#!/usr/bin/env python3
"""Execute one immutable RR2 shard under mandatory primary-cell hooks."""

from __future__ import annotations

import argparse
import contextvars
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    if temporary.exists():
        raise SystemExit(f"temporary output already exists: {temporary}")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise SystemExit(f"refusing to overwrite output: {path}") from error
    temporary.unlink()


def _option(args: list[str], name: str) -> str:
    matches = [index for index, value in enumerate(args) if value == name]
    if len(matches) != 1 or matches[0] + 1 >= len(args):
        raise SystemExit(f"runner arguments require exactly one {name}")
    return args[matches[0] + 1]


def _load_runner(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("forkaudit_rr2_primary_runner", path)
    if spec is None or spec.loader is None:
        raise SystemExit("cannot construct immutable primary-runner module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _bind_arguments(function: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    import inspect

    bound = inspect.signature(function).bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def _install_primary_scope_wrappers(*, runner_module: Any, recorder: Any) -> Callable[[], None]:
    """Capture only the immutable runner's formal factorial cells.

    The runner invokes clean memory cells both before the factorial (discarded
    priming and four arm warmups) and after it (fault controls).  Those calls
    are scientifically intentional but are outside the primary dispatch
    receipt.  A context-local gate keeps them on the unmodified runner path
    while retaining the recorder's fail-closed rank, ordering, nesting, and
    exact-cell-count checks inside ``_run_formal_factorial_cells``.
    """

    original_factorial = runner_module._run_formal_factorial_cells
    original_memory = runner_module._run_clean_memory_cell
    original_witness = runner_module._run_ownership_witness_cell
    factorial_rank: contextvars.ContextVar[int | None] = contextvars.ContextVar(
        "forkaudit_r40_primary_factorial_rank", default=None
    )

    def factorial_wrapper(*call_args: Any, **call_kwargs: Any) -> Any:
        values = _bind_arguments(original_factorial, call_args, call_kwargs)
        rank = int(values["rank"])
        # begin_factorial deliberately runs before setting the scope token.
        # A recursive/repeated factorial therefore fails closed in the
        # recorder without altering the active outer scope.
        recorder.begin_factorial(rank)
        token = factorial_rank.set(rank)
        try:
            result = original_factorial(*call_args, **call_kwargs)
            recorder.finish_factorial()
            return result
        finally:
            factorial_rank.reset(token)

    def memory_wrapper(*call_args: Any, **call_kwargs: Any) -> Any:
        values = _bind_arguments(original_memory, call_args, call_kwargs)
        if factorial_rank.get() is None:
            return original_memory(*call_args, **call_kwargs)
        with recorder.primary_cell(
            rank=int(values["rank"]),
            resident_count=int(values["resident_count"]),
            arm_id=str(values["arm_id"]),
            kv_policy=str(values["kv_policy"]),
            gdn_base_policy=str(values["gdn_base_policy"]),
            cell_role="formal_memory",
        ):
            return original_memory(*call_args, **call_kwargs)

    def witness_wrapper(*call_args: Any, **call_kwargs: Any) -> Any:
        values = _bind_arguments(original_witness, call_args, call_kwargs)
        if factorial_rank.get() is None:
            return original_witness(*call_args, **call_kwargs)
        with recorder.primary_cell(
            rank=int(values["rank"]),
            resident_count=int(values["resident_count"]),
            arm_id=str(values["arm_id"]),
            kv_policy=str(values["kv_policy"]),
            gdn_base_policy=str(values["gdn_base_policy"]),
            cell_role="ownership_witness",
        ):
            return original_witness(*call_args, **call_kwargs)

    runner_module._run_formal_factorial_cells = factorial_wrapper
    runner_module._run_clean_memory_cell = memory_wrapper
    runner_module._run_ownership_witness_cell = witness_wrapper

    def restore() -> None:
        runner_module._run_formal_factorial_cells = original_factorial
        runner_module._run_clean_memory_cell = original_memory
        runner_module._run_ownership_witness_cell = original_witness

    return restore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--r39-base-root", type=Path, required=True)
    parser.add_argument("--primary-source-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--runtime-preflight-manifest", type=Path, required=True)
    parser.add_argument("--expected-runtime-preflight-sha256", required=True)
    parser.add_argument("--primary-launcher", type=Path, required=True)
    parser.add_argument("--launcher-rank-identity", type=Path, required=True)
    parser.add_argument("--expected-launcher-rank-identity-sha256", required=True)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    runner_args = list(args.runner_args)
    if runner_args and runner_args[0] == "--":
        runner_args.pop(0)
    if _option(runner_args, "--stage") != "shard":
        raise SystemExit("primary hook wrapper accepts only --stage shard")
    rank = int(_option(runner_args, "--rank"))
    shard_output = Path(_option(runner_args, "--output")).resolve()
    code_root = args.code_root.resolve()
    runtime_root = args.runtime_root.resolve()
    runner = args.runner.resolve()
    try:
        runner.relative_to(code_root)
    except ValueError as error:
        raise SystemExit("primary runner escapes immutable code root") from error
    for root in (args.r39_base_root.resolve(), args.primary_source_root.resolve(), code_root):
        sys.path.insert(0, str(root))

    from r39_compiled_dispatch_receipts import install_runtime_hooks
    from r39_primary_compact_dispatch import (
        PRIMARY_CODE_LEDGER_SHA256,
        PRIMARY_LAUNCHER_RELATIVE_PATH,
        PRIMARY_LAUNCHER_SHA256,
        PRIMARY_MODEL_ARTIFACT_LEDGER_SHA256,
        PRIMARY_MODEL_ID,
        PRIMARY_MODEL_REVISION,
        PRIMARY_MODEL_WEIGHT_LEDGER_SHA256,
        PRIMARY_PROTOCOL_MANIFEST_SHA256,
        PRIMARY_RUNNER_SHA256,
        PrimaryDispatchRecorder,
        verify_payload,
        verify_primary_shard,
    )
    from r40_runtime_smoke import load_and_verify_runtime_preflight

    preflight_path = args.runtime_preflight_manifest.resolve()
    if _sha256_file(preflight_path) != args.expected_runtime_preflight_sha256:
        raise SystemExit("runtime preflight raw SHA drift")
    preflight = load_and_verify_runtime_preflight(preflight_path)

    assignment_path = Path(_option(runner_args, "--gpu-assignment-receipt")).resolve()
    assignment_expected_sha = _option(
        runner_args, "--expected-gpu-assignment-receipt-raw-sha256"
    )
    if _sha256_file(assignment_path) != assignment_expected_sha:
        raise SystemExit("GPU assignment receipt raw SHA drift")
    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    rows = assignment.get("rows") if isinstance(assignment, dict) else None
    if not isinstance(rows, list) or len(rows) != 8:
        raise SystemExit("GPU assignment receipt row count drift")
    assignment_row = rows[rank]
    expected_uuid = _option(runner_args, "--expected-gpu-uuid")
    visible_uuid = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not (
        isinstance(assignment_row, dict)
        and assignment_row.get("rank") == rank
        and assignment_row.get("uuid") == expected_uuid == visible_uuid
    ):
        raise SystemExit("rank/GPU identity differs from launcher assignment")

    launcher_identity_path = args.launcher_rank_identity.resolve()
    if _sha256_file(launcher_identity_path) != args.expected_launcher_rank_identity_sha256:
        raise SystemExit("proxy launcher identity raw SHA drift")
    launcher_identity = json.loads(launcher_identity_path.read_text(encoding="utf-8"))
    if not (
        isinstance(launcher_identity, dict)
        and launcher_identity.get("schema_version")
        == "forkaudit-r40-proxy-rank-launch-v1"
        and launcher_identity.get("rank") == rank
        and launcher_identity.get("process_id") == os.getpid()
        and launcher_identity.get("parent_process_id") == os.getppid()
        and launcher_identity.get("cuda_visible_devices") == visible_uuid
        and launcher_identity.get("runner") == runner.name
    ):
        raise SystemExit("proxy launcher identity does not match executing process")

    launcher = args.primary_launcher.resolve()
    if launcher.name != PRIMARY_LAUNCHER_RELATIVE_PATH:
        raise SystemExit("primary launcher name drift")
    if _sha256_file(launcher) != PRIMARY_LAUNCHER_SHA256:
        raise SystemExit("primary launcher source hash drift")
    if _sha256_file(runner) != PRIMARY_RUNNER_SHA256:
        raise SystemExit("primary runner source hash drift")

    code_ledger = Path(_option(runner_args, "--code-ledger")).resolve()
    model_artifact_ledger = Path(
        _option(runner_args, "--model-artifact-ledger")
    ).resolve()
    model_weight_ledger = Path(_option(runner_args, "--model-weight-ledger")).resolve()
    protocol_manifest = Path(_option(runner_args, "--protocol-manifest")).resolve()
    for path, expected, label in (
        (code_ledger, PRIMARY_CODE_LEDGER_SHA256, "code ledger"),
        (
            model_artifact_ledger,
            PRIMARY_MODEL_ARTIFACT_LEDGER_SHA256,
            "model artifact ledger",
        ),
        (
            model_weight_ledger,
            PRIMARY_MODEL_WEIGHT_LEDGER_SHA256,
            "model weight ledger",
        ),
        (protocol_manifest, PRIMARY_PROTOCOL_MANIFEST_SHA256, "protocol manifest"),
    ):
        if _sha256_file(path) != expected:
            raise SystemExit(f"{label} hash drift")
    model_id = os.environ.get("R40_PRIMARY_MODEL_ID")
    model_revision = os.environ.get("R40_PRIMARY_MODEL_REVISION")
    if model_id != PRIMARY_MODEL_ID or model_revision != PRIMARY_MODEL_REVISION:
        raise SystemExit("model identity/revision binding drift")

    capture = args.capture_root.resolve() / f"rank-{rank}"
    if capture.exists():
        raise SystemExit(f"rank capture root already exists: {capture}")
    cache_root = capture / "runtime-cache" / "triton"
    cache_root.mkdir(parents=True)
    os.environ["TRITON_CACHE_DIR"] = str(cache_root)
    recorder = PrimaryDispatchRecorder(
        cache_root=cache_root,
        code_root=code_root,
        runtime_root=runtime_root,
    )
    restore_runtime = install_runtime_hooks(recorder)
    if recorder.dispatch_source_bindings != preflight["dispatch_source_bindings"]:
        restore_runtime()
        raise SystemExit("rank callable/source bindings differ from no-CUDA preflight")
    if recorder.hook_installation != preflight["hook_installation"]:
        restore_runtime()
        raise SystemExit("rank hook installation differs from no-CUDA preflight")
    runner_module = _load_runner(runner)

    restore_primary_scope = _install_primary_scope_wrappers(
        runner_module=runner_module,
        recorder=recorder,
    )
    try:
        status = runner_module.main(runner_args)
    finally:
        restore_primary_scope()
        restore_runtime()
    if status not in (None, 0):
        return int(status)
    if not shard_output.is_file():
        raise SystemExit("immutable primary runner did not write its shard")
    payload = recorder.payload()
    runner_argv_sha256 = hashlib.sha256(
        json.dumps(runner_args, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    payload["execution_binding"] = {
        "runner_relative_path": runner.relative_to(code_root).as_posix(),
        "runner_sha256": _sha256_file(runner),
        "runner_argv": runner_args,
        "runner_argv_sha256": runner_argv_sha256,
        "primary_shard_path": str(shard_output),
        "primary_shard_sha256": _sha256_file(shard_output),
        "launcher_relative_path": launcher.name,
        "launcher_sha256": _sha256_file(launcher),
        "code_ledger_path": str(code_ledger),
        "code_ledger_sha256": _sha256_file(code_ledger),
        "model_artifact_ledger_path": str(model_artifact_ledger),
        "model_artifact_ledger_sha256": _sha256_file(model_artifact_ledger),
        "model_weight_ledger_path": str(model_weight_ledger),
        "model_weight_ledger_sha256": _sha256_file(model_weight_ledger),
        "protocol_manifest_path": str(protocol_manifest),
        "protocol_manifest_sha256": _sha256_file(protocol_manifest),
        "model_id": model_id,
        "model_revision": model_revision,
        "runtime_preflight_manifest_path": str(preflight_path),
        "runtime_preflight_manifest_sha256": _sha256_file(preflight_path),
    }
    payload["rank_identity"] = {
        "schema_version": "forkaudit-r40-rank-launch-identity-v1",
        "rank": rank,
        "process_id": os.getpid(),
        "parent_process_id": os.getppid(),
        "cuda_visible_devices": visible_uuid,
        "assigned_gpu_uuid": expected_uuid,
        "gpu_assignment_receipt_path": str(assignment_path),
        "gpu_assignment_receipt_raw_sha256": assignment_expected_sha,
        "gpu_assignment_row": assignment_row,
        "launcher_identity_path": str(launcher_identity_path),
        "launcher_identity_raw_sha256": args.expected_launcher_rank_identity_sha256,
        "launcher_identity": launcher_identity,
    }
    verify_payload(
        payload,
        cache_root=cache_root,
        code_root=code_root,
        runtime_root=runtime_root,
        expected_rank=rank,
        expected_source_bindings=preflight["dispatch_source_bindings"],
        expected_gpu_assignment_receipt=assignment,
        expected_gpu_assignment_raw_sha256=assignment_expected_sha,
        expected_launcher_identity=launcher_identity,
        expected_launcher_identity_raw_sha256=args.expected_launcher_rank_identity_sha256,
        expected_runtime_preflight_sha256=args.expected_runtime_preflight_sha256,
    )
    verify_primary_shard(
        json.loads(shard_output.read_text(encoding="utf-8")),
        expected_rank=rank,
        receipt=payload,
    )
    _write_json_atomic(capture / "raw" / "primary-compiled-dispatch-receipt.json", payload)
    _write_json_atomic(
        capture / "invocation.json",
        {
            "schema_version": "forkaudit-r40-primary-rank-invocation-v7",
            "rank": rank,
            "process_id": os.getpid(),
            "assigned_gpu_uuid": expected_uuid,
            "runner_sha256": _sha256_file(runner),
            "launcher_sha256": _sha256_file(launcher),
            "runner_argv": runner_args,
            "primary_shard_sha256": _sha256_file(shard_output),
            "runtime_preflight_manifest_sha256": args.expected_runtime_preflight_sha256,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
