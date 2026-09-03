from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


MANIFEST_SCHEMA = "qcomem-sft-dcp-checkpoint-manifest-v1"
EVAL_MODEL_ONLY_CONTRACT = "eval_model_only_fp32"
FULL_RESUME_CONTRACT = "resume_full_fp32"


class DCPCheckpointError(RuntimeError):
    pass


def stable_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=path.name, delete=False
    ) as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def checkpoint_payload_entries(root: Path) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.name not in {"checkpoint-manifest.json", "_SUCCESS"}
        ),
        key=lambda value: value.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not entries:
        raise DCPCheckpointError("DCP produced no payload files")
    return entries


def payload_directory_sha256(entries: list[dict[str, Any]]) -> str:
    """Digest the sorted DCP payload index; the manifest is excluded.

    For every lexicographically sorted POSIX relative path, hash
    ``path + NUL + decimal_size + NUL + file_sha256 + LF``.  Excluding the
    manifest avoids a circular digest while still binding every DCP byte.
    """

    digest = hashlib.sha256()
    paths = [entry.get("path") for entry in entries]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise DCPCheckpointError("checkpoint payload entries must be sorted and unique")
    for entry in entries:
        path = entry.get("path")
        size = entry.get("size_bytes")
        sha = entry.get("sha256")
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or ".." in Path(path).parts
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(sha, str)
            or len(sha) != 64
        ):
            raise DCPCheckpointError("invalid checkpoint payload entry")
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _require_distributed() -> tuple[int, int]:
    if not dist.is_initialized():
        raise DCPCheckpointError("DCP checkpoint requires an initialized process group")
    return dist.get_rank(), dist.get_world_size()


def save_eval_model_only_fp32(
    model: torch.nn.Module,
    checkpoint_root: Path,
    *,
    step: int,
    expected_parameters: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Collectively save sharded FP32 model state without a rank-0 gather."""

    rank, world_size = _require_distributed()
    if step < 1 or expected_parameters < 1:
        raise DCPCheckpointError("checkpoint step/parameter count must be positive")
    local_parameters = sum(parameter.numel() for parameter in model.parameters())
    if {parameter.dtype for parameter in model.parameters()} != {torch.float32}:
        raise DCPCheckpointError("eval-model-only DCP requires persistent FP32 shards")
    device = next(model.parameters()).device
    global_parameters = torch.tensor(local_parameters, dtype=torch.int64, device=device)
    dist.all_reduce(global_parameters)
    if int(global_parameters.item()) != expected_parameters:
        raise DCPCheckpointError(
            "eval-model-only DCP parameter coverage mismatch: "
            f"expected={expected_parameters}, actual={int(global_parameters.item())}"
        )
    parent = checkpoint_root.parent
    preparation_error: list[Any] = [None]
    if rank == 0:
        try:
            if checkpoint_root.exists():
                raise DCPCheckpointError(
                    f"refusing existing checkpoint {checkpoint_root}"
                )
            parent.mkdir(parents=True, exist_ok=True)
        except Exception as error:
            preparation_error[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(preparation_error, src=0)
    if preparation_error[0] is not None:
        raise DCPCheckpointError(preparation_error[0])
    nonce: list[Any] = [uuid.uuid4().hex if rank == 0 else None]
    dist.broadcast_object_list(nonce, src=0)
    temporary = parent / f".{checkpoint_root.name}.incomplete-{nonce[0]}"
    temporary_error: list[Any] = [None]
    if rank == 0 and temporary.exists():
        temporary_error[0] = f"refusing existing temporary checkpoint {temporary}"
    dist.broadcast_object_list(temporary_error, src=0)
    if temporary_error[0] is not None:
        raise DCPCheckpointError(temporary_error[0])
    dist.barrier()

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    options = StateDictOptions(
        full_state_dict=False,
        cpu_offload=False,
        strict=True,
    )
    model_state = get_model_state_dict(model, options=options)
    dcp.save({"model": model_state}, checkpoint_id=temporary)
    dist.barrier()
    manifest: dict[str, Any] | None = None
    if rank == 0:
        entries = checkpoint_payload_entries(temporary)
        payload_sha256 = payload_directory_sha256(entries)
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "contract": EVAL_MODEL_ONLY_CONTRACT,
            "step": step,
            "world_size": world_size,
            "global_parameter_count": expected_parameters,
            "persistent_parameter_dtype": "torch.float32",
            "logical_model_bytes": expected_parameters * 4,
            "actual_payload_bytes": sum(entry["size_bytes"] for entry in entries),
            "state": {
                "root_key": "model",
                "model": True,
                "optimizer": False,
                "scheduler": False,
                "rng": False,
            },
            "state_dict_api": (
                "torch.distributed.checkpoint.state_dict.get_model_state_dict"
            ),
            "dcp_api": "torch.distributed.checkpoint.save",
            "state_dict_options": {
                "full_state_dict": False,
                "cpu_offload": False,
                "strict": True,
            },
            "rank0_full_gather_used": False,
            "reshardable": True,
            "provenance": provenance,
            "payload_digest_definition": (
                "sha256 over sorted entries: path_utf8 + NUL + decimal_size + "
                "NUL + file_sha256_ascii + LF; checkpoint-manifest.json excluded"
            ),
            "payload_directory_sha256": payload_sha256,
            "payload_files": entries,
        }
        manifest["manifest_preimage_sha256"] = hashlib.sha256(
            stable_json(manifest).encode("utf-8")
        ).hexdigest()
        _atomic_json(temporary / "checkpoint-manifest.json", manifest)
        manifest_sha256 = sha256_file(temporary / "checkpoint-manifest.json")
        (temporary / "_SUCCESS").write_text(
            stable_json(
                {
                    "schema_version": "qcomem-sft-dcp-success-v1",
                    "checkpoint_manifest_sha256": manifest_sha256,
                    "payload_directory_sha256": payload_sha256,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, checkpoint_root)
        pointer = {
            "schema_version": "qcomem-sft-dcp-latest-v1",
            "contract": EVAL_MODEL_ONLY_CONTRACT,
            "step": step,
            "checkpoint_directory": checkpoint_root.name,
            "checkpoint_manifest_sha256": manifest_sha256,
            "payload_directory_sha256": payload_sha256,
        }
        _atomic_json(parent / "latest-eval-model-only.json", pointer)
        manifest["checkpoint_manifest_sha256"] = manifest_sha256
        manifest["checkpoint_path"] = str(checkpoint_root)
        manifest["payload_integrity_verified_once_at_save"] = True
        manifest["atomic_publish_completed"] = True
        manifest["success_marker_written"] = True
    dist.barrier()
    values: list[Any] = [manifest]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def validate_checkpoint_manifest(
    checkpoint_root: Path,
    *,
    expected_contract: str = EVAL_MODEL_ONLY_CONTRACT,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    manifest_path = checkpoint_root / "checkpoint-manifest.json"
    if not manifest_path.is_file():
        raise DCPCheckpointError("checkpoint manifest is missing")
    actual_manifest_sha256 = sha256_file(manifest_path)
    if expected_manifest_sha256 is not None and actual_manifest_sha256 != expected_manifest_sha256:
        raise DCPCheckpointError("checkpoint manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise DCPCheckpointError("checkpoint manifest schema mismatch")
    if manifest.get("contract") != expected_contract:
        raise DCPCheckpointError("checkpoint contract mismatch")
    success_path = checkpoint_root / "_SUCCESS"
    if not success_path.is_file():
        raise DCPCheckpointError("checkpoint _SUCCESS marker is missing")
    success = json.loads(success_path.read_text(encoding="utf-8"))
    if (
        not isinstance(success, dict)
        or success.get("schema_version") != "qcomem-sft-dcp-success-v1"
        or success.get("checkpoint_manifest_sha256") != actual_manifest_sha256
        or success.get("payload_directory_sha256")
        != manifest.get("payload_directory_sha256")
    ):
        raise DCPCheckpointError("checkpoint _SUCCESS marker does not bind payload")
    preimage_sha256 = manifest.get("manifest_preimage_sha256")
    preimage = dict(manifest)
    preimage.pop("manifest_preimage_sha256", None)
    if hashlib.sha256(stable_json(preimage).encode("utf-8")).hexdigest() != preimage_sha256:
        raise DCPCheckpointError("checkpoint manifest preimage SHA256 mismatch")
    expected_entries = manifest.get("payload_files")
    if not isinstance(expected_entries, list):
        raise DCPCheckpointError("checkpoint payload index is missing")
    actual_entries = checkpoint_payload_entries(checkpoint_root)
    if actual_entries != expected_entries:
        raise DCPCheckpointError("checkpoint payload file index/hash mismatch")
    directory_sha256 = payload_directory_sha256(actual_entries)
    if directory_sha256 != manifest.get("payload_directory_sha256"):
        raise DCPCheckpointError("checkpoint payload directory SHA256 mismatch")
    if sum(entry["size_bytes"] for entry in actual_entries) != manifest.get(
        "actual_payload_bytes"
    ):
        raise DCPCheckpointError("checkpoint actual payload byte count mismatch")
    return {
        **manifest,
        "checkpoint_manifest_sha256": actual_manifest_sha256,
        "checkpoint_path": str(checkpoint_root),
        "payload_integrity_verified": True,
    }


def load_eval_model_only_fp32(
    model: torch.nn.Module,
    checkpoint_root: Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Collectively validate and load a sharded eval-model-only checkpoint."""

    rank, _ = _require_distributed()
    # Hashing an approximately 129-GiB model payload on every rank would create
    # 8x redundant storage traffic.  Rank 0 performs exactly one full integrity
    # pass, then broadcasts the verified manifest before collective DCP load.
    validated: list[Any] = [None]
    validation_error: list[Any] = [None]
    if rank == 0:
        try:
            validated[0] = validate_checkpoint_manifest(
                checkpoint_root,
                expected_contract=EVAL_MODEL_ONLY_CONTRACT,
                expected_manifest_sha256=expected_manifest_sha256,
            )
        except Exception as error:  # broadcast before re-raising to avoid rank hangs
            validation_error[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(validation_error, src=0)
    if validation_error[0] is not None:
        raise DCPCheckpointError(
            f"rank-0 checkpoint integrity validation failed: {validation_error[0]}"
        )
    dist.broadcast_object_list(validated, src=0)
    manifest = validated[0]
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )

    options = StateDictOptions(
        full_state_dict=False,
        cpu_offload=False,
        strict=True,
    )
    model_state = get_model_state_dict(model, options=options)
    state = {"model": model_state}
    dcp.load(state, checkpoint_id=checkpoint_root)
    incompatible = set_model_state_dict(model, state["model"], options=options)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise DCPCheckpointError(
            "DCP model load returned incompatible keys: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return manifest
