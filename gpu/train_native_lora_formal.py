from __future__ import annotations

"""Run train_qcomem_lora with an in-job, post-update step-one hard gate.

The training implementation remains the shared audited path.  This wrapper
only observes LoRA gradients/FP32 updates and refuses to enter step two unless
all eight ranks, the native cache/version audit, warm start, and memory
headroom pass together.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

import train_qcomem_lora as training
from qcomem_native_lora_protocol import evaluate_step1_gate


_BASE_ADAMW = torch.optim.AdamW
_BASE_INSTALL = training.install_suffix_lora
_BASE_WARM_START = training.load_adapter_warm_start
_AUDIT_PARAMETERS: dict[str, torch.nn.Parameter] = {}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=path.name, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _step_zero_checkpoint_audit(path: Path) -> dict[str, Any]:
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        checks = {
            "mapping_payload": isinstance(loaded, dict),
            "format": isinstance(loaded, dict)
            and loaded.get("format") == "qcomem_suffix_lora_v1",
            "step": isinstance(loaded, dict) and loaded.get("step") == 0,
            "lora_present": isinstance(loaded, dict)
            and isinstance(loaded.get("lora"), dict)
            and bool(loaded["lora"]),
            "optimizer_absent": isinstance(loaded, dict)
            and "optimizer" not in loaded,
            "metadata_step": isinstance(loaded, dict)
            and isinstance(loaded.get("metadata"), dict)
            and loaded["metadata"].get("last_step") == 0,
        }
        return {
            "rank": _rank(),
            "ok": all(checks.values()),
            "sha256": digest,
            "checks": checks,
            "error": None,
        }
    except Exception as error:  # pragma: no cover - exercised through caller
        return {
            "rank": _rank(),
            "ok": False,
            "sha256": None,
            "checks": {},
            "error": f"{type(error).__name__}: {error}",
        }


def distributed_atomic_torch_save(path: Path, payload: dict[str, Any]) -> str:
    """Rank zero writes once; every rank collectively verifies the artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    write_result: dict[str, Any] | None = None
    if _rank() == 0:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=path.name, suffix=".tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
            torch.save(payload, temporary)
            os.replace(temporary, path)
            write_result = {"ok": True, "error": None}
        except Exception as error:
            write_result = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if _world_size() > 1:
        decision: list[Any] = [write_result]
        dist.broadcast_object_list(decision, src=0)
        write_result = decision[0]
    if not isinstance(write_result, dict) or not write_result.get("ok"):
        detail = write_result.get("error") if isinstance(write_result, dict) else None
        raise RuntimeError(f"rank-zero checkpoint write failed: {detail}")
    if _world_size() > 1:
        dist.barrier()
    local_audit = _step_zero_checkpoint_audit(path)
    audits: list[Any] = [None] * _world_size()
    if _world_size() > 1:
        dist.all_gather_object(audits, local_audit)
    else:
        audits[0] = local_audit
    digests = {row.get("sha256") for row in audits if isinstance(row, dict)}
    if (
        any(not isinstance(row, dict) or not row.get("ok") for row in audits)
        or len(digests) != 1
        or None in digests
    ):
        raise RuntimeError(f"distributed step-zero checkpoint audit failed: {audits}")
    return next(iter(digests))


def audited_install(*args: Any, **kwargs: Any) -> list[str]:
    installed = _BASE_INSTALL(*args, **kwargs)
    model = args[0] if args else kwargs["model"]
    global _AUDIT_PARAMETERS
    _AUDIT_PARAMETERS = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.endswith(("lora_a", "lora_b"))
    }
    return installed


def audited_warm_start(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Persist a target-semantic step-zero control after loading Interface LoRA."""

    result = _BASE_WARM_START(*args, **kwargs)
    model = kwargs["model"]
    target_metadata = kwargs["target_metadata"]
    output_raw = os.environ.get("NATIVE_LORA_STEP0_CHECKPOINT")
    if not output_raw:
        raise RuntimeError("NATIVE_LORA_STEP0_CHECKPOINT is required")
    output = Path(output_raw)
    metadata = {
        **target_metadata,
        "warm_start": result,
        "last_step": 0,
        "native_initialization_control": {
            "target_execution": "native-functional-cache",
            "optimizer_steps": 0,
            "source_interface_checkpoint_step": result["source_step"],
            "historical_merged_checkpoint_reused": False,
        },
    }
    payload = {
        "format": "qcomem_suffix_lora_v1",
        "step": 0,
        "lora": training.lora_state_dict(model),
        "metadata": metadata,
    }
    step_zero_sha256 = distributed_atomic_torch_save(output, payload)
    return {
        **result,
        "native_step_zero_checkpoint": str(output),
        "native_step_zero_checkpoint_sha256": step_zero_sha256,
        "native_step_zero_verified_by_all_ranks": True,
    }


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


class AuditedAdamW(_BASE_ADAMW):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._native_step_one_audited = False

    def step(self, closure=None):
        if self._native_step_one_audited:
            return super().step(closure=closure)
        expected_modules = int(os.environ.get("NATIVE_LORA_EXPECTED_MODULES", "36"))
        expected_tensors = int(
            os.environ.get("NATIVE_LORA_EXPECTED_PARAMETER_TENSORS", "72")
        )
        if len(_AUDIT_PARAMETERS) != expected_tensors:
            raise RuntimeError(
                f"step-one audit found {len(_AUDIT_PARAMETERS)} LoRA tensors, "
                f"expected {expected_tensors}"
            )
        modules = {name.rsplit(".", 1)[0] for name in _AUDIT_PARAMETERS}
        if len(modules) != expected_modules:
            raise RuntimeError(
                f"step-one audit found {len(modules)} LoRA modules, "
                f"expected {expected_modules}"
            )
        before: dict[str, torch.Tensor] = {}
        gradients = []
        for name, parameter in sorted(_AUDIT_PARAMETERS.items()):
            gradient = parameter.grad
            gradients.append(
                {
                    "name": name,
                    "present": gradient is not None,
                    "finite": bool(
                        gradient is not None and torch.isfinite(gradient).all().item()
                    ),
                    "nonzero": bool(
                        gradient is not None
                        and torch.count_nonzero(gradient).item() > 0
                    ),
                    "norm": (
                        float(
                            torch.linalg.vector_norm(
                                gradient.detach(), dtype=torch.float32
                            ).item()
                        )
                        if gradient is not None
                        else None
                    ),
                }
            )
            before[name] = parameter.detach().cpu().clone()

        result = super().step(closure=closure)
        updates = []
        for name, parameter in sorted(_AUDIT_PARAMETERS.items()):
            current = parameter.detach().cpu()
            delta = current - before[name]
            updates.append(
                {
                    "name": name,
                    "finite": bool(
                        torch.isfinite(current).all().item()
                        and torch.isfinite(delta).all().item()
                    ),
                    "nonzero": bool(torch.count_nonzero(delta).item() > 0),
                    "delta_norm": float(
                        torch.linalg.vector_norm(delta, dtype=torch.float32).item()
                    ),
                    "max_abs_delta": float(delta.float().abs().max().item()),
                }
            )
        device = torch.cuda.current_device()
        total = int(torch.cuda.get_device_properties(device).total_memory)
        max_allocated = int(torch.cuda.max_memory_allocated(device))
        max_reserved = int(torch.cuda.max_memory_reserved(device))
        local = {
            "rank": _rank(),
            "modules": len(modules),
            "parameter_tensors": len(_AUDIT_PARAMETERS),
            "finite_gradient_tensors": sum(row["finite"] for row in gradients),
            "nonzero_gradient_tensors": sum(row["nonzero"] for row in gradients),
            "finite_update_tensors": sum(row["finite"] for row in updates),
            "nonzero_update_tensors": sum(row["nonzero"] for row in updates),
            "total_memory_bytes": total,
            "max_allocated_bytes": max_allocated,
            "max_reserved_bytes": max_reserved,
            "headroom_bytes": total - max_reserved,
            "gradients": gradients,
            "updates": updates,
        }
        world_size = _world_size()
        records: list[Any] = [None] * world_size
        if world_size > 1:
            dist.all_gather_object(records, local)
            dist.barrier()
        else:
            records[0] = local

        gate_path_raw = os.environ.get("NATIVE_LORA_STEP1_GATE_FILE")
        init_sha = os.environ.get("EXPECTED_INIT_ADAPTER_SHA256")
        if not gate_path_raw or not init_sha:
            raise RuntimeError(
                "NATIVE_LORA_STEP1_GATE_FILE and EXPECTED_INIT_ADAPTER_SHA256 are required"
            )
        gate_path = Path(gate_path_raw)
        gate: dict[str, Any] | None = None
        if _rank() == 0:
            metadata_path = gate_path.parent / "metadata.json"
            if not metadata_path.is_file():
                raise RuntimeError("rank-zero step-one metadata was not written")
            metadata = json.loads(metadata_path.read_text())
            gate = evaluate_step1_gate(
                records,
                metadata,
                expected_world_size=world_size,
                expected_modules=expected_modules,
                expected_parameter_tensors=expected_tensors,
                minimum_headroom_bytes=int(
                    os.environ.get(
                        "NATIVE_LORA_MINIMUM_HEADROOM_BYTES", "4294967296"
                    )
                ),
                expected_init_checkpoint_sha256=init_sha,
            )
            gate.update(
                {
                    "training_continues_in_same_job_after_pass": True,
                    "separate_smoke_job_used": False,
                    "historical_merged_checkpoint_reused": False,
                }
            )
            atomic_json(gate_path, gate)
        decision: list[Any] = [gate]
        if world_size > 1:
            dist.broadcast_object_list(decision, src=0)
        gate = decision[0]
        if not isinstance(gate, dict) or gate.get("status") != "passed":
            raise RuntimeError("native LoRA post-update step-one hard gate failed")
        self._native_step_one_audited = True
        return result


def main() -> None:
    training.install_suffix_lora = audited_install
    training.load_adapter_warm_start = audited_warm_start
    training.torch.optim.AdamW = AuditedAdamW
    training.main()


if __name__ == "__main__":
    main()
