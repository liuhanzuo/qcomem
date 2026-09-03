from __future__ import annotations

"""Read-only CPU counterexamples for the frozen R40 v4 verifier.

This program never writes files and never initializes or uses a GPU.  Before
loading v4 code it verifies the independently recorded hashes of all ten v4
source/package files and the v4 source ledger.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch


AUDIT_ROOT = Path(__file__).resolve().parents[1]
V4_ROOT = AUDIT_ROOT.parent / "r40_independent_live_binding_v4_real_binder"
V4_BINDING = AUDIT_ROOT / "V4_SOURCE_FILES.sha256"
EXPECTED_V4_LEDGER_SHA256 = "b79bbc84821ecacf09d939faadd2818ec5674df88601b97ca0bb3ffd1db4616d"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_v4_binding() -> Dict[str, Any]:
    checked: List[str] = []
    for line_number, line in enumerate(V4_BINDING.read_text(encoding="ascii").splitlines(), 1):
        if not line.strip():
            continue
        expected, relative = line.split(None, 1)
        relative = relative.strip()
        target = (AUDIT_ROOT / relative).resolve()
        try:
            target.relative_to(V4_ROOT.resolve())
        except ValueError as exc:
            raise RuntimeError(f"v4 binding path escapes v4 root at line {line_number}") from exc
        observed = _sha256(target)
        if observed != expected:
            raise RuntimeError(
                f"v4 source hash mismatch at {relative}: expected {expected}, observed {observed}"
            )
        checked.append(relative)
    if len(checked) != 10 or len(set(checked)) != 10:
        raise RuntimeError(f"external v4 binding must contain ten unique files, observed {len(checked)}")
    ledger = V4_ROOT / "source-code.sha256"
    observed_ledger = _sha256(ledger)
    if observed_ledger != EXPECTED_V4_LEDGER_SHA256:
        raise RuntimeError(
            f"v4 source ledger hash mismatch: expected {EXPECTED_V4_LEDGER_SHA256}, "
            f"observed {observed_ledger}"
        )
    return {
        "external_v4_files_verified": len(checked),
        "v4_source_ledger_sha256": observed_ledger,
    }


def _load_v4_module() -> Any:
    module_path = V4_ROOT / "executed_source" / "r40_real_binding.py"
    spec = importlib.util.spec_from_file_location("posthoc_audited_v4_real_binding", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not create import spec for frozen v4 verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selected() -> List[Dict[str, Any]]:
    value = json.loads((V4_ROOT / "preregistration.json").read_text(encoding="utf-8"))
    return [dict(row) for row in value["selected_coordinates"]]


def _owner(offset: int) -> Any:
    layers = []
    for layer_index in range(3):
        base = offset + layer_index * 20
        layers.append(
            SimpleNamespace(
                conv_states={0: torch.tensor([base + 1, base + 2], dtype=torch.bfloat16, device="cpu")},
                recurrent_states={0: torch.tensor([base + 3, base + 4], dtype=torch.bfloat16, device="cpu")},
            )
        )
    return SimpleNamespace(layers=layers)


def _clean_objects() -> Any:
    persistent = _owner(100)
    requests = []
    for _ in range(8):
        request = _owner(0)
        for layer_index in range(3):
            request.layers[layer_index].conv_states[0] = persistent.layers[layer_index].conv_states[0].clone()
            request.layers[layer_index].recurrent_states[0] = persistent.layers[layer_index].recurrent_states[0].clone()
        requests.append(request)
    return persistent, SimpleNamespace(requests=requests)


def _serialized(module: Any, persistent: Any, group: Any, phase: str) -> Dict[str, Any]:
    storages: Dict[Any, str] = {}
    rows: List[Dict[str, Any]] = []
    owners: Iterable[Any] = [("persistent", None, persistent)] + [
        ("request", request_index, request) for request_index, request in enumerate(group.requests)
    ]
    for owner_kind, request_index, value in owners:
        for layer_index in range(3):
            for family in ("conv", "recurrent"):
                tensor = getattr(value.layers[layer_index], family + "_states")[0]
                key = module.storage_key(tensor)
                if key not in storages:
                    storages[key] = f"storage-{len(storages):04d}"
                rows.append(
                    {
                        "owner_kind": owner_kind,
                        "request_index": request_index,
                        "layer_index": layer_index,
                        "state_family": family,
                        "state_index": 0,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "content_sha256": module.digest(tensor),
                        "storage_id": storages[key],
                    }
                )
    return {"phase": phase, "storage_witness": {"rows": rows}}


def _record(probe_id: str, action: Any) -> Dict[str, Any]:
    try:
        action()
    except Exception as exc:  # A repaired verifier must reach this path.
        return {
            "probe_id": probe_id,
            "observed_verifier_outcome": "REJECTED",
            "expected_secure_outcome": "REJECTED",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        }
    return {
        "probe_id": probe_id,
        "observed_verifier_outcome": "ACCEPTED",
        "expected_secure_outcome": "REJECTED",
        "exception_type": None,
        "exception_message": None,
    }


def _peer_alias_after_transition(module: Any, selected: Sequence[Mapping[str, Any]]) -> None:
    persistent, group = _clean_objects()
    verifier = module.ActualBindingVerifier(persistent, selected)
    verifier.verify_built_group(group)
    for row in selected:
        if row["owner_kind"] == "request" and int(row["request_index"]) == 0:
            layer = group.requests[0].layers[int(row["layer_index"])]
            mapping = layer.conv_states if row["state_family"] == "conv" else layer.recurrent_states
            mapping[int(row["state_index"])] = module.tensor_at(group.requests[1], row)
    gdn = _serialized(module, persistent, group, "post_transition")
    verifier.verify_serialized_phase(gdn, "post_transition")


def _forged_storage_id(module: Any, selected: Sequence[Mapping[str, Any]]) -> None:
    persistent, group = _clean_objects()
    verifier = module.ActualBindingVerifier(persistent, selected)
    verifier.verify_built_group(group)
    gdn = _serialized(module, persistent, group, "setup_pre_transition")
    row = next(
        value
        for value in gdn["storage_witness"]["rows"]
        if value["owner_kind"] == "request"
        and value["request_index"] == 0
        and value["layer_index"] == 0
        and value["state_family"] == "conv"
    )
    row["storage_id"] = "storage-9999"
    verifier.verify_serialized_phase(gdn, "setup_pre_transition")


def _persistent_mutation_after_freeze(module: Any, selected: Sequence[Mapping[str, Any]]) -> None:
    persistent, group = _clean_objects()
    verifier = module.ActualBindingVerifier(persistent, selected)
    verifier.verify_built_group(group)
    persistent.layers[0].recurrent_states[0].add_(7)
    gdn = _serialized(module, persistent, group, "setup_pre_transition")
    verifier.verify_serialized_phase(gdn, "setup_pre_transition")


def run_all() -> Dict[str, Any]:
    torch.set_grad_enabled(False)
    binding = verify_v4_binding()
    module = _load_v4_module()
    selected = _selected()
    probes = [
        _record("peer_alias_after_transition", lambda: _peer_alias_after_transition(module, selected)),
        _record("forged_storage_id", lambda: _forged_storage_id(module, selected)),
        _record(
            "persistent_mutation_after_freeze",
            lambda: _persistent_mutation_after_freeze(module, selected),
        ),
    ]
    all_accepted = all(row["observed_verifier_outcome"] == "ACCEPTED" for row in probes)
    return {
        "schema_version": "forkaudit-r40-v4-posthoc-counterexamples-v1",
        "audit_kind": "posthoc-read-only-cpu-counterexample",
        "v4_formal_status": "HOLD",
        "formal_gpu_execution": "not-run",
        "qs_invocation": "not-run",
        "positive_paper_evidence": False,
        "device": "cpu",
        **binding,
        "probes": probes,
        "all_counterexamples_accepted": all_accepted,
    }


def _render_text(value: Mapping[str, Any]) -> str:
    lines = [
        "POSTHOC_AUDIT_ONLY",
        f"V4_FORMAL_STATUS={value['v4_formal_status']}",
        "FORMAL_GPU_EXECUTION=not-run",
        "QS_INVOCATION=not-run",
        "POSITIVE_PAPER_EVIDENCE=false",
        f"EXTERNAL_V4_FILES_VERIFIED={value['external_v4_files_verified']}",
    ]
    for row in value["probes"]:
        lines.append(f"{row['probe_id']}={row['observed_verifier_outcome']}")
    lines.append(f"ALL_COUNTEREXAMPLES_ACCEPTED={str(value['all_counterexamples_accepted']).lower()}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()
    value = run_all()
    if args.format == "json":
        print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    else:
        sys.stdout.write(_render_text(value))
    return 0 if value["all_counterexamples_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
