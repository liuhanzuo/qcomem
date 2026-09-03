from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "executed_source"))
from r40_real_binding import ActualBindingVerifier, RealBindingError, digest, storage_key  # noqa: E402


def selected():
    return json.loads((ROOT / "preregistration.json").read_text())["selected_coordinates"]


def owner(offset: int):
    layers = []
    for layer in range(3):
        base = offset + layer * 20
        layers.append(SimpleNamespace(
            conv_states={0: torch.tensor([base + 1, base + 2], dtype=torch.bfloat16)},
            recurrent_states={0: torch.tensor([base + 3, base + 4], dtype=torch.bfloat16)},
        ))
    return SimpleNamespace(layers=layers)


def clean_objects():
    persistent = owner(100)
    requests = []
    for _ in range(8):
        request = owner(0)
        for layer in range(3):
            request.layers[layer].conv_states[0] = persistent.layers[layer].conv_states[0].clone()
            request.layers[layer].recurrent_states[0] = persistent.layers[layer].recurrent_states[0].clone()
        requests.append(request)
    return persistent, SimpleNamespace(requests=requests)


def serialized(persistent, group, phase):
    storages = {}
    rows = []
    for owner_kind, request_index, value in [("persistent", None, persistent)] + [("request", i, r) for i, r in enumerate(group.requests)]:
        for layer in range(3):
            for family in ("conv", "recurrent"):
                tensor = getattr(value.layers[layer], family + "_states")[0]
                key = storage_key(tensor)
                if key not in storages:
                    storages[key] = f"storage-{len(storages):04d}"
                rows.append({
                    "owner_kind": owner_kind, "request_index": request_index,
                    "layer_index": layer, "state_family": family, "state_index": 0,
                    "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                    "content_sha256": digest(tensor), "storage_id": storages[key],
                })
    return {"phase": phase, "storage_witness": {"rows": rows}}


class RealBindingTests(unittest.TestCase):
    def test_clean_real_group_and_actual_phase_rows_pass(self):
        persistent, group = clean_objects()
        verifier = ActualBindingVerifier(persistent, selected())
        verifier.verify_built_group(group)
        verifier.verify_serialized_phase(serialized(persistent, group, "setup_pre_transition"), "setup_pre_transition")
        for row in selected():
            if row["owner_kind"] == "request" and row["request_index"] == 0:
                layer = group.requests[0].layers[row["layer_index"]]
                mapping = layer.conv_states if row["state_family"] == "conv" else layer.recurrent_states
                mapping[0] = mapping[0].clone().add_(1)
        verifier.verify_serialized_phase(serialized(persistent, group, "post_transition"), "post_transition")

    def test_real_builder_coherent_cross_layer_swap_fails(self):
        persistent, group = clean_objects()
        group.requests[0].layers[0].conv_states[0], group.requests[0].layers[1].conv_states[0] = group.requests[0].layers[1].conv_states[0], group.requests[0].layers[0].conv_states[0]
        with self.assertRaisesRegex(RealBindingError, "coordinate/content"):
            ActualBindingVerifier(persistent, selected()).verify_built_group(group)

    def test_real_builder_request_base_alias_fails(self):
        persistent, group = clean_objects()
        group.requests[1].layers[0].recurrent_states[0] = persistent.layers[0].recurrent_states[0]
        with self.assertRaisesRegex(RealBindingError, "request/base alias"):
            ActualBindingVerifier(persistent, selected()).verify_built_group(group)

    def test_real_post_rebind_stale_mapping_fails(self):
        persistent, group = clean_objects()
        verifier = ActualBindingVerifier(persistent, selected())
        verifier.verify_built_group(group)
        with self.assertRaisesRegex(RealBindingError, "stale pre-transition"):
            verifier.verify_serialized_phase(serialized(persistent, group, "post_transition"), "post_transition")

    def test_real_builder_same_geometry_one_way_mapping_error_fails(self):
        persistent, group = clean_objects()
        group.requests[0].layers[2].conv_states[0] = group.requests[0].layers[1].conv_states[0].clone()
        with self.assertRaisesRegex(RealBindingError, "coordinate/content"):
            ActualBindingVerifier(persistent, selected()).verify_built_group(group)

    def test_actual_phase_serializer_row_tamper_fails(self):
        persistent, group = clean_objects()
        verifier = ActualBindingVerifier(persistent, selected())
        verifier.verify_built_group(group)
        value = serialized(persistent, group, "setup_pre_transition")
        next(row for row in value["storage_witness"]["rows"] if row["owner_kind"] == "request" and row["request_index"] == 0 and row["layer_index"] == 0 and row["state_family"] == "conv")["content_sha256"] = "0" * 64
        with self.assertRaisesRegex(RealBindingError, "serializer content"):
            verifier.verify_serialized_phase(value, "setup_pre_transition")

    def test_completed_request_aliasing_incomplete_peer_fails(self):
        persistent, group = clean_objects()
        verifier = ActualBindingVerifier(persistent, selected())
        verifier.verify_built_group(group)
        for row in selected():
            if row["owner_kind"] == "request" and row["request_index"] == 0:
                layer = group.requests[0].layers[row["layer_index"]]
                mapping = layer.conv_states if row["state_family"] == "conv" else layer.recurrent_states
                mapping[0] = mapping[0].clone().add_(1)
        group.requests[0].layers[0].conv_states[0] = group.requests[1].layers[0].conv_states[0]
        with self.assertRaisesRegex(RealBindingError, "requests alias"):
            verifier.verify_serialized_phase(serialized(persistent, group, "post_transition"), "post_transition")

    def test_forged_normalized_serializer_storage_id_fails(self):
        persistent, group = clean_objects()
        verifier = ActualBindingVerifier(persistent, selected())
        verifier.verify_built_group(group)
        value = serialized(persistent, group, "setup_pre_transition")
        value["storage_witness"]["rows"][0]["storage_id"] = "storage-9999"
        with self.assertRaisesRegex(RealBindingError, "normalized storage_id"):
            verifier.verify_serialized_phase(value, "setup_pre_transition")

    def test_persistent_mutation_after_prebuild_freeze_fails(self):
        persistent, group = clean_objects()
        verifier = ActualBindingVerifier(persistent, selected())
        verifier.verify_built_group(group)
        persistent.layers[0].recurrent_states[0].add_(7)
        with self.assertRaisesRegex(RealBindingError, "persistent source content"):
            verifier.verify_serialized_phase(serialized(persistent, group, "setup_pre_transition"), "setup_pre_transition")


if __name__ == "__main__":
    unittest.main()
