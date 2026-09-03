from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "executed_source"))

from r40_passive_clone_lineage import PassiveCloneLineageMode, PersistentSourceRegistry  # noqa: E402
from r40_compact_rebind_fix import fresh_compact_state  # noqa: E402
from r40_real_binding import ActualBindingVerifier, BORROWED_POLICY, RealBindingError, storage_key  # noqa: E402
from test_real_binding import owner, selected, serialized  # noqa: E402


def borrowed_objects():
    persistent = owner(100)
    requests = []
    for _ in range(8):
        request = owner(0)
        for layer in range(3):
            request.layers[layer].conv_states[0] = persistent.layers[layer].conv_states[0]
            request.layers[layer].recurrent_states[0] = persistent.layers[layer].recurrent_states[0]
        requests.append(request)
    return persistent, SimpleNamespace(requests=requests)


def rebind(group, request_index):
    for layer in group.requests[request_index].layers:
        for mapping in (layer.conv_states, layer.recurrent_states):
            mapping[0] = mapping[0].clone().add_(1)


def noncompact_borrowed_objects():
    persistent = owner(100)
    for layer_index, layer in enumerate(persistent.layers):
        for family_index, mapping in enumerate((layer.conv_states, layer.recurrent_states)):
            backing = __import__("torch").arange(40, dtype=__import__("torch").bfloat16) + layer_index * 100 + family_index * 10
            mapping[0] = backing[3:7:2]
    requests = []
    for _ in range(8):
        request = owner(0)
        for layer_index, layer in enumerate(request.layers):
            layer.conv_states[0] = persistent.layers[layer_index].conv_states[0]
            layer.recurrent_states[0] = persistent.layers[layer_index].recurrent_states[0]
        requests.append(request)
    return persistent, SimpleNamespace(requests=requests)


class BorrowedTransitionBinding(unittest.TestCase):
    def verifier(self, persistent):
        return ActualBindingVerifier(persistent, selected(), setup_policy=BORROWED_POLICY)

    def test_exact_setup_transition_and_generation_lifecycle(self):
        persistent, group = borrowed_objects(); verifier = self.verifier(persistent); verifier.verify_built_group(group)
        setup = verifier.verify_serialized_phase(serialized(persistent, group, "setup_pre_transition"), "setup_pre_transition", [])
        self.assertEqual((setup["private_request_rows_verified"], setup["borrowed_request_rows_verified"]), (0, 48))
        rebind(group, 0); verifier.observe_generation_step(0, 0)
        transition = verifier.verify_serialized_phase(serialized(persistent, group, "post_transition"), "post_transition", [0])
        self.assertEqual((transition["private_request_rows_verified"], transition["borrowed_request_rows_verified"]), (6, 42))
        for request_index in range(1, 8): rebind(group, request_index); verifier.observe_generation_step(0, request_index)
        for round_index in range(1, 8):
            for request_index in range(8): rebind(group, request_index); verifier.observe_generation_step(round_index, request_index)
        final = verifier.verify_serialized_phase(serialized(persistent, group, "post_generation"), "post_generation", list(range(8)))
        self.assertEqual((final["private_request_rows_verified"], final["borrowed_request_rows_verified"]), (48, 0))
        receipt = verifier.functional_rebind_receipt()
        self.assertEqual((receipt["call_count"], receipt["edge_count"], receipt["edges_per_call"]), (64, 384, 6))
        self.assertTrue(all(receipt[key] for key in ("all_new_tensor_objects", "all_new_storages", "all_descriptors_authorized", "all_contents_recorded")))
        edge = receipt["calls"][0]["edges"][0]
        self.assertEqual(set(edge), {"coordinate", "version", "pre", "post", "new_tensor_object", "new_storage", "descriptor_authorized", "content_recorded"})
        self.assertEqual(set(edge["pre"]), {"object_id", "storage_key", "descriptor", "content_sha256"})

    def test_noncompact_borrowed_setup_authorizes_only_compact_transition(self):
        persistent, group = noncompact_borrowed_objects(); verifier = self.verifier(persistent); verifier.verify_built_group(group)
        for layer in group.requests[0].layers:
            for mapping in (layer.conv_states, layer.recurrent_states): mapping[0] = fresh_compact_state(mapping[0]).add_(1)
        verifier.observe_generation_step(0, 0)
        for descriptor in verifier.authorized_descriptors.values():
            self.assertEqual((descriptor["stride"], descriptor["storage_offset"], descriptor["storage_nbytes"], descriptor["interval"]), ([1], 0, descriptor["tensor_nbytes"], (0, descriptor["tensor_nbytes"])))

        persistent, group = noncompact_borrowed_objects(); verifier = self.verifier(persistent); verifier.verify_built_group(group)
        for layer in group.requests[0].layers:
            for mapping in (layer.conv_states, layer.recurrent_states): mapping[0] = fresh_compact_state(mapping[0]).add_(1)
        backing = __import__("torch").empty(10, dtype=group.requests[0].layers[0].conv_states[0].dtype)
        group.requests[0].layers[0].conv_states[0] = backing[2:6:2]
        with self.assertRaisesRegex(RealBindingError, "descriptor/offset/interval unauthorized"):
            verifier.observe_generation_step(0, 0)

    def test_opaque_borrowed_capability_binds_actual_aliases(self):
        persistent, group = borrowed_objects(); verifier = self.verifier(persistent); verifier.verify_built_group(group)
        coordinates = [(layer, family, 0) for layer in range(3) for family in ("conv", "recurrent")]
        registry = PersistentSourceRegistry(persistent, coordinates); mode = PassiveCloneLineageMode(registry)
        with mode: pass
        summary = mode.verify_borrowed(group.requests, coordinates)
        self.assertEqual((summary["captured_lineage_edges"], summary["request_count"]), (0, 8))
        verifier.attach_lineage_capability(mode.issue_capability(group.requests, selected(), verifier, setup_policy=BORROWED_POLICY))
        self.assertEqual(verifier.lineage_receipt["selected_exact_alias_count"], 5)

    def test_copied_setup_and_early_non_target_write_fail(self):
        persistent, group = borrowed_objects(); group.requests[1].layers[0].conv_states[0] = group.requests[1].layers[0].conv_states[0].clone()
        with self.assertRaisesRegex(RealBindingError, "exact persistent alias"): self.verifier(persistent).verify_built_group(group)
        persistent, group = borrowed_objects(); verifier = self.verifier(persistent); verifier.verify_built_group(group); rebind(group, 0); group.requests[7].layers[1].recurrent_states[0] = group.requests[7].layers[1].recurrent_states[0].clone()
        with self.assertRaisesRegex(RealBindingError, "non-target request changed"): verifier.observe_generation_step(0, 0)

    def test_completed_request_base_or_peer_alias_fails(self):
        for kind in ("base", "peer"):
            persistent, group = borrowed_objects(); verifier = self.verifier(persistent); verifier.verify_built_group(group); rebind(group, 0)
            group.requests[0].layers[0].conv_states[0] = persistent.layers[0].conv_states[0] if kind == "base" else group.requests[1].layers[0].conv_states[0]
            with self.assertRaisesRegex(RealBindingError, "fresh|alias"): verifier.observe_generation_step(0, 0)

    def test_serializer_must_report_borrowed_and_private_storage_roles(self):
        persistent, group = borrowed_objects(); verifier = self.verifier(persistent); verifier.verify_built_group(group)
        payload = serialized(persistent, group, "setup_pre_transition")
        row = next(row for row in payload["storage_witness"]["rows"] if row["owner_kind"] == "request")
        row["storage_id"] = "storage-9999"
        with self.assertRaisesRegex(RealBindingError, "normalized storage_id"): verifier.verify_serialized_phase(payload, "setup_pre_transition", [])


if __name__ == "__main__":
    unittest.main()
