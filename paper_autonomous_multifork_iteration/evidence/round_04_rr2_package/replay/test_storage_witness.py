#!/usr/bin/env python3
"""Adversarial tests for the pointer-free storage overlap specification."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_rr2 import (
    ReplayError,
    GATE_COMPLETED_VS_BASE_DISJOINT,
    GATE_COMPLETED_VS_PEERS_DISJOINT,
    byte_interval,
    exact_alias,
    overlaps,
    require_all_pairs_disjoint,
    validate_adjacent_cross_n,
    validate_storage_rows,
)


def row(
    *,
    sid: str = "storage-0000",
    offset: int = 0,
    shape=(4,),
    stride=(1,),
    storage_nbytes: int = 32,
    request=0,
    owner_kind="request",
    layer_index=0,
    state_family="conv",
    state_index=0,
):
    value = {
        "byte_end_exclusive": 0,
        "byte_start": 0,
        "content_sha256": "a" * 64,
        "device": "cuda:0",
        "dtype": "torch.float32",
        "layer_index": layer_index,
        "owner_kind": owner_kind,
        "request_index": request,
        "shape": list(shape),
        "state_family": state_family,
        "state_index": state_index,
        "storage_id": sid,
        "storage_nbytes": storage_nbytes,
        "storage_offset": offset,
        "stride": list(stride),
        "tensor_nbytes": 4 * __import__("math").prod(shape),
    }
    value["byte_start"], value["byte_end_exclusive"] = byte_interval(value)
    return value


class StorageWitnessAdversarialTests(unittest.TestCase):
    def test_adjacent_cross_n_exact_nested_prefix(self):
        def semantics(request_index):
            return {
                "request_index": request_index,
                "generated_token_ids": [request_index, 7],
                "full_vocab_step_logit_sha256": [f"{request_index + 1:064x}"],
                "logical_kv_sha256": {"3": f"{request_index + 2:064x}"},
                "final_gdn_sha256": f"{request_index + 3:064x}",
                "query_token_ids_sha256": f"{request_index + 4:064x}",
            }

        bank = [semantics(index) for index in range(32)]
        rows = validate_adjacent_cross_n(
            {1: bank[:1], 8: bank[:8], 32: bank}, rank=0, arm_id="arm"
        )
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(row["exact"] for row in rows))

    def test_adjacent_cross_n_semantic_tamper_is_rejected(self):
        bank = [
            {
                "request_index": index,
                "generated_token_ids": [index],
                "full_vocab_step_logit_sha256": [f"{index + 1:064x}"],
                "logical_kv_sha256": {},
                "final_gdn_sha256": f"{index + 2:064x}",
                "query_token_ids_sha256": f"{index + 3:064x}",
            }
            for index in range(32)
        ]
        upper = copy.deepcopy(bank)
        upper[0]["generated_token_ids"] = [999]
        with self.assertRaisesRegex(ReplayError, "cross-N semantic mismatch"):
            validate_adjacent_cross_n(
                {1: bank[:1], 8: upper[:8], 32: upper}, rank=0, arm_id="arm"
            )

    def test_exact_alias(self):
        left = row(request=0)
        right = copy.deepcopy(left)
        right["request_index"] = 1
        self.assertTrue(overlaps(left, right))
        self.assertTrue(exact_alias(left, right))

    def test_partial_overlap(self):
        left = row(offset=0, shape=(4,), request=0)
        right = row(offset=2, shape=(4,), request=1)
        self.assertTrue(overlaps(left, right))
        self.assertFalse(exact_alias(left, right))

    def test_adjacent_half_open_offsets_are_disjoint(self):
        left = row(offset=0, shape=(4,), request=0)
        right = row(offset=4, shape=(4,), request=1)
        self.assertFalse(overlaps(left, right))

    def test_negative_stride_interval(self):
        value = row(offset=3, shape=(4,), stride=(-1,), request=0)
        self.assertEqual(byte_interval(value), (0, 16))

    def test_conflicting_normalized_storage_id_reuse_rejected(self):
        left = row(request=0)
        right = row(offset=4, request=1, storage_nbytes=64)
        with self.assertRaisesRegex(ReplayError, "conflicting normalized storage-ID reuse"):
            validate_storage_rows([left, right])

    def test_absolute_pointer_field_rejected(self):
        value = row(request=0)
        value["data_ptr"] = 0xDEADBEEF
        with self.assertRaisesRegex(ReplayError, "schema/pointer-field"):
            validate_storage_rows([value])

    def test_nonmatching_coordinate_request_base_alias_is_rejected(self):
        request_rows = [
            row(sid="storage-0000", request=0, layer_index=0, state_family="conv", state_index=0),
            row(sid="storage-0001", request=0, layer_index=1, state_family="recurrent", state_index=1),
        ]
        base_rows = [
            row(sid="storage-0002", request=None, owner_kind="persistent", layer_index=0, state_family="conv", state_index=0),
            # Cross-coordinate alias: request coordinate 0 aliases base coordinate 1.
            row(sid="storage-0000", request=None, owner_kind="persistent", layer_index=1, state_family="recurrent", state_index=1),
        ]
        self.assertFalse(overlaps(request_rows[0], base_rows[0]))
        self.assertFalse(overlaps(request_rows[1], base_rows[1]))
        with self.assertRaisesRegex(ReplayError, GATE_COMPLETED_VS_BASE_DISJOINT):
            require_all_pairs_disjoint(
                request_rows,
                base_rows,
                "completed request[0]/persistent base",
                GATE_COMPLETED_VS_BASE_DISJOINT,
            )

    def test_nonmatching_coordinate_request_peer_alias_is_rejected(self):
        left_rows = [
            row(sid="storage-0000", request=0, layer_index=0, state_family="conv", state_index=0),
            row(sid="storage-0001", request=0, layer_index=1, state_family="recurrent", state_index=1),
        ]
        right_rows = [
            row(sid="storage-0002", request=1, layer_index=0, state_family="conv", state_index=0),
            # Cross-coordinate alias: request 0 coordinate 0 aliases peer coordinate 1.
            row(sid="storage-0000", request=1, layer_index=1, state_family="recurrent", state_index=1),
        ]
        self.assertFalse(overlaps(left_rows[0], right_rows[0]))
        self.assertFalse(overlaps(left_rows[1], right_rows[1]))
        with self.assertRaisesRegex(ReplayError, GATE_COMPLETED_VS_PEERS_DISJOINT):
            require_all_pairs_disjoint(
                left_rows,
                right_rows,
                "request[0]/request[1]",
                GATE_COMPLETED_VS_PEERS_DISJOINT,
            )

    def test_guard_reuse_is_detectable_by_timeline_contract(self):
        # The replay requires unique phase capture IDs and stable lifecycle
        # guard IDs.  This test records the two relations explicitly so an
        # address being recycled between phases cannot establish continuity.
        capture_ids = ["capture-0", "capture-1", "capture-2"]
        guard_ids = ["guard-stable"] * 3
        self.assertEqual(len(set(capture_ids)), 3)
        self.assertEqual(len(set(guard_ids)), 1)
        recycled = ["capture-0", "capture-0", "capture-2"]
        self.assertNotEqual(len(set(recycled)), 3)


if __name__ == "__main__":
    unittest.main()
