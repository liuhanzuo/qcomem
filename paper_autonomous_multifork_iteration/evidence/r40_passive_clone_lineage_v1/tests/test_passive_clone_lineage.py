from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "prototype"))

from passive_clone_lineage import (  # noqa: E402
    LineageViolation,
    PassiveCloneLineageMode,
    PersistentSourceRegistry,
    storage_descriptor,
    tensor_at,
    tensor_bytes_sha256,
)


PRODUCTION_SOURCE = (
    ROOT.parent
    / "round_04_rr2_package"
    / "executed_source"
    / "gpu"
    / "qcomem_vllm_paged_multifork_resident.py"
)
BORROW = "borrow-immutable-base-functional-rebind"
MATERIALIZE = "materialize-request-base-functional-rebind"
PRODUCTION_SOURCE_SHA256 = "546efd59e2833034bc2e24d4cc0e6077f5a408275e359af43cd96f7f71cad16e"


def load_frozen_production_helper():
    """Compile exact frozen tensor-memo and GDN preparation implementations."""

    tree = ast.parse(PRODUCTION_SOURCE.read_text())
    wanted = {
        "_require",
        "_storage_key",
        "_seed_tensor_memo",
        "_prepare_request_gdn_base",
    }
    body = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    ]
    if {node.name for node in body} != wanted:
        raise AssertionError("frozen production helper AST surface drifted")
    namespace = {
        "Any": Any,
        "torch": torch,
        "MULTIFORK_GDN_BASE_POLICIES": (BORROW, MATERIALIZE),
        "GDN_BORROW_IMMUTABLE_BASE": BORROW,
    }
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(PRODUCTION_SOURCE), "exec"), namespace)
    return namespace["_seed_tensor_memo"], namespace["_prepare_request_gdn_base"]


PRODUCTION_SEED_TENSOR_MEMO, PRODUCTION_PREPARE = load_frozen_production_helper()


@dataclass
class Layer:
    conv_states: dict[int, torch.Tensor]
    recurrent_states: dict[int, torch.Tensor]


@dataclass
class Cache:
    layers: list[Layer]


def coordinates() -> tuple[tuple[int, str, int], ...]:
    return tuple(
        (layer_index, family, 0)
        for layer_index in range(30)
        for family in ("conv", "recurrent")
    )


def make_persistent() -> Cache:
    # Every coordinate deliberately has identical content and geometry but a
    # distinct object/storage.  Content and descriptor checks alone therefore
    # cannot identify source provenance.
    return Cache(
        layers=[
            Layer(
                conv_states={0: torch.tensor([7.0, 7.0], dtype=torch.bfloat16)},
                recurrent_states={0: torch.tensor([7.0, 7.0], dtype=torch.bfloat16)},
            )
            for _ in range(30)
        ]
    )


def alias_request(persistent: Cache) -> Cache:
    memo: dict[int, Any] = {}
    PRODUCTION_SEED_TENSOR_MEMO(persistent, memo, set())
    request = copy.deepcopy(persistent, memo)
    if not isinstance(request, Cache):
        raise AssertionError("frozen tensor-memo path returned an unexpected request type")
    return request


def run_exact_helper_builder(
    persistent: Cache,
    *,
    request_count: int,
    policy: str,
    wrong_source_before_clone: bool = False,
) -> SimpleNamespace:
    plan = SimpleNamespace(linear_layer_indices=tuple(range(30)))
    requests = []
    for request_index in range(request_count):
        request = alias_request(persistent)
        if wrong_source_before_clone and request_index == 0:
            # This is injected before the exact production helper runs.  The
            # helper therefore really clones coordinate B and writes its output
            # into coordinate A; there is no post-build swap.
            request.layers[0].conv_states[0] = persistent.layers[1].conv_states[0]
        PRODUCTION_PREPARE(persistent, request, plan, policy=policy)
        requests.append(request)
    return SimpleNamespace(requests=tuple(requests))


def source_snapshots(persistent: Cache) -> dict[tuple[int, str, int], tuple[str, tuple[Any, ...]]]:
    return {
        coordinate: (
            tensor_bytes_sha256(tensor_at(persistent, coordinate)),
            storage_descriptor(tensor_at(persistent, coordinate)),
        )
        for coordinate in coordinates()
    }


class PassiveCloneLineageTests(unittest.TestCase):
    def test_frozen_production_source_binding(self):
        self.assertEqual(
            hashlib.sha256(PRODUCTION_SOURCE.read_bytes()).hexdigest(),
            PRODUCTION_SOURCE_SHA256,
        )
        tree = ast.parse(PRODUCTION_SOURCE.read_text())
        helper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_prepare_request_gdn_base"
        )
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "clone"
                for node in ast.walk(helper)
            )
        )

    def test_correct_exact_production_clone_path_passes(self):
        persistent = make_persistent()
        before = source_snapshots(persistent)
        registry = PersistentSourceRegistry(persistent, coordinates())
        mode = PassiveCloneLineageMode(registry)
        with torch.inference_mode(), mode:
            group = run_exact_helper_builder(
                persistent, request_count=2, policy=MATERIALIZE
            )
        receipt = mode.verify_materialized(group.requests, coordinates())
        self.assertEqual(receipt["captured_lineage_edges"], 120)
        self.assertTrue(receipt["all_edges_direct_aten_clone"])
        self.assertEqual(before, source_snapshots(persistent))

    def test_same_geometry_same_content_wrong_source_before_clone_fails(self):
        persistent = make_persistent()
        left = persistent.layers[0].conv_states[0]
        right = persistent.layers[1].conv_states[0]
        self.assertTrue(torch.equal(left, right))
        self.assertEqual(left.shape, right.shape)
        self.assertEqual(left.dtype, right.dtype)
        self.assertNotEqual(storage_descriptor(left), storage_descriptor(right))
        registry = PersistentSourceRegistry(persistent, coordinates())
        mode = PassiveCloneLineageMode(registry)
        with torch.inference_mode(), mode:
            group = run_exact_helper_builder(
                persistent,
                request_count=1,
                policy=MATERIALIZE,
                wrong_source_before_clone=True,
            )
        # The frozen helper's own storage/content/geometry conditions all pass;
        # the independent operator lineage is what rejects this build.
        with self.assertRaisesRegex(
            LineageViolation,
            r"wrong source.*\(0, 'conv', 0\).*\(1, 'conv', 0\)",
        ):
            mode.verify_materialized(group.requests, coordinates())

    def test_borrowed_exact_alias_path_passes_without_clone_edges(self):
        persistent = make_persistent()
        before = source_snapshots(persistent)
        registry = PersistentSourceRegistry(persistent, coordinates())
        mode = PassiveCloneLineageMode(registry)
        with torch.inference_mode(), mode:
            group = run_exact_helper_builder(persistent, request_count=2, policy=BORROW)
        receipt = mode.verify_borrowed(group.requests, coordinates())
        self.assertEqual(receipt["captured_lineage_edges"], 0)
        self.assertTrue(receipt["all_exact_expected_source_aliases"])
        self.assertEqual(before, source_snapshots(persistent))

    def test_dispatch_event_preserves_actual_python_handles_and_values(self):
        persistent = make_persistent()
        source = persistent.layers[0].conv_states[0]
        source_before = tensor_bytes_sha256(source)
        registry = PersistentSourceRegistry(persistent, coordinates())
        mode = PassiveCloneLineageMode(registry)
        with torch.inference_mode(), mode:
            destination = source.clone()
        self.assertEqual(len(mode.events), 1)
        event = mode.events[0]
        self.assertIs(event.source, source)
        self.assertIs(event.destination, destination)
        self.assertEqual(event.origin_coordinate, (0, "conv", 0))
        self.assertEqual(source_before, tensor_bytes_sha256(source))
        self.assertTrue(torch.equal(source, destination))
        self.assertNotEqual(storage_descriptor(source), storage_descriptor(destination))

    def test_copy_event_preserves_actual_python_handles_and_values(self):
        persistent = make_persistent()
        source = persistent.layers[0].conv_states[0]
        source_before = tensor_bytes_sha256(source)
        registry = PersistentSourceRegistry(persistent, coordinates())
        mode = PassiveCloneLineageMode(registry)
        with torch.inference_mode(), mode:
            destination = torch.empty_like(source)
            returned = destination.copy_(source)
        self.assertEqual(len(mode.events), 1)
        event = mode.events[0]
        self.assertEqual(event.operator, "aten.copy_.default")
        self.assertIs(event.source, source)
        self.assertIs(event.destination, destination)
        self.assertIs(returned, destination)
        self.assertEqual(source_before, tensor_bytes_sha256(source))
        self.assertTrue(torch.equal(source, destination))


if __name__ == "__main__":
    unittest.main(verbosity=2)
