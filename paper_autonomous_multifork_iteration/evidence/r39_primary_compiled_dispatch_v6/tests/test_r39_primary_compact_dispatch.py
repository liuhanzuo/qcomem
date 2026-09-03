from __future__ import annotations

import contextlib
import copy
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Iterator

import r39_compiled_dispatch_receipts as base
from r39_primary_compact_dispatch import (
    ATTENTION_COLUMNS,
    GDN_COLUMNS,
    Geometry,
    PRIMARY_ARMS,
    PRIMARY_WORLD_SIZE,
    PrimaryDispatchError,
    expected_cells,
    expected_rank_counts,
    verify_payload,
    verify_primary_shard,
    _shape_record,
)
from r39_primary_rank_entrypoint import _install_primary_scope_wrappers


TINY = Geometry(
    resident_counts=(1,),
    arms=("kv=tiny-kv|gdn=tiny-gdn",),
    roles=("formal_memory", "ownership_witness"),
    generation_steps=2,
    document_tokens=3,
    full_layers=(1,),
    linear_layers=(0,),
)


def build_tiny(root: Path):
    cache, code, runtime, verbose = base._demo_fixture(root)
    artifact = verbose["attention_calls"][0]["selected_compiled_artifact"]
    config = verbose["attention_calls"][0]["selected_compile_config"]
    shape_round_0 = {
        "q": [32, 16, 256],
        "k": [34, 128, 2, 256],
        "v": [34, 128, 2, 256],
        "out": [32, 16, 256],
        "block_table": [1, 33],
        "max_seqlen_q": 32,
        "max_seqlen_k": 4127,
        "softmax_scale": 0.0625,
    }
    shape_round_1 = {
        "q": [1, 16, 256],
        "k": [34, 128, 2, 256],
        "v": [34, 128, 2, 256],
        "out": [1, 16, 256],
        "block_table": [1, 33],
        "max_seqlen_q": 1,
        "max_seqlen_k": 4128,
        "softmax_scale": 0.0625,
    }
    cells = []
    attention = []
    gdn = []
    attention_cursor = 0
    gdn_cursor = 0
    for cell_index, metadata in enumerate(expected_cells(TINY)):
        cells.append(
            {
                "cell_index": cell_index,
                "rank": 0,
                **metadata,
                "attention_call_range": [attention_cursor, attention_cursor + 2],
                "gdn_call_range": [gdn_cursor, gdn_cursor + 3],
                "expected_attention_calls": 2,
                "expected_gdn_document_prefill_calls": 1,
                "expected_gdn_request_calls": 2,
            }
        )
        attention.extend([[cell_index, 0, 0, 0, 0, 0], [cell_index, 1, 1, 0, 0, 0]])
        gdn.extend(
            [
                [cell_index, 0, 0, 3, False, 1, 0, 1, 0, 1],
                [cell_index, 1, 0, 32, True, 1, 0, 1, 0, 1],
                [cell_index, 2, 0, 1, True, 0, 1, 0, 1, 1],
            ]
        )
        attention_cursor += 2
        gdn_cursor += 3
    counts = expected_rank_counts(TINY)
    payload = {
        "schema_version": "forkaudit-r39-primary-compiled-dispatch-receipt-v3",
        "rank": 0,
        "scope": {
            "primary_protocol": "qcomem-qwen35-forkaudit-review-revision-v1",
            "primary_cells_only": True,
            "vllm_attention": base.TARGET_VLLM_ENTRYPOINT,
            "gdn": verbose["scope"]["gdn"],
        },
        "hook_installation": verbose["hook_installation"],
        "gdn_source_bindings": verbose["gdn_source_bindings"],
        "geometry": {
            "world_size": 8,
            "resident_counts": [1],
            "generation_steps": 2,
            "document_tokens": 3,
            "full_layer_indices": [1],
            "linear_layer_indices": [0],
            "arm_ids": ["kv=tiny-kv|gdn=tiny-gdn"],
            "cell_roles": ["formal_memory", "ownership_witness"],
            **counts,
        },
        "tables": {
            "compiled_artifacts": [artifact],
            "selected_compile_configurations": [config],
            "call_shapes": [_shape_record(shape_round_0), _shape_record(shape_round_1)],
            "autotune_observations": [{"mode": "no-autotuner-observed"}],
        },
        "cells": cells,
        "attention_call_columns": ATTENTION_COLUMNS,
        "attention_calls": attention,
        "gdn_call_columns": GDN_COLUMNS,
        "gdn_calls": gdn,
        "execution_binding": {
            "runner_relative_path": "run_qcomem_qwen35_forkaudit_review_revision.py",
            "runner_sha256": "9da619fc037e2c670b146d778fd9f4d5344212b7e525f3d3f26a077f79d67775",
            "runner_argv": ["--stage", "shard", "--rank", "0", "--output", "/frozen/forkaudit-shard-0.json"],
            "runner_argv_sha256": "799243df111d4b163a0a848fcfef7c7982916424bf828225a0eee27ff52fab61",
            "primary_shard_path": "/frozen/forkaudit-shard-0.json",
            "primary_shard_sha256": "a" * 64,
        },
    }
    return cache, code, runtime, payload


def fixture_bindings(payload):
    return copy.deepcopy(payload["gdn_source_bindings"])


class _StrictScopeRecorder:
    """Small fail-closed recorder used to test the real entrypoint wrappers."""

    def __init__(self) -> None:
        self.rank: int | None = None
        self.finished = False
        self.active = False
        self.cells: list[dict[str, Any]] = []
        self.begin_calls = 0
        self.finish_calls = 0

    def begin_factorial(self, rank: int) -> None:
        if self.rank is not None:
            raise PrimaryDispatchError("primary factorial entered twice")
        if type(rank) is not int or not 0 <= rank < PRIMARY_WORLD_SIZE:
            raise PrimaryDispatchError("rank invalid")
        self.rank = rank
        self.begin_calls += 1

    def finish_factorial(self) -> None:
        if self.active:
            raise PrimaryDispatchError("primary cell remained active")
        if len(self.cells) != len(expected_cells()):
            raise PrimaryDispatchError("primary factorial cell count drift")
        self.finished = True
        self.finish_calls += 1

    @contextlib.contextmanager
    def primary_cell(
        self,
        *,
        rank: int,
        resident_count: int,
        arm_id: str,
        kv_policy: str,
        gdn_base_policy: str,
        cell_role: str,
    ) -> Iterator[None]:
        if self.rank != rank:
            raise PrimaryDispatchError("cell/rank factorial binding drift")
        if self.finished:
            raise PrimaryDispatchError("cell began after factorial closure")
        if self.active:
            raise PrimaryDispatchError("nested primary cells are invalid")
        observed = {
            "resident_count": resident_count,
            "arm_id": arm_id,
            "kv_policy": kv_policy,
            "gdn_base_policy": gdn_base_policy,
            "cell_role": cell_role,
        }
        if observed != expected_cells()[len(self.cells)]:
            raise PrimaryDispatchError("primary cell order/geometry drift")
        self.active = True
        try:
            yield
        finally:
            self.active = False
        self.cells.append(observed)


def _scope_call_kwargs(rank: int, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": rank,
        "resident_count": metadata["resident_count"],
        "arm_id": metadata["arm_id"],
        "kv_policy": metadata["kv_policy"],
        "gdn_base_policy": metadata["gdn_base_policy"],
    }


def _scope_runner_fixture(mode: str = "normal") -> tuple[Any, list[tuple[str, int]], tuple[Any, Any, Any]]:
    runner = types.SimpleNamespace()
    calls: list[tuple[str, int]] = []

    def memory(
        *,
        rank: int,
        resident_count: int,
        arm_id: str,
        kv_policy: str,
        gdn_base_policy: str,
        nested: bool = False,
    ) -> str:
        calls.append(("memory", rank))
        if nested:
            runner._run_ownership_witness_cell(
                rank=rank,
                resident_count=resident_count,
                arm_id=arm_id,
                kv_policy=kv_policy,
                gdn_base_policy=gdn_base_policy,
            )
        return "memory"

    def witness(
        *,
        rank: int,
        resident_count: int,
        arm_id: str,
        kv_policy: str,
        gdn_base_policy: str,
    ) -> str:
        calls.append(("witness", rank))
        return "witness"

    def factorial(*, rank: int, reenter: bool = False) -> str:
        if reenter:
            return runner._run_formal_factorial_cells(rank=rank)
        if mode == "wrong-rank":
            metadata = expected_cells()[0]
            runner._run_clean_memory_cell(
                **_scope_call_kwargs((rank + 1) % PRIMARY_WORLD_SIZE, metadata)
            )
            return "wrong-rank-unreachable"
        if mode == "nested":
            metadata = expected_cells()[0]
            runner._run_clean_memory_cell(
                **_scope_call_kwargs(rank, metadata), nested=True
            )
            return "nested-unreachable"
        if mode == "sentinel":
            raise RuntimeError("factorial sentinel")
        rows = expected_cells()
        if mode == "short":
            rows = rows[:-1]
        for metadata in rows:
            function = (
                runner._run_clean_memory_cell
                if metadata["cell_role"] == "formal_memory"
                else runner._run_ownership_witness_cell
            )
            function(**_scope_call_kwargs(rank, metadata))
        return "factorial"

    runner._run_formal_factorial_cells = factorial
    runner._run_clean_memory_cell = memory
    runner._run_ownership_witness_cell = witness
    originals = (factorial, memory, witness)
    return runner, calls, originals


class PrimaryCompactDispatchTests(unittest.TestCase):
    def test_gdn_mutually_exclusive_routes_are_exact_and_fail_closed(self) -> None:
        multi = base._GDNContext(
            call_index=0,
            layer_idx=0,
            sequence_length=32,
            cache_has_previous_state=True,
            execution_phase="request-cell",
            chunk_kernel_events=1,
            recurrent_kernel_events=0,
            conv_rebind_events=1,
            inplace_conv_update_events=0,
            recurrent_rebind_events=1,
        )
        single = base._GDNContext(
            call_index=1,
            layer_idx=0,
            sequence_length=1,
            cache_has_previous_state=True,
            execution_phase="request-cell",
            chunk_kernel_events=0,
            recurrent_kernel_events=1,
            conv_rebind_events=0,
            inplace_conv_update_events=1,
            recurrent_rebind_events=1,
        )
        self.assertEqual(base._validate_gdn_route_counts(multi), "multi-token")
        self.assertEqual(
            base._validate_gdn_route_counts(single), "cached-single-token"
        )

        for context, field in (
            (copy.deepcopy(multi), "recurrent_kernel_events"),
            (copy.deepcopy(single), "chunk_kernel_events"),
            (copy.deepcopy(multi), "inplace_conv_update_events"),
            (copy.deepcopy(single), "conv_rebind_events"),
            (copy.deepcopy(single), "recurrent_rebind_events"),
        ):
            setattr(context, field, getattr(context, field) + 1)
            with self.assertRaisesRegex(
                base.DispatchReceiptError, "route count drift"
            ):
                base._validate_gdn_route_counts(context)

    def test_runtime_hook_declares_and_patches_all_four_route_callables(self) -> None:
        self.assertEqual(
            base.GDN_SOURCE_KEYS,
            {
                "transformers_accelerate_wrapper",
                "transformers_qwen35_moe_gdn_forward",
                "transformers_qwen35_moe_torch_chunk_gated_delta_rule",
                "transformers_qwen35_moe_torch_recurrent_gated_delta_rule",
                "transformers_qwen35_moe_torch_causal_conv1d_update",
                "qcomem_install_native_functional_linear_cache",
                "qcomem_functional_update_conv_state",
                "qcomem_functional_update_recurrent_state",
            },
        )
        hook_source = Path(base.__file__).read_text(encoding="utf-8")
        for required in (
            "qwen_module.torch_chunk_gated_delta_rule = torch_chunk_wrapper",
            "qwen_module.torch_recurrent_gated_delta_rule = torch_recurrent_wrapper",
            "qwen_module.torch_causal_conv1d_update = torch_causal_conv_update_wrapper",
            'recorder.record_gdn_event("chunk-kernel")',
            'recorder.record_gdn_event("recurrent-kernel")',
            'recorder.record_gdn_event("conv-rebind")',
            'recorder.record_gdn_event("inplace-conv-update")',
            'recorder.record_gdn_event("recurrent-rebind")',
        ):
            self.assertIn(required, hook_source)

    def test_frozen_primary_counts(self) -> None:
        self.assertEqual(len(PRIMARY_ARMS), 4)
        self.assertEqual(
            expected_rank_counts(),
            {
                "cell_count": 24,
                "attention_call_count": 26240,
                "gdn_document_prefill_call_count": 720,
                "gdn_request_call_count": 78720,
                "gdn_call_count": 79440,
            },
        )

    def test_entrypoint_scope_captures_only_factorial_for_all_eight_ranks(self) -> None:
        total_ignored_warmups = 0
        total_captured_cells = 0
        formal_rows = expected_cells()
        memory_rows = [
            row for row in formal_rows if row["cell_role"] == "formal_memory"
        ]
        for rank in range(PRIMARY_WORLD_SIZE):
            runner, calls, originals = _scope_runner_fixture()
            recorder = _StrictScopeRecorder()
            restore = _install_primary_scope_wrappers(
                runner_module=runner, recorder=recorder
            )
            try:
                # Immutable order: one discarded priming cell plus four arm
                # warmups execute before the formal factorial.
                for index in range(5):
                    runner._run_clean_memory_cell(
                        **_scope_call_kwargs(rank, memory_rows[index % 4])
                    )
                self.assertEqual(recorder.cells, [])
                total_ignored_warmups += 5

                self.assertEqual(
                    runner._run_formal_factorial_cells(rank=rank), "factorial"
                )
                self.assertTrue(recorder.finished)
                self.assertEqual(recorder.begin_calls, 1)
                self.assertEqual(recorder.finish_calls, 1)
                self.assertEqual(recorder.cells, formal_rows)
                self.assertEqual(len(recorder.cells), 24)
                total_captured_cells += len(recorder.cells)

                # Future/post-factorial controls remain on the immutable
                # runner path without expanding the primary receipt.
                runner._run_clean_memory_cell(
                    **_scope_call_kwargs(rank, memory_rows[0])
                )
                runner._run_ownership_witness_cell(
                    **_scope_call_kwargs(rank, formal_rows[1])
                )
                self.assertEqual(len(recorder.cells), 24)
                self.assertEqual(
                    sum(kind == "memory" for kind, _ in calls), 5 + 12 + 1
                )
                self.assertEqual(
                    sum(kind == "witness" for kind, _ in calls), 12 + 1
                )
            finally:
                restore()
            self.assertIs(runner._run_formal_factorial_cells, originals[0])
            self.assertIs(runner._run_clean_memory_cell, originals[1])
            self.assertIs(runner._run_ownership_witness_cell, originals[2])
        self.assertEqual(total_ignored_warmups, 40)
        self.assertEqual(total_captured_cells, 192)

    def test_entrypoint_scope_rejects_wrong_rank_and_nested_cells(self) -> None:
        for mode, message in (
            ("wrong-rank", "cell/rank factorial binding drift"),
            ("nested", "nested primary cells are invalid"),
        ):
            runner, calls, _ = _scope_runner_fixture(mode)
            recorder = _StrictScopeRecorder()
            restore = _install_primary_scope_wrappers(
                runner_module=runner, recorder=recorder
            )
            try:
                with self.assertRaisesRegex(PrimaryDispatchError, message):
                    runner._run_formal_factorial_cells(rank=0)
                self.assertFalse(recorder.active)
                self.assertEqual(recorder.finish_calls, 0)
                # The factorial token must be gone even on failure: this
                # outside call executes normally and remains uncaptured.
                runner._run_clean_memory_cell(
                    **_scope_call_kwargs(0, expected_cells()[0])
                )
                self.assertEqual(recorder.cells, [])
                self.assertEqual(calls[-1], ("memory", 0))
            finally:
                restore()

    def test_entrypoint_scope_rejects_factorial_reentry_and_second_entry(self) -> None:
        runner, calls, _ = _scope_runner_fixture()
        recorder = _StrictScopeRecorder()
        restore = _install_primary_scope_wrappers(
            runner_module=runner, recorder=recorder
        )
        try:
            with self.assertRaisesRegex(
                PrimaryDispatchError, "primary factorial entered twice"
            ):
                runner._run_formal_factorial_cells(rank=0, reenter=True)
            self.assertEqual(calls, [])
            runner._run_clean_memory_cell(
                **_scope_call_kwargs(0, expected_cells()[0])
            )
            self.assertEqual(recorder.cells, [])
        finally:
            restore()

        runner, _, _ = _scope_runner_fixture()
        recorder = _StrictScopeRecorder()
        restore = _install_primary_scope_wrappers(
            runner_module=runner, recorder=recorder
        )
        try:
            runner._run_formal_factorial_cells(rank=0)
            with self.assertRaisesRegex(
                PrimaryDispatchError, "primary factorial entered twice"
            ):
                runner._run_formal_factorial_cells(rank=0)
            self.assertEqual(len(recorder.cells), 24)
        finally:
            restore()

    def test_entrypoint_scope_preserves_original_error_and_short_count_failure(self) -> None:
        for mode, error_type, message in (
            ("sentinel", RuntimeError, "factorial sentinel"),
            ("short", PrimaryDispatchError, "primary factorial cell count drift"),
        ):
            runner, calls, _ = _scope_runner_fixture(mode)
            recorder = _StrictScopeRecorder()
            restore = _install_primary_scope_wrappers(
                runner_module=runner, recorder=recorder
            )
            try:
                with self.assertRaisesRegex(error_type, message):
                    runner._run_formal_factorial_cells(rank=0)
                self.assertFalse(recorder.active)
                self.assertEqual(recorder.finish_calls, 0)
                before = len(recorder.cells)
                runner._run_ownership_witness_cell(
                    **_scope_call_kwargs(0, expected_cells()[1])
                )
                self.assertEqual(len(recorder.cells), before)
                self.assertEqual(calls[-1], ("witness", 0))
            finally:
                restore()

    def test_tiny_compact_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, code, runtime, payload = build_tiny(Path(temporary))
            result = verify_payload(
                payload,
                cache_root=cache,
                code_root=code,
                runtime_root=runtime,
                geometry=TINY,
                expected_rank=0,
                expected_gdn_bindings=fixture_bindings(payload),
            )
            self.assertEqual(result["replay_verdict"], "pass")
            self.assertEqual(result["attention_call_count"], 4)
            self.assertEqual(result["gdn_call_count"], 6)

    def test_missing_call_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, code, runtime, payload = build_tiny(Path(temporary))
            candidate = copy.deepcopy(payload)
            candidate["attention_calls"].pop()
            with self.assertRaises(PrimaryDispatchError):
                verify_payload(
                    candidate,
                    cache_root=cache,
                    code_root=code,
                    runtime_root=runtime,
                    geometry=TINY,
                    expected_rank=0,
                    expected_gdn_bindings=fixture_bindings(payload),
                )

    def test_artifact_reference_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, code, runtime, payload = build_tiny(Path(temporary))
            candidate = copy.deepcopy(payload)
            candidate["tables"]["compiled_artifacts"][0]["artifact_id"] = "0" * 64
            with self.assertRaises(base.DispatchReceiptError):
                verify_payload(
                    candidate,
                    cache_root=cache,
                    code_root=code,
                    runtime_root=runtime,
                    geometry=TINY,
                    expected_rank=0,
                    expected_gdn_bindings=fixture_bindings(payload),
                )

    def test_cross_key_callable_substitution_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, code, runtime, payload = build_tiny(Path(temporary))
            frozen = fixture_bindings(payload)
            candidate = copy.deepcopy(payload)
            candidate["gdn_source_bindings"]["transformers_accelerate_wrapper"] = copy.deepcopy(
                candidate["gdn_source_bindings"][
                    "transformers_qwen35_moe_torch_chunk_gated_delta_rule"
                ]
            )
            with self.assertRaises(PrimaryDispatchError):
                verify_payload(
                    candidate,
                    cache_root=cache,
                    code_root=code,
                    runtime_root=runtime,
                    geometry=TINY,
                    expected_rank=0,
                    expected_gdn_bindings=frozen,
                )

    def test_scope_column_shape_and_rank_substitutions_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, code, runtime, payload = build_tiny(Path(temporary))
            frozen = fixture_bindings(payload)
            candidates = []
            candidate = copy.deepcopy(payload)
            candidate["scope"]["gdn"] = "compiled-superkernel-attested"
            candidates.append(candidate)
            candidate = copy.deepcopy(payload)
            candidate["attention_call_columns"] = ["lies"]
            candidates.append(candidate)
            candidate = copy.deepcopy(payload)
            candidate["tables"]["call_shapes"][0]["shape"]["q"] = [999, 999]
            candidates.append(candidate)
            candidate = copy.deepcopy(payload)
            candidate["rank"] = 1
            candidates.append(candidate)
            for candidate in candidates:
                with self.assertRaises(PrimaryDispatchError):
                    verify_payload(
                        candidate,
                        cache_root=cache,
                        code_root=code,
                        runtime_root=runtime,
                        geometry=TINY,
                        expected_rank=0,
                        expected_gdn_bindings=frozen,
                    )

    def test_primary_shard_ledger_binding(self) -> None:
        calls = [
            {
                "request_index": 0,
                "layer_idx": 1,
                "query_tokens": 32,
                "physical_block_pool_shape": [34, 128, 2, 256],
                "active_block_table_shape": [1, 33],
                "kv_tokens": 4127,
                "softmax_scale": 0.0625,
            },
            {
                "request_index": 0,
                "layer_idx": 1,
                "query_tokens": 1,
                "physical_block_pool_shape": [34, 128, 2, 256],
                "active_block_table_shape": [1, 33],
                "kv_tokens": 4128,
                "softmax_scale": 0.0625,
            },
        ]
        shard = {
            "schema_version": "qcomem-forkaudit-review-shard-v1",
            "protocol": "qcomem-qwen35-forkaudit-review-revision-v1",
            "status": "completed_formal_gpu_shard",
            "rank": 0,
            "world_size": 8,
            "protocol_config": {
                "resident_counts": [1],
                "generation_steps": 2,
                "document_tokens": 3,
                "factorial_arm_ids": ["kv=tiny-kv|gdn=tiny-gdn"],
            },
            "factorial": [
                {
                    "resident_count": 1,
                    "cells": [
                        {
                            "arm_id": "kv=tiny-kv|gdn=tiny-gdn",
                            "memory_kernel_ledgers": [
                                {
                                    "request_index": 0,
                                    "total_calls": 2,
                                    "verified": True,
                                    "dense_fallback_calls": 0,
                                    "calls": copy.deepcopy(calls),
                                }
                            ],
                            "witness_kernel_ledgers": [
                                {
                                    "request_index": 0,
                                    "total_calls": 2,
                                    "verified": True,
                                    "dense_fallback_calls": 0,
                                    "calls": copy.deepcopy(calls),
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        result = verify_primary_shard(shard, expected_rank=0, geometry=TINY)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["attention_ledger_call_count"], 4)
        with tempfile.TemporaryDirectory() as temporary:
            cache, code, runtime, payload = build_tiny(Path(temporary))
            verify_payload(
                payload,
                cache_root=cache,
                code_root=code,
                runtime_root=runtime,
                geometry=TINY,
                expected_rank=0,
                expected_gdn_bindings=fixture_bindings(payload),
            )
            candidate = copy.deepcopy(payload)
            record = candidate["tables"]["call_shapes"][0]
            record["shape"]["max_seqlen_k"] = 4128
            record["shape_sha256"] = base._sha256_bytes(
                base._canonical_bytes(record["shape"])
            )
            # The altered row remains structurally valid and self-consistent,
            # but it no longer describes the immutable runner's actual call.
            verify_payload(
                candidate,
                cache_root=cache,
                code_root=code,
                runtime_root=runtime,
                geometry=TINY,
                expected_rank=0,
                expected_gdn_bindings=fixture_bindings(payload),
            )
            with self.assertRaises(PrimaryDispatchError):
                verify_primary_shard(
                    shard,
                    expected_rank=0,
                    geometry=TINY,
                    receipt=candidate,
                )


if __name__ == "__main__":
    unittest.main()
