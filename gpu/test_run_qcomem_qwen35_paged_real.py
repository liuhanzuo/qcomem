from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

from run_qcomem_qwen35_paged_real import (
    _fork_cache_metadata_sharing_tensors,
    _run_capability_gate,
)

import torch


class RealRunnerOrchestrationTest(unittest.TestCase):
    def test_metadata_fork_shares_tensor_storage_without_sharing_metadata(self) -> None:
        source = SimpleNamespace(
            layers=[SimpleNamespace(states={0: torch.randn(2, 3)})]
        )
        fork = _fork_cache_metadata_sharing_tensors(source)
        self.assertIsNot(fork, source)
        self.assertIsNot(fork.layers, source.layers)
        self.assertIsNot(fork.layers[0].states, source.layers[0].states)
        self.assertIs(fork.layers[0].states[0], source.layers[0].states[0])

    def test_gate_builds_independent_stock_then_native_and_pages_stock(self) -> None:
        stock = object()
        native = object()
        install = object()
        caller = object()
        pair = SimpleNamespace(conversion=object())
        ledger = object()
        backend = object()
        backbone = SimpleNamespace(config=object())
        plan = SimpleNamespace(
            full=object(),
            metadata=lambda: {
                "linear_layer_count": 30,
                "full_attention_layer_count": 10,
            },
        )

        with (
            patch(
                "run_qcomem_qwen35_paged_real._build_dense_document_cache",
                side_effect=((stock, None), (native, install)),
            ) as build,
            patch(
                "run_qcomem_qwen35_paged_real._native_layer_gate",
                return_value={"passed": True},
            ),
            patch(
                "run_qcomem_qwen35_paged_real._same_query_caller",
                return_value=caller,
            ),
            patch(
                "run_qcomem_qwen35_paged_real._native_same_caller_gate",
                return_value={"passed": True},
            ) as native_gate,
            patch(
                "run_qcomem_qwen35_paged_real.clone_dense_and_prepare_paged_cache_pair",
                return_value=pair,
            ) as prepare,
            patch(
                "run_qcomem_qwen35_paged_real.PagedAttentionHitLedger",
                return_value=ledger,
            ),
            patch(
                "run_qcomem_qwen35_paged_real.register_qwen35_paged_backend",
                return_value=backend,
            ),
            patch(
                "run_qcomem_qwen35_paged_real.run_same_caller_eager_paged_gate",
                return_value={"passed": True},
            ),
            patch(
                "run_qcomem_qwen35_paged_real.require_passed_reference_gate_before_benchmark",
                return_value={"benchmark_gate_passed": True},
            ),
        ):
            result = _run_capability_gate(
                backbone=backbone,
                model=object(),
                document=object(),
                query=object(),
                stack_plan=plan,
                page_size=128,
                group_size=64,
                append_page_size=16,
                rtol=0.02,
                atol=0.05,
            )

        self.assertTrue(result["passed"])
        self.assertEqual(
            build.call_args_list,
            [
                call(backbone, unittest.mock.ANY, functional_linear=False),
                call(backbone, unittest.mock.ANY, functional_linear=True),
            ],
        )
        native_kwargs = native_gate.call_args.kwargs
        self.assertIs(native_kwargs["standard_cache"], stock)
        self.assertIs(native_kwargs["native_cache"], native)
        self.assertIs(prepare.call_args.args[0], stock)
        self.assertEqual(prepare.call_args.kwargs["bits"], 16)


if __name__ == "__main__":
    unittest.main()
