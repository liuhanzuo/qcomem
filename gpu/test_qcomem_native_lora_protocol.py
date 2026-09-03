from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aggregate_native_lora_heldout import aggregate_heldout
from aggregate_native_lora_semantic_gate import aggregate_native_semantic
from qcomem_native_lora_protocol import (
    NativeLoRAProtocolError,
    domain_view_record,
    evaluate_step1_gate,
    sampler_scheduled_records,
    stable_json,
    token_ids_sha256,
)
from train_native_lora_formal import distributed_atomic_torch_save


def _distributed_step_zero_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    output: str,
    results: str,
) -> None:
    import torch
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        payload = {
            "format": "qcomem_suffix_lora_v1",
            "step": 0,
            "lora": {"layer.lora_a": torch.tensor([1.0])},
            "metadata": {"last_step": 0},
        }
        digest = distributed_atomic_torch_save(Path(output), payload)
        Path(results, f"rank-{rank}.json").write_text(
            json.dumps({"rank": rank, "sha256": digest})
        )
    finally:
        dist.destroy_process_group()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def parent_row(document: list[int], query: list[int], *, suffix: str = "0") -> dict:
    target = [91, 92]
    prompt = document + query
    return {
        "schema_version": "qcomem-deployment-aware-example-v1",
        "source_split": "train",
        "stratum": "domain",
        "dataset": "qasper",
        "example_id": digest("example" + suffix),
        "source_id_sha256": digest("source" + suffix),
        "document_id_sha256": digest("document-id" + suffix),
        "prompt_sha256": digest("prompt-text" + suffix),
        "input_ids": prompt + target,
        "labels": [-100] * len(prompt) + target,
        "deployment_boundary": {
            "applicable": True,
            "document_input_ids": document,
            "query_input_ids": query,
            "document_tokens": len(document),
            "query_tokens": len(query),
            "prompt_tokens": len(prompt),
            "document_input_ids_sha256": token_ids_sha256(document),
            "query_input_ids_sha256": token_ids_sha256(query),
            "prompt_input_ids_sha256": token_ids_sha256(prompt),
            "answer_or_eos_tokens_in_query": False,
        },
    }


class NativeLoRAProtocolTest(unittest.TestCase):
    def test_two_rank_cpu_step_zero_checkpoint_agrees_on_sha_and_payload(self) -> None:
        import torch
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            rendezvous = root / "gloo-rendezvous"
            output = root / "checkpoint-000000.pt"
            results = root / "results"
            results.mkdir()
            mp.spawn(
                _distributed_step_zero_worker,
                args=(2, str(rendezvous), str(output), str(results)),
                nprocs=2,
                join=True,
            )
            rows = [
                json.loads((results / f"rank-{rank}.json").read_text())
                for rank in range(2)
            ]
            self.assertEqual(len({row["sha256"] for row in rows}), 1)
            self.assertEqual(
                rows[0]["sha256"], hashlib.sha256(output.read_bytes()).hexdigest()
            )
            payload = torch.load(output, map_location="cpu", weights_only=True)
            self.assertEqual(payload["step"], 0)
            self.assertNotIn("optimizer", payload)

    def test_distributed_step_zero_checkpoint_is_written_only_by_rank_zero(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "checkpoint-000000.pt"
            payload = {
                "format": "qcomem_suffix_lora_v1",
                "step": 0,
                "lora": {"layer.lora_a": torch.ones(1)},
                "metadata": {"last_step": 0},
            }

            def gather_success(records, local):
                records[:] = [local, {**local, "rank": 1}]

            with (
                mock.patch("train_native_lora_formal._rank", return_value=0),
                mock.patch("train_native_lora_formal._world_size", return_value=2),
                mock.patch("train_native_lora_formal.dist.broadcast_object_list"),
                mock.patch("train_native_lora_formal.dist.barrier") as barrier,
                mock.patch(
                    "train_native_lora_formal.dist.all_gather_object",
                    side_effect=gather_success,
                ),
            ):
                digest = distributed_atomic_torch_save(path, payload)
            self.assertEqual(torch.load(path, weights_only=True)["step"], 0)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            barrier.assert_called_once_with()

            def broadcast_success(decision, *, src):
                self.assertEqual(src, 0)
                decision[0] = {"ok": True, "error": None}

            with (
                mock.patch("train_native_lora_formal._rank", return_value=1),
                mock.patch("train_native_lora_formal._world_size", return_value=2),
                mock.patch(
                    "train_native_lora_formal.dist.broadcast_object_list",
                    side_effect=broadcast_success,
                ),
                mock.patch("train_native_lora_formal.dist.barrier"),
                mock.patch(
                    "train_native_lora_formal.dist.all_gather_object",
                    side_effect=gather_success,
                ),
                mock.patch("train_native_lora_formal.torch.save") as save,
            ):
                distributed_atomic_torch_save(path, payload)
            save.assert_not_called()

    def test_rank_zero_step_zero_write_failure_is_collective_before_barrier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "checkpoint-000000.pt"
            with (
                mock.patch("train_native_lora_formal._rank", return_value=0),
                mock.patch("train_native_lora_formal._world_size", return_value=2),
                mock.patch(
                    "train_native_lora_formal.torch.save",
                    side_effect=OSError("simulated write failure"),
                ),
                mock.patch(
                    "train_native_lora_formal.dist.broadcast_object_list"
                ) as broadcast,
                mock.patch("train_native_lora_formal.dist.barrier") as barrier,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "rank-zero checkpoint write failed"
                ):
                    distributed_atomic_torch_save(path, {"step": 0})
            broadcast.assert_called_once()
            barrier.assert_not_called()

    def test_heldout_selection_is_example_equal_and_can_select_step64(self) -> None:
        shards = []
        for rank in range(2):
            checkpoints = []
            for step, loss in ((0, 0.4), (64, 0.2), (128, 0.3)):
                checkpoints.append(
                    {
                        "training_step": step,
                        "checkpoint": f"/checkpoint-{step}",
                        "checkpoint_sha256": str(step).zfill(64),
                        "rows": [
                            {
                                "example_id": f"example-{rank}",
                                "loss": loss + rank * 0.1,
                                "forward_kl": loss,
                                "reverse_kl": loss,
                                "cache_audit": {"hard_gate_passed": True},
                            }
                        ],
                    }
                )
            shards.append(
                {
                    "status": "completed_shard",
                    "rank": rank,
                    "data_sha256": "d" * 64,
                    "test_v2_used": False,
                    "checkpoints": checkpoints,
                }
            )
        result = aggregate_heldout(
            shards,
            expected_world_size=2,
            expected_examples=2,
            expected_data_sha256="d" * 64,
        )
        self.assertEqual(result["selected"]["training_step"], 64)

    def test_native_semantic_aggregate_uses_every_position(self) -> None:
        shards = []
        for rank in range(2):
            shards.append(
                {
                    "status": "completed_shard",
                    "rank": rank,
                    "data_sha256": "d" * 64,
                    "checkpoint_sha256": "c" * 64,
                    "test_v2_used": False,
                    "global_samples_requested": 2,
                    "rows": [
                        {
                            "example_id": f"example-{rank}",
                            "cache_audit": {"hard_gate_passed": True},
                            "positions": [
                                {
                                    "position": 0,
                                    "top1_match": True,
                                    "kl_functional_to_mutable": 0.0,
                                    "max_abs_logit_error": 0.0,
                                },
                                {
                                    "position": 1,
                                    "top1_match": True,
                                    "kl_functional_to_mutable": 0.0,
                                    "max_abs_logit_error": 0.0,
                                },
                            ],
                        }
                    ],
                }
            )
        result = aggregate_native_semantic(
            shards,
            expected_world_size=2,
            expected_data_sha256="d" * 64,
            expected_checkpoint_sha256="c" * 64,
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["query_positions"], 4)

    def test_domain_view_preserves_query_and_head_tail_bounds_document(self) -> None:
        result = domain_view_record(
            parent_row(list(range(10)), [71, 72, 73]),
            max_document_tokens=6,
            max_query_tokens=4,
        )
        self.assertEqual(result["document_ids"], [0, 1, 2, 7, 8, 9])
        self.assertEqual(result["query_ids"], [71, 72, 73])
        self.assertTrue(result["view"]["document_truncated"])
        self.assertEqual(result["view"]["query_truncation"], "forbidden")
        self.assertNotIn("dataset", result)
        self.assertEqual(result["source_dataset"], "qasper")

    def test_query_is_never_silently_truncated(self) -> None:
        with self.assertRaisesRegex(NativeLoRAProtocolError, "never truncated"):
            domain_view_record(
                parent_row([1, 2], [3, 4, 5]),
                max_document_tokens=2,
                max_query_tokens=2,
            )

    def test_sampler_inverse_schedule_puts_longest_rows_in_step_one(self) -> None:
        rows = [
            domain_view_record(
                parent_row(list(range(length)), [88], suffix=str(length)),
                max_document_tokens=20,
                max_query_tokens=2,
            )
            for length in range(2, 10)
        ]
        encoded, audit = sampler_scheduled_records(rows, seed=17, world_size=2)
        import torch

        permutation = torch.randperm(
            len(encoded), generator=torch.Generator().manual_seed(17)
        ).tolist()
        first = [encoded[index] for index in permutation[:2]]
        self.assertEqual(
            [len(row["document_ids"]) for row in first],
            [9, 8],
        )
        self.assertEqual(audit["first_step_sequence_tokens"], [10, 9])

    def test_step_one_gate_bundles_grad_update_cache_and_memory(self) -> None:
        gradient_rank = {
            "module_count": 2,
            "finite_module_count": 2,
            "nonzero_module_count": 2,
        }
        cache_rank = {
            "execution": "native-functional-cache",
            "hard_gate_passed": True,
            "original_cache_versions_unchanged": True,
            "all_cache_paths_rebound": True,
            "query_positions_expected": 7,
            "query_positions_observed": 7,
        }
        metadata = {
            "warm_start": {
                "source_mode": "interface",
                "source_step": 200,
                "checkpoint_sha256": "a" * 64,
            },
            "last_gradient_coverage": {
                "step": 1,
                "hard_gate_passed": True,
                "by_rank": [gradient_rank, gradient_rank],
            },
            "last_detached_capability": {
                "step": 1,
                "hard_gate_passed": True,
                "by_rank": [cache_rank, cache_rank],
            },
            "test_v2_used": False,
        }
        ranks = [
            {
                "rank": rank,
                "modules": 2,
                "parameter_tensors": 4,
                "finite_gradient_tensors": 4,
                "nonzero_gradient_tensors": 4,
                "finite_update_tensors": 4,
                "nonzero_update_tensors": 4,
                "headroom_bytes": 1024,
            }
            for rank in range(2)
        ]
        result = evaluate_step1_gate(
            ranks,
            metadata,
            expected_world_size=2,
            expected_modules=2,
            expected_parameter_tensors=4,
            minimum_headroom_bytes=512,
            expected_init_checkpoint_sha256="a" * 64,
        )
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["single_token_autograd_claimed"])
        ranks[0]["nonzero_update_tensors"] = 3
        failed = evaluate_step1_gate(
            ranks,
            metadata,
            expected_world_size=2,
            expected_modules=2,
            expected_parameter_tensors=4,
            minimum_headroom_bytes=512,
            expected_init_checkpoint_sha256="a" * 64,
        )
        self.assertEqual(failed["status"], "failed")

    def test_token_hash_uses_canonical_list_json(self) -> None:
        self.assertEqual(
            token_ids_sha256([1, 2, 3]),
            hashlib.sha256(stable_json([1, 2, 3]).encode("utf-8")).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
