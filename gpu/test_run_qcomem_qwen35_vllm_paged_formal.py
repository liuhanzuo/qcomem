from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from run_qcomem_qwen35_vllm_paged_formal import (
    KERNEL_MODE,
    POST_ROPE_POSITION_IDS_CONTRACT,
    _median_measurements,
    _model_manifest_sha,
    _record_fresh_trial,
    _run_fresh_abba,
    _validate_static,
    aggregate_pg19_gate_shards,
    sha256_file,
    summarize_validation_shards,
)


FULL_LAYERS = tuple(range(3, 40, 4))
PRODUCTION_MASK = "prevalidated-no-padding-tail-causal"
STRICT_MASK = "strict-canonical-audit"


def fused_intercept(calls_per_layer: int, mask_contract: str) -> dict:
    calls = []
    for layer in FULL_LAYERS:
        for _ in range(calls_per_layer):
            calls.append(
                {
                    "layer_idx": layer,
                    "kernel_mode": KERNEL_MODE,
                    "fused_gpu_kernel_calls": 1,
                    "full_kv_concatenations": 0,
                    "full_document_staging_copy_nbytes": 0,
                    "quantization": "Q16",
                    "materialized_attention_mask_nbytes": (
                        0 if mask_contract == PRODUCTION_MASK else 4096
                    ),
                    "mask_validation_host_syncs": (
                        0 if mask_contract == PRODUCTION_MASK else 1
                    ),
                    "position_ids_contract": POST_ROPE_POSITION_IDS_CONTRACT,
                    "position_ids_validated": True,
                    "position_ids_semantically_consumed_upstream": True,
                    "position_ids_strict_tail_values_checked": (
                        mask_contract == STRICT_MASK
                    ),
                    "position_ids_validation_host_syncs": (
                        0 if mask_contract == PRODUCTION_MASK else 1
                    ),
                }
            )
    return {
        "verified": True,
        "kernel_mode": KERNEL_MODE,
        "expected_layer_indices": FULL_LAYERS,
        "counts": {layer: calls_per_layer for layer in FULL_LAYERS},
        "total_calls": len(FULL_LAYERS) * calls_per_layer,
        "dense_fallback_calls": 0,
        "full_kv_concatenations": 0,
        "mask_contract": mask_contract,
        "materialized_attention_mask_nbytes": (
            0 if mask_contract == PRODUCTION_MASK else 4096 * len(calls)
        ),
        "mask_validation_host_syncs": (
            0 if mask_contract == PRODUCTION_MASK else len(calls)
        ),
        "position_ids_contract": POST_ROPE_POSITION_IDS_CONTRACT,
        "position_ids_validation_host_syncs": (
            0 if mask_contract == PRODUCTION_MASK else len(calls)
        ),
        "calls": calls,
    }


def isolated_window(rank: int) -> dict:
    return {
        "window_index": rank,
        "source_object": f"train/book-{rank}.txt",
        "passed": True,
        "layer_indices": FULL_LAYERS,
        "layer_count": 10,
        "dense_fallback_calls": 0,
        "rows": [
            {
                "layer_idx": layer,
                "close": True,
                "finite": True,
                "audit": {
                    "fused_gpu_kernel_calls": 1,
                    "full_kv_concatenations": 0,
                    "position_ids_contract": POST_ROPE_POSITION_IDS_CONTRACT,
                    "position_ids_validated": True,
                    "position_ids_semantically_consumed_upstream": True,
                    "position_ids_strict_tail_values_checked": True,
                },
            }
            for layer in FULL_LAYERS
        ],
    }


def gate_shard(rank: int) -> dict:
    semantic = {
        "window_index": rank,
        "source_object": f"train/book-{rank}.txt",
        "full_vocab_forward_kl": 1e-5 * (rank + 1),
        "top1_exact": True,
        "intercept": fused_intercept(1, STRICT_MASK),
        "fork": {"full_document_staging_copy_nbytes": 0},
    }
    return {
        "static": {
            "status": "static_dry_run_passed",
            "model_manifest_sha256": "model-manifest",
        },
        "status": "completed_pg19_gate_shard",
        "passed": True,
        "rank": rank,
        "world_size": 8,
        "kernel_mode": KERNEL_MODE,
        "data_audit": {"split": "train", "books": 8},
        "windows_sha256": "windows-sha",
        "isolated_gate": {"passed": True, "windows": [isolated_window(rank)]},
        "semantic_gate": {
            "passed": True,
            "top1_agreement": 1.0,
            "example_equal_mean_full_vocab_forward_kl": semantic[
                "full_vocab_forward_kl"
            ],
            "mean_kl_threshold": 1e-3,
            "rows": [semantic],
        },
        "validation_consumed": False,
        "test_v2_consumed": False,
    }


def stock_trial(tokens: list[int], offset: int = 0) -> dict:
    return {
        "config": "stock-eager",
        "configuration_scope": (
            "transformers-eager-full-attention-with-functional-gdn-fork-control"
        ),
        "kernel_mode": "transformers-eager",
        "generated_token_ids": tokens,
        "ttft_seconds": 2.0 + offset * 0.01,
        "median_tpot_seconds": 1.0 + offset * 0.01,
        "cuda_peak_request_delta_bytes": 500,
        "persistent_total_resident_nbytes": 2_000,
        "persistent_resident_nbytes": 2_000,
        "query_fork_full_document_staging_copy_nbytes": 0,
        "full_document_staging_copy_nbytes": 0,
        "fork": {"full_document_staging_copy_nbytes": 0},
        "q16_document_build": {
            "performed": False,
            "document_build_pack_seconds": 0.0,
            "document_build_copy_nbytes": 0,
            "document_build_cuda_peak_delta_bytes": 0,
        },
        "intercept": None,
    }


def fused_trial(tokens: list[int], offset: int = 0) -> dict:
    return {
        "config": "vllm-paged-q16",
        "configuration_scope": "vllm-q16-full-attention-with-functional-gdn-fork",
        "kernel_mode": KERNEL_MODE,
        "generated_token_ids": tokens,
        "ttft_seconds": 1.0 + offset * 0.01,
        "median_tpot_seconds": 0.5 + offset * 0.01,
        "cuda_peak_request_delta_bytes": 250,
        "persistent_total_resident_nbytes": 1_500,
        "persistent_resident_nbytes": 1_500,
        "query_fork_full_document_staging_copy_nbytes": 0,
        "full_document_staging_copy_nbytes": 0,
        "fork": {"full_document_staging_copy_nbytes": 0},
        "q16_document_build": {
            "performed": True,
            "document_build_pack_seconds": 0.25 + offset * 0.001,
            "document_build_copy_nbytes": 1_000,
            "document_build_cuda_peak_delta_bytes": 1_100,
        },
        "dense_document_kv_nbytes": 1_000,
        "persistent_q16_document_payload_nbytes": 1_000,
        "persistent_q16_allocated_block_pool_nbytes": 1_200,
        "intercept": fused_intercept(len(tokens), PRODUCTION_MASK),
    }


class FormalRunnerStaticTest(unittest.TestCase):
    def test_q8_q4_fail_before_hash_or_model_environment_access(self) -> None:
        for bits in (8, 4):
            args = SimpleNamespace(bits=bits)
            with (
                mock.patch(
                    "run_qcomem_qwen35_vllm_paged_formal.sha256_file"
                ) as file_hash,
                mock.patch(
                    "run_qcomem_qwen35_vllm_paged_formal.audit_frozen_kernel_environment"
                ) as environment,
            ):
                with self.assertRaisesRegex(RuntimeError, "Q16 only"):
                    _validate_static(args)
            file_hash.assert_not_called()
            environment.assert_not_called()

    def test_static_dry_run_freezes_model_geometry_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            config = {
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "num_hidden_layers": 40,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 2,
                    "head_dim": 256,
                    "layer_types": [
                        "full_attention" if index in FULL_LAYERS else "linear_attention"
                        for index in range(40)
                    ],
                }
            }
            (model / "config.json").write_text(json.dumps(config))
            (model / "generation_config.json").write_text("{}\n")
            (model / "model.safetensors.index.json").write_text("{}\n")
            inputs = []
            for name, payload in (
                ("pg19.jsonl", "train\n"),
                ("pg19.manifest.json", "{}\n"),
                ("validation.jsonl", "validation\n"),
            ):
                path = root / name
                path.write_text(payload)
                inputs.append(path)
            manifest_sha, _ = _model_manifest_sha(model)
            args = SimpleNamespace(
                bits=16,
                world_size=8,
                rank=0,
                source_index_start=6,
                source_index_end=9,
                page_size=128,
                max_new_tokens=8,
                pg19_data=inputs[0],
                pg19_manifest=inputs[1],
                validation_data=inputs[2],
                expected_pg19_sha256=sha256_file(inputs[0]),
                expected_pg19_manifest_sha256=sha256_file(inputs[1]),
                expected_validation_sha256=sha256_file(inputs[2]),
                model=model,
                expected_model_manifest_sha256=manifest_sha,
            )
            environment = {
                "matches_frozen_environment": True,
                "mismatches": {},
            }
            with mock.patch(
                "run_qcomem_qwen35_vllm_paged_formal.audit_frozen_kernel_environment",
                return_value=environment,
            ):
                result = _validate_static(args)
            self.assertEqual(result["status"], "static_dry_run_passed")
            self.assertEqual(result["geometry"]["full_attention_layer_indices"], list(FULL_LAYERS))
            self.assertEqual(result["model_manifest_sha256"], manifest_sha)
            self.assertFalse(result["gpu_initialized"])

    def test_token_divergence_stops_before_recording_second_config(self) -> None:
        trials = {"stock-eager": [], "vllm-paged-q16": []}
        observed = {}
        _record_fresh_trial(trials, observed, "stock-eager", stock_trial([1, 2]))
        with self.assertRaisesRegex(RuntimeError, "stop before further paired timing"):
            _record_fresh_trial(
                trials, observed, "vllm-paged-q16", fused_trial([1, 3])
            )
        self.assertEqual(len(trials["stock-eager"]), 1)
        self.assertEqual(trials["vllm-paged-q16"], [])

    def test_abba_orchestration_is_two_warmups_plus_eight_measurements(self) -> None:
        def measured(*args):
            return {"generated_token_ids": [7, 8], "config": args[-1]}

        with (
            mock.patch(
                "run_qcomem_qwen35_vllm_paged_formal._measure_config",
                side_effect=measured,
            ) as measure,
            mock.patch(
                "run_qcomem_qwen35_vllm_paged_formal.torch.cuda.empty_cache"
            ) as empty_cache,
        ):
            warmup, order, trials = _run_fresh_abba(
                SimpleNamespace(rank=0),
                object(),
                object(),
                object(),
                object(),
                object(),
                object(),
            )
        self.assertEqual(warmup, ("stock-eager", "vllm-paged-q16"))
        self.assertEqual(len(order), 8)
        self.assertEqual(measure.call_count, 10)
        self.assertEqual(empty_cache.call_count, 10)
        self.assertEqual(len(trials["stock-eager"]), 4)
        self.assertEqual(len(trials["vllm-paged-q16"]), 4)


class FormalAggregationTest(unittest.TestCase):
    def _authorization(self, root: Path) -> tuple[dict, list[Path]]:
        paths = []
        for rank in range(8):
            path = root / f"pg19-gate-shard-{rank}.json"
            path.write_text(json.dumps(gate_shard(rank)))
            paths.append(path)
        return (
            aggregate_pg19_gate_shards(
                paths,
                expected_windows_sha256="windows-sha",
                mean_kl_threshold=1e-3,
            ),
            paths,
        )

    def test_pg19_eight_rank_gate_aggregates_one_unique_train_book_each(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            authorization, _ = self._authorization(Path(directory))
        self.assertEqual(authorization["status"], "pg19_gate_authorized")
        self.assertEqual(authorization["parallel_gate_world_size"], 8)
        self.assertEqual(len(authorization["source_objects"]), 8)
        self.assertEqual(authorization["semantic_gate"]["top1_agreement"], 1.0)
        self.assertFalse(authorization["validation_consumed"])

    def test_hard_summary_checks_abba_kernel_memory_and_data_ledgers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authorization, _ = self._authorization(root)
            authorization_path = root / "pg19-kernel-authorization.json"
            authorization_path.write_text(json.dumps(authorization, sort_keys=True))
            authorization_sha = sha256_file(authorization_path)
            ledger = root / "code.sha256"
            ledger.write_text("deadbeef  frozen.py\n")
            ledger_sha = sha256_file(ledger)
            model_artifacts = root / "model-artifacts.sha256"
            model_artifacts.write_text("artifact  config.json\n")
            model_artifact_sha = sha256_file(model_artifacts)
            model_weights = root / "model-weights.sha256"
            model_weights.write_text("weight  shard.safetensors\n")
            model_weight_sha = sha256_file(model_weights)
            validation = root / "validation"
            validation.mkdir()
            pairs = [
                (dataset, source)
                for dataset in ("qasper", "2wikimqa")
                for source in range(6, 10)
            ]
            for rank, (dataset, source) in enumerate(pairs):
                tokens = [101, 102]
                stock_trials = [stock_trial(tokens, trial) for trial in range(4)]
                fused_trials = [fused_trial(tokens, trial) for trial in range(4)]
                stock = _median_measurements("stock-eager", stock_trials)
                fused = _median_measurements("vllm-paged-q16", fused_trials)
                order = (
                    ("stock-eager", "vllm-paged-q16", "vllm-paged-q16", "stock-eager")
                    if rank % 2 == 0
                    else ("vllm-paged-q16", "stock-eager", "stock-eager", "vllm-paged-q16")
                ) * 2
                row = {
                    "status": "completed_shard",
                    "rank": rank,
                    "world_size": 8,
                    "kernel_mode": KERNEL_MODE,
                    "static": {"model_manifest_sha256": "model-manifest"},
                    "authorization_sha256": authorization_sha,
                    "workload_metadata": {
                        "test_v2_consumed": False,
                        "source_revisions": ["revision"],
                        "datasets": ["2wikimqa", "qasper"],
                    },
                    "workload": {
                        "workload_id": f"{dataset}-{source}",
                        "dataset": dataset,
                        "source_index": source,
                    },
                    "measurement_order": order,
                    "measurement_protocol": (
                        "fresh-state-ABBAx2-four-trials-per-config"
                    ),
                    "warmup_runs_per_config": 1,
                    "fresh_measurement_runs_per_config": 4,
                    "measurements": {
                        "stock-eager": stock,
                        "vllm-paged-q16": fused,
                    },
                    "paired": {
                        "generated_tokens_exact": True,
                        "ttft_ratio_fused_vs_stock": fused["ttft_seconds"]
                        / stock["ttft_seconds"],
                        "tpot_ratio_fused_vs_stock": fused["median_tpot_seconds"]
                        / stock["median_tpot_seconds"],
                        "persistent_ratio_fused_vs_stock": 0.75,
                        "cuda_peak_delta_ratio_fused_vs_stock": 0.5,
                    },
                    "test_v2_consumed": False,
                }
                (validation / f"vllm-paged-q16-shard-{rank}.json").write_text(
                    json.dumps(row)
                )
            summary = summarize_validation_shards(
                root,
                authorization_path=authorization_path,
                expected_authorization_sha256=authorization_sha,
                expected_code_ledger_sha256=ledger_sha,
                expected_model_manifest_sha256="model-manifest",
                expected_model_artifact_ledger_sha256=model_artifact_sha,
                expected_model_weight_ledger_sha256=model_weight_sha,
                expected_source_revision="revision",
                expected_calls_per_layer=2,
            )
            self.assertEqual(summary["generated_tokens_exact_fraction"], 1.0)
            self.assertEqual(summary["fresh_measurement_runs_per_config"], 4)
            self.assertEqual(summary["dense_fallback_calls"], 0)
            self.assertEqual(
                summary["position_ids_contract"], POST_ROPE_POSITION_IDS_CONTRACT
            )
            self.assertEqual(
                summary["production_position_ids_validation_host_syncs"], 0
            )
            self.assertEqual(summary["median_query_cuda_peak_delta_ratio_fused_vs_stock"], 0.5)
            self.assertEqual(summary["median_q16_document_payload_nbytes"], 1_000)
            self.assertFalse(summary["multi_query_serving_completed"])
            self.assertFalse(summary["nvml_process_memory_sampled"])

            broken = json.loads(
                (validation / "vllm-paged-q16-shard-0.json").read_text()
            )
            broken["measurements"]["vllm-paged-q16"]["fresh_trials"][0][
                "intercept"
            ]["mask_validation_host_syncs"] = 1
            (validation / "vllm-paged-q16-shard-0.json").write_text(
                json.dumps(broken)
            )
            with self.assertRaisesRegex(RuntimeError, "synchronized for mask"):
                summarize_validation_shards(
                    root,
                    authorization_path=authorization_path,
                    expected_authorization_sha256=authorization_sha,
                    expected_code_ledger_sha256=ledger_sha,
                    expected_model_manifest_sha256="model-manifest",
                    expected_model_artifact_ledger_sha256=model_artifact_sha,
                    expected_model_weight_ledger_sha256=model_weight_sha,
                    expected_source_revision="revision",
                    expected_calls_per_layer=2,
                )


if __name__ == "__main__":
    unittest.main()
