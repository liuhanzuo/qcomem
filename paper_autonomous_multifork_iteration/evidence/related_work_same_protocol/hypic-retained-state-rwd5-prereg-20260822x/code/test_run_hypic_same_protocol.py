import json
import tempfile
import unittest
from pathlib import Path

import run_hypic_same_protocol as hypic
from run_related_work_serving_baseline import BaselineError


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(value) for value in ids)


class HypicProtocolTest(unittest.TestCase):
    @staticmethod
    def cell(mode, rank, *, cached_tokens=None, prompt_sha=None):
        dataset, source_index = hypic.EXPECTED_PAIRS[rank]
        document_tokens = 100
        if cached_tokens is None:
            cached_tokens = {
                "full_recompute": 0,
                "prefix_cache": 96,
                "transition_rope_recompute": document_tokens - hypic.SEAM_TOKENS,
            }[mode]
        expected_configuration = {"pic_enable": mode == "transition_rope_recompute"}
        server_info_sha = f"{rank + 1:064x}"
        return {
            "schema": "forkaudit-hypic-same-protocol-shard-v1",
            "status": "completed",
            "formal_evidence_eligible": True,
            "mode": mode,
            "rank": rank,
            "official_commit": hypic.HYPIC_COMMIT,
            "server_launch_receipt": {
                "schema": "hypic-server-launch-receipt-v1",
                "official_commit": hypic.HYPIC_COMMIT,
                "mode": mode,
                "rank": rank,
                "tp_size": 1,
                "model_path": "/frozen/Qwen3.5-35B-A3B",
                "data_sha256": hypic.DATA_SHA256,
                "client_sha256": hypic.sha256_file(Path(hypic.__file__)),
                "source_ledger_raw_sha256": "1" * 64,
                "environment_ledger_raw_sha256": "2" * 64,
                "preregistration_sha256": "3" * 64,
                "launch_command_sha256": f"{10 + hypic.MODES.index(mode):064x}",
                "model_weight_ledger_raw_sha256": (
                    "8314a82c9188b9b817193e039b0b0eb0636b328512f19b0c12455853b7e20014"
                ),
                "model_artifact_ledger_raw_sha256": (
                    "d78424684a27718b44616c80c788e7f8b726feed674fe015901b87de1993f7dd"
                ),
                "server_info_sha256": server_info_sha,
                "server_process": {
                    "schema": "hypic-live-server-process-v1",
                    "cmdline_sha256": "f" * 64,
                },
                "hardware": {
                    "gpu_name": "NVIDIA H20-3e",
                    "gpu_uuid": f"GPU-{rank:04d}",
                    "gpu_memory_mib": 143771,
                },
                "packages": {
                    "transformers_base": "5.8.1",
                    "torch_base": "2.11.0",
                    "torch_cuda": "12.9",
                    "sglang-kernel_base": "0.4.4",
                    "sgl-deep-gemm_base": "0.1.3",
                    "sglang_base": "0.5.14",
                    "flashinfer-python_base": "0.5.3",
                    "flashinfer-cubin_base": "0.5.3",
                },
            },
            "server_info_sha256": server_info_sha,
            "server_configuration": {
                "expected": expected_configuration,
                "observed": dict(expected_configuration),
            },
            "workload": {
                "workload_id": f"{dataset}-{source_index}",
                "dataset": dataset,
                "source_index": source_index,
                "references": ["answer"],
                "document_tokens": document_tokens,
                "query_tokens": 20,
                "prompt_token_sha256": prompt_sha or f"prompt-{rank}",
                "document_token_sha256": f"document-{rank}",
                "segment_offsets": [[0, 10], [10, 100], [100, 120]],
                "warm_prompt_token_sha256": f"warm-{rank}",
                "warm_segment_disjointness": {
                    "passed": True,
                    "comparison_count": 4,
                    "matrix": [{"equal": False}] * 4,
                },
                "token_identity_verified": True,
            },
            "protocol": {
                "data_sha256": hypic.DATA_SHA256,
                "source_revision": hypic.SOURCE_REVISION,
                "max_input_tokens": 4096,
                "max_new_tokens": 32,
                "greedy": True,
                "expected_tp_size": 1,
            },
            "measured": {
                "prediction": f"{mode}-{rank}",
                "f1": 0.5,
                "ttft_seconds": 1.0 + rank,
                "median_tpot_seconds": 0.1,
                "generated_tokens_per_second": 5.0,
                "e2e_seconds": 2.0,
                "finish_reason": "length",
                "usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 32,
                },
            },
            "cache_observation": {
                "cached_tokens": cached_tokens,
                "expected_cached_tokens": (
                    document_tokens - hypic.SEAM_TOKENS
                    if mode == "transition_rope_recompute"
                    else 0
                ),
            },
            "warm_cache_observation": {
                "cached_tokens": (
                    0
                    if mode == "full_recompute"
                    else (
                        96
                        if mode == "prefix_cache"
                        else document_tokens - hypic.SEAM_TOKENS
                    )
                ),
                "document_tokens": document_tokens,
                "hit_path_exercised": True,
            },
        }

    def test_separator_is_not_a_model_token(self):
        tokenizer = FakeTokenizer()
        ids, offsets = hypic.split_and_tokenize_exact(
            "prefix" + hypic.SEP + "document" + hypic.SEP + "query", tokenizer
        )
        self.assertEqual(ids, [ord(c) for c in "prefixdocumentquery"])
        self.assertEqual(offsets, [[0, 6], [6, 14], [14, 19]])

    def test_token_digest_binds_order_and_width(self):
        self.assertEqual(hypic.token_sha256([1, 2, 3]), hypic.token_sha256([1, 2, 3]))
        self.assertNotEqual(hypic.token_sha256([1, 2, 3]), hypic.token_sha256([1, 3, 2]))

    def test_request_prompts_keep_separator_out_of_baselines(self):
        workload = {
            "warm_prime_text": "warm-prefix" + hypic.SEP + "warm-context" + hypic.SEP + "warm-dummy",
            "warm_measured_text": "warm-prefix" + hypic.SEP + "warm-context" + hypic.SEP + "warm-query",
            "hypic_prime_text": "prefix" + hypic.SEP + "document" + hypic.SEP + "dummy",
            "measured_text": "prefix" + hypic.SEP + "document" + hypic.SEP + "query",
            "warm_document_token_ids": [7, 8],
            "warm_measured_token_ids": [7, 8, 9],
            "document_token_ids": [1, 2],
            "direct_token_ids": [1, 2, 3],
        }
        baseline = hypic.request_prompts(workload, "prefix_cache")
        pic = hypic.request_prompts(workload, "transition_rope_recompute")
        self.assertEqual(baseline["formal_measured"], [1, 2, 3])
        self.assertIsInstance(pic["formal_measured"], str)
        self.assertIn(hypic.SEP, pic["formal_measured"])

    def test_aggregate_retains_approximate_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for mode in hypic.MODES:
                for rank, _ in enumerate(hypic.EXPECTED_PAIRS):
                    value = self.cell(mode, rank)
                    (root / f"{mode}-rank-{rank}.json").write_text(json.dumps(value))
            output = root / "summary.json"
            hypic.aggregate_stage(type("Args", (), {"input_dir": root, "output": output})())
            summary = json.loads(output.read_text())
            self.assertTrue(summary["cache_hypothesis_passed"])
            self.assertTrue(summary["approximate_method"])
            self.assertEqual(
                summary["modes"]["transition_rope_recompute"]
                ["prediction_text_exact_vs_full_recompute"],
                [False] * 8,
            )

    def test_exact_warm_text_can_force_distinct_first_tokens(self):
        tokenizer = FakeTokenizer()
        query_text, query_ids = hypic.text_with_exact_token_count(
            tokenizer, 24, "query", lead_candidates=("Quartz",)
        )
        dummy_text, dummy_ids = hypic.text_with_exact_token_count(
            tokenizer,
            24,
            "dummy",
            lead_candidates=("Violet",),
            forbidden_first=query_ids[0],
        )
        self.assertEqual(len(query_ids), 24)
        self.assertEqual(len(dummy_ids), 24)
        self.assertNotEqual(query_ids[0], dummy_ids[0])
        self.assertEqual(tokenizer.encode(query_text), query_ids)
        self.assertEqual(tokenizer.encode(dummy_text), dummy_ids)

    def test_segment_disjointness_rejects_any_cacheable_collision(self):
        receipt = hypic.segment_disjointness_receipt(
            {"prefix": [1], "context": [2]}, {"prefix": [3], "context": [4]}
        )
        self.assertTrue(receipt["passed"])
        with self.assertRaisesRegex(BaselineError, "segment collision"):
            hypic.segment_disjointness_receipt(
                {"prefix": [1], "context": [2]}, {"prefix": [3], "context": [2]}
            )

    def test_aggregate_rejects_query_leaking_prefix_hit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for mode in hypic.MODES:
                for rank, _ in enumerate(hypic.EXPECTED_PAIRS):
                    cached = 101 if mode == "prefix_cache" and rank == 0 else None
                    value = self.cell(mode, rank, cached_tokens=cached)
                    (root / f"{mode}-rank-{rank}.json").write_text(json.dumps(value))
            with self.assertRaisesRegex(BaselineError, "prefix cache boundary"):
                hypic.aggregate_stage(
                    type("Args", (), {"input_dir": root, "output": root / "summary.json"})()
                )

    def test_aggregate_rejects_cross_mode_prompt_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for mode in hypic.MODES:
                for rank, _ in enumerate(hypic.EXPECTED_PAIRS):
                    prompt_sha = "forged" if mode == "prefix_cache" and rank == 3 else None
                    value = self.cell(mode, rank, prompt_sha=prompt_sha)
                    (root / f"{mode}-rank-{rank}.json").write_text(json.dumps(value))
            with self.assertRaisesRegex(BaselineError, "cross-mode workload drift"):
                hypic.aggregate_stage(
                    type("Args", (), {"input_dir": root, "output": root / "summary.json"})()
                )

    def test_aggregate_rejects_cross_mode_environment_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for mode in hypic.MODES:
                for rank, _ in enumerate(hypic.EXPECTED_PAIRS):
                    value = self.cell(mode, rank)
                    if mode == "transition_rope_recompute" and rank == 4:
                        value["server_launch_receipt"][
                            "environment_ledger_raw_sha256"
                        ] = "9" * 64
                    (root / f"{mode}-rank-{rank}.json").write_text(json.dumps(value))
            with self.assertRaisesRegex(BaselineError, "cross-mode launch drift"):
                hypic.aggregate_stage(
                    type("Args", (), {"input_dir": root, "output": root / "summary.json"})()
                )

    def test_launcher_captures_server_pid_from_inside_setsid_session(self):
        launcher = Path(__file__).with_name("launch_hypic_same_protocol_8gpu.sh").read_text()
        self.assertIn('printf "%s\\n" "$$" > "$pid_file"', launcher)
        self.assertIn('SERVER_PIDS[$rank]=$(cat "$pid_file")', launcher)
        self.assertNotIn('SERVER_PIDS[$rank]=$!', launcher)
        self.assertIn('/tmp/Qwen3.5-35B-A3B-hypic-model-view', launcher)
        self.assertIn('SGLANG_NUMA_BIND_V2=0', launcher)
        self.assertIn('SGLANG_IS_FLASHINFER_AVAILABLE=0', launcher)
        self.assertIn('--sampling-backend pytorch', launcher)
        self.assertIn('--enable-cache-report', launcher)
        self.assertNotIn('--enable-metrics', launcher)


if __name__ == "__main__":
    unittest.main()
