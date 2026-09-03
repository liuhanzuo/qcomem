from __future__ import annotations

import contextlib
import json
import tempfile
import threading
import unittest
from http.server import HTTPServer
from pathlib import Path
from urllib import request as urllib_request

import run_qcomem_streaming_related_compare as target


def measured(prediction: str, f1: float, scale: float = 1.0):
    return {
        "prediction": prediction,
        "f1": f1,
        "ttft_seconds": 0.1 * scale,
        "e2e_seconds": 0.5 * scale,
        "median_tpot_seconds": 0.01 * scale,
        "generated_tokens_per_second": 20.0 / scale,
        "usage": {"completion_tokens": 10},
    }


class ComparisonTests(unittest.TestCase):
    def test_phase_contract(self):
        self.assertEqual(
            target.phases_for_config("qcomem-d7-r8-a8-l8"),
            ("cache_off", "cache_on"),
        )
        with self.assertRaises(target.CompareError):
            target.phases_for_config("dense-recompute")
        with self.assertRaises(target.CompareError):
            target.phases_for_config("not-a-method")

    def test_sse_event_has_common_schema(self):
        raw = target._sse_event(
            "token",
            finish_reason=None,
            usage={"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
        )
        self.assertTrue(raw.startswith(b"data: "))
        value = json.loads(raw.removeprefix(b"data: ").strip())
        self.assertEqual(value["choices"][0]["text"], "token")
        self.assertEqual(value["usage"]["completion_tokens"], 1)

    def test_http_stream_is_consumed_by_the_frozen_common_client(self):
        from run_related_work_serving_baseline import stream_completion

        class Tokenizer:
            @staticmethod
            def decode(tokens, skip_special_tokens=True):
                del skip_special_tokens
                return "".join(chr(96 + token) for token in tokens)

        class Runtime:
            document_ids = (10, 11)
            query_ids = (12, 13, 14, 15, 16, 17, 18, 19)
            full_ids = document_ids + query_ids
            tokenizer = Tokenizer()
            metrics = {"queries": 0, "hits": 0}
            inference_entries = 0

            @contextlib.contextmanager
            def inference_context(self):
                type(self).inference_entries += 1
                yield

            def reset(self):
                return None

            def state_receipt(self):
                return {"config": "qcomem-d7-r8-a8-l8"}

            def prime(self, config, prompt):
                del config, prompt
                return {"request_role": "prime", "persistent_state_nbytes": 7}

            def complete(self, config, prompt, max_tokens):
                del config, prompt
                return iter((1, 2, 3)[:max_tokens])

        server = HTTPServer(("127.0.0.1", 0), target.make_handler(Runtime()))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = stream_completion(
                f"http://127.0.0.1:{server.server_port}/v1/completions",
                {
                    "stream": True,
                    "prompt": list(Runtime.full_ids),
                    "max_tokens": 3,
                    "qcomem_config": "qcomem-d7-r8-a8-l8",
                },
                timeout=5.0,
            )
            self.assertEqual(result["prediction"], "abc")
            self.assertEqual(result["usage"]["completion_tokens"], 3)
            self.assertGreater(result["ttft_seconds"], 0.0)

            prime_request = urllib_request.Request(
                f"http://127.0.0.1:{server.server_port}/prime",
                data=target.canonical_json_bytes(
                    {
                        "qcomem_config": "qcomem-d7-r8-a8-l8",
                        "prompt": list(Runtime.document_ids),
                    }
                ),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib_request.urlopen(prime_request, timeout=5.0) as response:
                receipt = json.loads(response.read())
            self.assertEqual(receipt["request_role"], "prime")
            self.assertEqual(receipt["persistent_state_nbytes"], 7)
            self.assertEqual(Runtime.inference_entries, 2)
        finally:
            server.shutdown()
            thread.join(timeout=5.0)
            server.server_close()

    def test_aggregate_replays_all_cells_and_imports_bound_summaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for config in target.EXPECTED_CONFIGS:
                for phase in target.phases_for_config(config):
                    for rank, (dataset, source_index) in enumerate(target.EXPECTED_PAIRS):
                        hit = phase == "cache_on"
                        value = {
                            "schema": "forkaudit-comem-common-streaming-shard-v1",
                            "status": "completed",
                            "system": "comem-transformers-stream-wrapper-v1",
                            "config": config,
                            "phase": phase,
                            "rank": rank,
                            "world_size": 8,
                            "workload": {
                                "workload_id": f"{dataset}-{source_index}",
                                "dataset": dataset,
                                "source_index": source_index,
                            },
                            "measured": measured(f"prediction-{rank}", 0.4),
                            "state_receipt": {
                                "config": config,
                                "cache_hit": hit,
                                "persistent_state_nbytes": 2**20,
                            },
                            "prefix_counters": {
                                "measured_delta": {"queries": 1.0, "hits": 1.0 if hit else 0.0}
                            },
                        }
                        target.atomic_json(
                            root / f"{config}-{phase}-rank-{rank}.json", value
                        )

            def baseline(system: str):
                return {
                    "schema": "forkaudit-related-serving-summary-v1",
                    "scientific_run_valid": True,
                    "system": system,
                    "comparison_boundary": "same-model-same-slice-openai-streaming-serving-only",
                    "phases": {
                        "cache_off": {
                            "mean_f1": 0.4,
                            "median_ttft_seconds": 0.2,
                            "median_tpot_seconds": 0.02,
                            "median_generated_tokens_per_second": 10.0,
                        },
                        "cache_on": {
                            "mean_f1": 0.4,
                            "median_ttft_seconds": 0.1,
                            "median_tpot_seconds": 0.01,
                            "median_generated_tokens_per_second": 20.0,
                        }
                    },
                }

            vllm = root / "vllm.json"
            sglang = root / "sglang.json"
            target.atomic_json(vllm, baseline("vllm-test"))
            target.atomic_json(sglang, baseline("sglang-test"))
            output = root / "summary.json"
            args = type(
                "Args",
                (),
                {
                    "input_dir": root,
                    "vllm_summary": vllm,
                    "sglang_summary": sglang,
                    "output": output,
                },
            )()
            target.aggregate_stage(args)
            summary = json.loads(output.read_text())
            self.assertTrue(summary["hypothesis_passed"])
            self.assertEqual(len(summary["leaderboard"]), 9)
            self.assertEqual(
                summary["configs"]["qcomem-d7-r8-a8-l8"][
                    "cache_off_vs_on_prediction_exact"
                ],
                [True] * 8,
            )
            self.assertEqual(
                summary["configs"]["full-prefix-q16"]["phases"]["cache_on"][
                    "median_persistent_state_mib"
                ],
                1.0,
            )

    def test_clean_false_positive_is_not_averaged_away(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = "qcomem-d7-r8-a8-l8"
            for phase in ("cache_off", "cache_on"):
                for rank, (dataset, source_index) in enumerate(target.EXPECTED_PAIRS):
                    prediction = "changed" if phase == "cache_on" and rank == 0 else "same"
                    value = {
                        "schema": "forkaudit-comem-common-streaming-shard-v1",
                        "status": "completed",
                        "config": config,
                        "phase": phase,
                        "rank": rank,
                        "world_size": 8,
                        "workload": {"dataset": dataset, "source_index": source_index},
                        "measured": measured(prediction, 0.4),
                        "state_receipt": {
                            "config": config,
                            "cache_hit": phase == "cache_on",
                            "persistent_state_nbytes": 1,
                        },
                        "prefix_counters": {
                            "measured_delta": {"hits": 1.0 if phase == "cache_on" else 0.0}
                        },
                    }
                    target.atomic_json(root / f"{config}-{phase}-rank-{rank}.json", value)
            off = [target._load_shard(root / f"{config}-cache_off-rank-{rank}.json", config, "cache_off", rank) for rank in range(8)]
            on = [target._load_shard(root / f"{config}-cache_on-rank-{rank}.json", config, "cache_on", rank) for rank in range(8)]
            exact = [off[index]["measured"]["prediction"] == on[index]["measured"]["prediction"] for index in range(8)]
            self.assertEqual(exact.count(False), 1)


if __name__ == "__main__":
    unittest.main()
