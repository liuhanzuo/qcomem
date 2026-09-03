from __future__ import annotations

import io
import json
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

import run_related_work_serving_baseline as baseline


class _Response:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return iter(self.lines)

    def __exit__(self, *args):
        return False


class ServingBaselineTests(unittest.TestCase):
    def test_self_contained_f1_and_frozen_workload_loader(self):
        self.assertEqual(baseline.answer_f1("The blue fox", "blue fox"), 1.0)

        class FakeTokenizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def apply_chat_template(self, messages, **kwargs):
                return "<bos>" + messages[0]["content"] + "<assistant>"

            def encode(self, text, add_special_tokens=False):
                return list(text.encode("utf-8"))

        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data.jsonl"
            rows = []
            for dataset in ("qasper", "2wikimqa"):
                for source_index in range(4, 10):
                    rows.append(
                        {
                            "dataset": dataset,
                            "_source_index": source_index,
                            "_source_revision": baseline.SOURCE_REVISION,
                            "input": f"question-{source_index}",
                            "context": f"context-{source_index}",
                            "answers": [f"answer-{source_index}"],
                        }
                    )
            data.write_text("".join(json.dumps(row) + "\n" for row in rows))
            fake_transformers = types.SimpleNamespace(AutoTokenizer=FakeTokenizer)
            with mock.patch.object(
                baseline, "DATA_SHA256", baseline.sha256_file(data)
            ), mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
                workload = baseline.load_workload(Path("/model"), data, rank=7)
            self.assertEqual(
                (workload["dataset"], workload["source_index"]),
                ("2wikimqa", 9),
            )
            self.assertEqual(
                workload["metadata"]["data_sha256"], baseline.sha256_file(data)
            )
            self.assertEqual(
                workload["metadata"]["prompt_protocol"], "longbench-v1-official"
            )
            self.assertTrue(workload["document_token_ids"])
            self.assertTrue(workload["query_token_ids"])

    def test_stream_completion_uses_authoritative_usage(self):
        events = [
            {"choices": [{"text": "answer", "finish_reason": None}]},
            {
                "choices": [{"text": "", "finish_reason": "length"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 2},
            },
        ]
        lines = [f"data: {json.dumps(event)}\n".encode() for event in events]
        lines.append(b"data: [DONE]\n")
        ticks = iter((1.0, 1.1, 1.2, 1.3))

        value = baseline.stream_completion(
            "http://unused/v1/completions",
            {"stream": True},
            timeout=1.0,
            clock=lambda: next(ticks),
            opener=lambda *args, **kwargs: _Response(lines),
        )
        self.assertEqual(value["prediction"], "answer")
        self.assertEqual(value["usage"]["completion_tokens"], 2)
        self.assertAlmostEqual(value["ttft_seconds"], 0.1)
        self.assertAlmostEqual(value["median_tpot_seconds"], 0.2)

    def test_stream_completion_accepts_empty_cache_prime_but_rejects_error(self):
        empty_event = {
            "choices": [{"text": "", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 1},
        }
        lines = [f"data: {json.dumps(empty_event)}\n".encode(), b"data: [DONE]\n"]
        ticks = iter((1.0, 1.1, 1.2))
        value = baseline.stream_completion(
            "http://unused/v1/completions",
            {"stream": True},
            timeout=1.0,
            require_text=False,
            clock=lambda: next(ticks),
            opener=lambda *args, **kwargs: _Response(lines),
        )
        self.assertEqual(value["prediction"], "")
        self.assertEqual(value["usage"]["completion_tokens"], 1)

        error = {"error": {"message": "backend failed", "type": "server_error"}}
        error_lines = [f"data: {json.dumps(error)}\n".encode(), b"data: [DONE]\n"]
        with self.assertRaisesRegex(baseline.BaselineError, "server error"):
            baseline.stream_completion(
                "http://unused/v1/completions",
                {"stream": True},
                timeout=1.0,
                opener=lambda *args, **kwargs: _Response(error_lines),
            )

    def test_prometheus_prefix_counter_aliases(self):
        parsed = baseline.parse_prometheus(
            "vllm:prefix_cache_queries{worker=\"0\"} 100\n"
            "vllm:prefix_cache_hits{worker=\"0\"} 80\n"
        )
        snap = baseline.prefix_counter_snapshot(parsed)
        self.assertEqual(snap, {"queries": 100.0, "hits": 80.0})
        delta = baseline.counter_delta(
            {"queries": 100.0, "hits": 80.0},
            {"queries": 120.0, "hits": 96.0},
        )
        self.assertEqual(delta["hit_rate"], 0.8)

        parsed_sglang = baseline.parse_prometheus(
            'sglang:prompt_tokens_total{model_name="qwen"} 4096\n'
            'sglang:cached_tokens_total{model_name="qwen"} 3168\n'
        )
        self.assertEqual(
            baseline.prefix_counter_snapshot(parsed_sglang),
            {"queries": 4096.0, "hits": 3168.0},
        )

    def _write_shards(self, root: Path, *, changed: bool = False, hit: bool = True):
        for phase in ("cache_off", "cache_on"):
            for rank, (dataset, source_index) in enumerate(baseline.EXPECTED_PAIRS):
                prediction = f"answer-{rank}"
                if changed and phase == "cache_on" and rank == 0:
                    prediction = "different"
                shard = {
                    "schema": "forkaudit-related-serving-shard-v1",
                    "status": "completed",
                    "system": "vllm-0.26-prefix-align",
                    "phase": phase,
                    "rank": rank,
                    "world_size": 8,
                    "workload": {
                        "dataset": dataset,
                        "source_index": source_index,
                    },
                    "measured": {
                        "prediction": prediction,
                        "f1": 0.5,
                        "ttft_seconds": 0.2,
                        "e2e_seconds": 1.0,
                        "median_tpot_seconds": 0.025,
                        "generated_tokens_per_second": 32.0,
                        "usage": {
                            "completion_tokens": 32,
                            "prompt_tokens_details": {
                                "cached_tokens": 4000 if phase == "cache_on" and hit else 0
                            },
                        },
                    },
                    "prefix_counters": {
                        "measured_delta": {"queries": 4000.0, "hits": 4000.0 if hit else 0.0}
                    },
                }
                baseline.atomic_json(root / f"{phase}-rank-{rank}.json", shard)

    def test_aggregate_positive_and_valid_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_shards(root)
            args = type("Args", (), {
                "input_dir": root,
                "output": root / "summary.json",
                "system": "vllm-0.26-prefix-align",
            })()
            baseline.aggregate_stage(args)
            value = json.loads(args.output.read_text())
            self.assertTrue(value["hypothesis_passed"])
            self.assertEqual(value["scientific_outcome"], "valid_positive")

            self._write_shards(root, changed=True)
            baseline.aggregate_stage(args)
            value = json.loads(args.output.read_text())
            self.assertFalse(value["hypothesis_passed"])
            self.assertTrue(value["scientific_run_valid"])
            self.assertEqual(value["scientific_outcome"], "valid_negative")


if __name__ == "__main__":
    unittest.main()
