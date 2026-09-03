#!/usr/bin/env python3
"""CPU fixtures for the candidate-import-free R30 replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import r30_e2e_replay as replay


def write_json(path: Path, value: object) -> str:
    path.write_bytes(replay.canonical_bytes(value))
    return replay.sha256_file(path)


def token_sha(values: list[int]) -> str:
    return replay.int64_sha256(values)


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.artifacts = root / "artifacts"
        self.artifacts.mkdir()
        self.input_path = root / "input.json"
        self.reference_path = root / "reference.json"
        self.candidate_path = root / "candidate.json"
        self.output_path = root / "replay.json"
        self.inputs = self._inputs()
        self.input_sha = write_json(self.input_path, self.inputs)
        self.reference, reference_arrays = self._reference()
        self._write_sidecars(self.reference, reference_arrays, "reference")
        self.reference_sha = write_json(self.reference_path, self.reference)
        self.candidate, candidate_arrays = self._candidate()
        self._write_sidecars(self.candidate, candidate_arrays, "candidate")
        self.candidate_sha = write_json(self.candidate_path, self.candidate)

    @staticmethod
    def _inputs() -> dict:
        cases = []
        for case_index in range(2):
            document = [1000 + case_index] * 4095
            queries = []
            for request_index in range(2):
                tokens = [2000 + case_index * 10 + request_index] * 32
                queries.append(
                    {
                        "request_index": request_index,
                        "token_ids": tokens,
                        "token_ids_sha256": token_sha(tokens),
                    }
                )
            cases.append(
                {
                    "case_index": case_index,
                    "case_id": f"case-{case_index}",
                    "document_token_ids": document,
                    "document_token_ids_sha256": token_sha(document),
                    "queries": queries,
                }
            )
        return {
            "schema_version": replay.INPUT_SCHEMA,
            "created_before_any_model_output": True,
            "candidate_or_reference_model_invoked": False,
            "selection": {"greedy_steps": 4},
            "cases": cases,
        }

    def _reference(self) -> tuple[dict, dict[str, np.ndarray]]:
        rows = []
        sidecars = []
        arrays = {}
        for case in self.inputs["cases"]:
            for query in case["queries"]:
                history = case["document_token_ids"] + query["token_ids"]
                generated = []
                steps = []
                for step in range(4):
                    token = (case["case_index"] * 2 + query["request_index"] + step) % 5
                    array = np.full(5, -1.0, dtype=np.float32)
                    array[token] = 3.0
                    record_id = f"ref/c{case['case_index']}/q{query['request_index']}/s{step}"
                    arrays[record_id] = array
                    sidecars.append(self._receipt(record_id, f"reference/{record_id}.npy", array))
                    steps.append(
                        {
                            "step_index": step,
                            "raw_history_token_count": len(history),
                            "raw_history_token_ids_sha256": token_sha(history),
                            "generated_token_id": token,
                            "logit_record_id": record_id,
                        }
                    )
                    generated.append(token)
                    history.append(token)
                rows.append(
                    {
                        "case_index": case["case_index"],
                        "request_index": query["request_index"],
                        "document_token_ids_sha256": case["document_token_ids_sha256"],
                        "query_token_ids_sha256": query["token_ids_sha256"],
                        "generated_token_ids": generated,
                        "steps": steps,
                    }
                )
        return (
            {
                "schema_version": replay.REFERENCE_SCHEMA,
                "input_manifest_sha256": self.input_sha,
                "reference_source_distinct": True,
                "candidate_cache_trace_tensor_objects_imported": False,
                "full_model_recompute_each_step": True,
                "use_cache": False,
                "imports": {"observed_forbidden_modules": []},
                "rows": rows,
                "sidecars": sidecars,
            },
            arrays,
        )

    @staticmethod
    def _receipt(record_id: str, relative: str, array: np.ndarray) -> dict:
        return {
            "record_id": record_id,
            "path": relative,
            "sha256": "pending",
            "shape": list(array.shape),
            "dtype": "float32",
            "argmax_token_id": int(np.argmax(array)),
        }

    def _write_sidecars(self, owner: dict, arrays: dict[str, np.ndarray], prefix: str) -> None:
        for receipt in owner["sidecars"]:
            path = self.artifacts / receipt["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as handle:
                np.save(handle, arrays[receipt["record_id"]], allow_pickle=False)
            receipt["sha256"] = replay.sha256_file(path)

    @staticmethod
    def _gdn_snapshot(borrowed_setup: bool, setup: bool, phase: str) -> dict:
        rows = []
        for request in range(2):
            for layer in range(40):
                if layer in replay.FULL_LAYERS:
                    continue
                for family in ("conv_states", "recurrent_states"):
                    state = f"{layer}-{family}"
                    base_id = hashlib.sha256(f"base-{state}".encode()).hexdigest()
                    request_id = (
                        base_id
                        if borrowed_setup and setup
                        else hashlib.sha256(f"request-{request}-{state}-{phase}".encode()).hexdigest()
                    )
                    rows.append(
                        {
                            "request_index": request,
                            "layer_index": layer,
                            "family": family,
                            "state_index": 0,
                            "request_storage_id": request_id,
                            "request_byte_interval": [0, 4],
                            "base_storage_id": base_id,
                            "base_byte_interval": [0, 4],
                            "content_sha256": "a" * 64,
                            "base_content_sha256": "a" * 64,
                            "exact_base_alias": borrowed_setup and setup,
                            "base_overlap": borrowed_setup and setup,
                            "peer_overlap_count": 1 if borrowed_setup and setup else 0,
                        }
                    )
        return {
            "phase": phase,
            "all_request_base_disjoint": not (borrowed_setup and setup),
            "all_request_peer_disjoint": not (borrowed_setup and setup),
            "exact_base_alias_count": 120 if borrowed_setup and setup else 0,
            "rows": rows,
        }

    @staticmethod
    def _kv_snapshot(shared: bool, setup: bool, step: int | None, phase: str) -> dict:
        layers = []
        for layer in replay.FULL_LAYERS:
            source = [f"source-k-{layer}", f"source-v-{layer}"]
            requests = []
            for request in range(2):
                request_storage = source if shared else [f"fresh-{request}-k-{layer}", f"fresh-{request}-v-{layer}"]
                reservation = [1000 + layer * 10 + request]
                active = list(range(32))
                active[31] = 31 if setup else reservation[0]
                requests.append(
                    {
                        "request_index": request,
                        "sequence_length": 4095 if setup else 4095 + 32 + int(step),
                        "request_storage_ids": request_storage,
                        "shares_source_storage": shared,
                        "reservation_ids": reservation,
                        "active_block_table": active,
                        "active_tail_physical_id": active[31],
                        "tail_is_source_document_block": setup,
                        "tail_is_private_reservation": not setup,
                        "append_event_count": 0 if setup else int(step) + 1,
                    }
                )
            layers.append({"layer_index": layer, "source_storage_ids": source, "requests": requests})
        return {"phase": phase, "layers": layers}

    @staticmethod
    def _repairs() -> list[dict]:
        rows = []
        item_rows = [{"base_disjoint": True, "all_peers_disjoint": True}] * 30
        for step in range(1, 4):
            for request in range(2):
                base = {
                    "schema_version": replay.REPAIR_SCHEMA,
                    "request_index": request,
                    "resident_count": 2,
                    "conv_tensor_count": 30,
                    "cloned_tensor_count": 0,
                    "ownership_only_change": True,
                    "fault_id_specialization": False,
                    "rows": item_rows,
                }
                rows.append({"step_index": step, "request_index": request, "primary": base, "immediate_repeat": dict(base)})
        return rows

    @staticmethod
    def _intercepts(kv_policy: str) -> list[dict]:
        calls = [
            {"current_append_delta_tokens": 32 if index < 10 else 1, "materialized_attention_mask_nbytes": 0}
            for index in range(40)
        ]
        identity = {"module": "vllm", "qualname": "unified_attention", "signature": "(q,k,v)", "callable_id": 9}
        return [
            {
                "verified": True,
                "request_index": request,
                "resident_count": 2,
                "request_policy": kv_policy,
                "same_unified_attention_kernel": True,
                "total_calls": 40,
                "counts": {str(layer): 4 for layer in replay.FULL_LAYERS},
                "calls": calls,
                "kernel_identity": identity,
            }
            for request in range(2)
        ]

    def _candidate(self) -> tuple[dict, dict[str, np.ndarray]]:
        reference_map = {(row["case_index"], row["request_index"]): row for row in self.reference["rows"]}
        rows = []
        sidecars = []
        arrays = {}
        for case in range(2):
            for arm in replay.ARM_IDS:
                shared = arm.startswith("shared-")
                borrowed = arm.endswith("-borrowed")
                kv_policy = "vllm-q16-shared-document-reuse" if shared else "vllm-q16-fresh-full-copy-control"
                gdn_policy = "borrow-immutable-base-functional-rebind" if borrowed else "materialize-request-base-functional-rebind"
                for track in replay.TRACKS:
                    trajectories = []
                    for request in range(2):
                        reference_row = reference_map[(case, request)]
                        generated = list(reference_row["generated_token_ids"])
                        steps = []
                        query = self.inputs["cases"][case]["queries"][request]
                        for step, token in enumerate(generated):
                            record_id = f"cand/{arm}/{track}/c{case}/q{request}/s{step}"
                            array = np.full(5, -1.0, dtype=np.float32)
                            array[token] = 2.5
                            arrays[record_id] = array
                            sidecars.append(self._receipt(record_id, f"candidate/{record_id}.npy", array))
                            expected_input = query["token_ids"] if step == 0 else [generated[step - 1]]
                            steps.append(
                                {
                                    "step_index": step,
                                    "input_token_count": len(expected_input),
                                    "input_token_ids_sha256": token_sha(expected_input),
                                    "single_token_repair_applied": step > 0,
                                    "logit_record_id": record_id,
                                    "candidate_argmax_token_id": token,
                                }
                            )
                        trajectories.append(
                            {
                                "request_index": request,
                                "query_token_ids_sha256": query["token_ids_sha256"],
                                "generated_token_ids": generated,
                                "steps": steps,
                            }
                        )
                    rounds = [
                        {
                            "step_index": step,
                            "gdn": self._gdn_snapshot(False, False, f"after-round-{step}"),
                            "kv": self._kv_snapshot(shared, False, step, f"after-round-{step}"),
                        }
                        for step in range(4)
                    ]
                    rows.append(
                        {
                            "case_index": case,
                            "arm_id": arm,
                            "track": track,
                            "kv_policy": kv_policy,
                            "gdn_base_policy": gdn_policy,
                            "source_document_immutable": True,
                            "source_document_sha256_before": {"3": "a"},
                            "source_document_sha256_after": {"3": "a"},
                            "persistent_gdn_immutable": True,
                            "persistent_gdn_before": {"sha256": "b"},
                            "persistent_gdn_after": {"sha256": "b"},
                            "setup_ownership": {
                                "gdn": self._gdn_snapshot(borrowed, True, "setup-before-query"),
                                "kv": self._kv_snapshot(shared, True, None, "setup-before-query"),
                            },
                            "round_ownership": rounds,
                            "repair_receipts": self._repairs(),
                            "intercepts": self._intercepts(kv_policy),
                            "trajectories": trajectories,
                        }
                    )
        return (
            {
                "schema_version": replay.CANDIDATE_SCHEMA,
                "input_manifest_sha256": self.input_sha,
                "reference_result_sha256": self.reference_sha,
                "reference_logits_or_candidate_objects_consumed": False,
                "repair_prerequisite": {
                    "source_sha256": "4" * 64,
                    "clean_result_sha256": "a" * 64,
                    "detached_replay_sha256": "f" * 64,
                    "clean_regression_passed_before_execution": True,
                },
                "claim_boundary": {"fixed_runtime_only": True, "runtime_portability_claimed": False},
                "rows": rows,
                "sidecars": sidecars,
            },
            arrays,
        )

    def args(self) -> SimpleNamespace:
        return SimpleNamespace(
            input_manifest=self.input_path,
            expected_input_sha256=self.input_sha,
            reference=self.reference_path,
            expected_reference_sha256=self.reference_sha,
            candidate=self.candidate_path,
            expected_candidate_sha256=self.candidate_sha,
            artifact_root=self.artifacts,
            expected_repair_sha256="4" * 64,
            expected_clean_result_sha256="a" * 64,
            expected_detached_replay_sha256="f" * 64,
            output=self.output_path,
        )


class ReplayTest(unittest.TestCase):
    def test_complete_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            value = replay.replay(fixture.args())
            self.assertTrue(value["primary_gate_passed"])
            self.assertEqual(value["full_vocabulary_secondary"]["comparisons"], 64)
            self.assertEqual(value["full_vocabulary_secondary"]["summary"]["top1_equal_count"], 64)

    def test_token_mismatch_is_valid_negative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            row = next(row for row in fixture.candidate["rows"] if row["track"] == "greedy")
            row["trajectories"][0]["generated_token_ids"][3] = (row["trajectories"][0]["generated_token_ids"][3] + 1) % 5
            step = row["trajectories"][0]["steps"][3]
            record_id = step["logit_record_id"]
            receipt = next(item for item in fixture.candidate["sidecars"] if item["record_id"] == record_id)
            path = fixture.artifacts / receipt["path"]
            array = np.full(5, -1.0, dtype=np.float32)
            array[row["trajectories"][0]["generated_token_ids"][3]] = 2.5
            with path.open("wb") as handle:
                np.save(handle, array, allow_pickle=False)
            receipt["sha256"] = replay.sha256_file(path)
            receipt["argmax_token_id"] = int(np.argmax(array))
            step["candidate_argmax_token_id"] = int(np.argmax(array))
            fixture.candidate_sha = write_json(fixture.candidate_path, fixture.candidate)
            args = fixture.args()
            args.expected_candidate_sha256 = fixture.candidate_sha
            value = replay.replay(args)
            self.assertFalse(value["primary_gate_passed"])
            self.assertEqual(value["scientific_outcome"], "valid_negative_primary_semantic_control")


if __name__ == "__main__":
    unittest.main()
