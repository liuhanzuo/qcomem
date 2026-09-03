"""Disk-only verifier for the fixed method-v3 campaign."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from v3_authority import (
    ALLOCATOR_PHASES, Authority, CAMPAIGN_ID, FORMAL_CONFIG_PATH, LANES,
    MODEL_REVISION, VOCAB_SIZE, load_authority, load_fixed_formal_config,
)
from v3_capture import FORMAL_BACKEND_KIND, LIVE_SOURCE_KIND
from v3_common import (
    ContractError, exact_keys, load_json_file, require, require_regular_file_under,
    require_sha256, safe_relative_path, sha256_bytes, sha256_file, verify_seal,
)
from v3_formal import FaultBinding, FormalView, validate_formal_mapping


STRUCTURAL_FIELDS = (
    "kv_logical_length", "kv_content_sha256", "gdn_content_sha256",
    "kv_version", "gdn_version", "kv_commit_epoch", "gdn_commit_epoch",
)
LIVE_FIELDS = (
    "request_id", "kv_logical_length", "kv_content_sha256", "gdn_content_sha256",
    "kv_inventory_sha256", "gdn_inventory_sha256", "kv_version", "gdn_version",
    "kv_commit_epoch", "gdn_commit_epoch", "observation_id", "sync_event_id",
    "gpu_uuid", "device_index", "source_kind", "backend_kind", "synchronized",
)
RECEIPT_FIELDS = (
    "schema_version", "campaign_id", "run_id", "lane", "fault_id", "gpu_uuid",
    "device_index", "call_key", "model_revision", "authoritative_config_sha256",
    "formal_config_sha256", "method_core_manifest_sha256", "preregistration_sha256",
    "schedule_sha256", "atomic_policy_sha256", "designer_snapshot_manifest_sha256",
    "surfaced_token_id", "logits", "live_pre", "live_post", "state_source",
    "payload_sha256",
)


@dataclass(frozen=True)
class LaneExpectation:
    run_id: str
    lane: str
    fault: FaultBinding
    formal_config_sha256: str


@dataclass
class LaneData:
    expectation: LaneExpectation
    receipts: list[Mapping[str, Any]]
    logits: list[bytes]
    allocator: Mapping[str, Any]
    atomic_verdict: Mapping[str, Any]


class _DiskCampaignVerifier:
    """Internal engine; all scientific observations are reread from disk paths."""

    def __init__(self, authority: Authority, formal: FormalView, *, vocab_size: int = VOCAB_SIZE,
                 expected_backend_kind: str = FORMAL_BACKEND_KIND) -> None:
        require(vocab_size > 0, "verifier vocab size")
        require(expected_backend_kind in (FORMAL_BACKEND_KIND, "cpu-mock-live-v1"), "backend kind")
        self.authority = authority
        self.formal = formal
        self.vocab_size = vocab_size
        self.expected_backend_kind = expected_backend_kind
        self._global_ids: set[str] = set()

    def _register_global_id(self, value: Any, label: str) -> None:
        require(isinstance(value, str) and value != "", label)
        require(value not in self._global_ids, "duplicate global observation/sync ID")
        self._global_ids.add(value)

    def _identity_fields(self, expectation: LaneExpectation) -> Mapping[str, Any]:
        return {
            "campaign_id": CAMPAIGN_ID,
            "run_id": expectation.run_id,
            "lane": expectation.lane,
            "fault_id": expectation.fault.fault_id,
            "gpu_uuid": expectation.fault.gpu_uuid,
            "device_index": expectation.fault.device_index,
            "authoritative_config_sha256": self.authority.config_file_sha256,
            "formal_config_sha256": expectation.formal_config_sha256,
            "method_core_manifest_sha256": self.authority.method_core_manifest_sha256,
        }

    def _validate_lane_binding(self, value: Mapping[str, Any], expectation: LaneExpectation) -> None:
        exact_keys(value, (
            "schema_version", "campaign_id", "run_id", "lane", "fault_id", "gpu_uuid",
            "device_index", "model_revision", "authoritative_config_sha256",
            "formal_config_sha256", "method_core_manifest_sha256", "preregistration_sha256",
            "schedule_sha256", "atomic_policy_sha256", "designer_snapshot_manifest_sha256",
            "backend_kind", "payload_sha256",
        ), "lane binding")
        verify_seal(value, "lane binding")
        require(value.get("schema_version") == "forkaudit-method-v3-lane-binding-v1", "lane binding schema")
        for field, expected in self._identity_fields(expectation).items():
            require(value.get(field) == expected, "lane binding " + field)
        require(value.get("model_revision") == MODEL_REVISION, "lane model revision")
        require(value.get("preregistration_sha256") == self.authority.preregistration_sha256,
                "lane preregistration")
        require(value.get("schedule_sha256") == self.authority.schedule_sha256, "lane schedule")
        require(value.get("atomic_policy_sha256") == self.authority.atomic_policy_sha256, "lane atomic policy")
        require(value.get("designer_snapshot_manifest_sha256") == self.authority.designer_snapshot_manifest_sha256,
                "lane designer snapshot")
        require(value.get("backend_kind") == self.expected_backend_kind, "lane backend kind")

    def _validate_live(self, value: Mapping[str, Any], request_id: str,
                       expectation: LaneExpectation, label: str) -> None:
        exact_keys(value, LIVE_FIELDS, label)
        require(value.get("request_id") == request_id, label + " request")
        for field in ("kv_logical_length", "kv_version", "gdn_version", "kv_commit_epoch", "gdn_commit_epoch"):
            require(type(value.get(field)) is int and value[field] >= 0, label + " " + field)
        for field in ("kv_content_sha256", "gdn_content_sha256", "kv_inventory_sha256", "gdn_inventory_sha256"):
            require_sha256(value.get(field), label + " " + field)
        require(value.get("gpu_uuid") == expectation.fault.gpu_uuid, label + " GPU")
        require(value.get("device_index") == expectation.fault.device_index, label + " device")
        require(value.get("source_kind") == LIVE_SOURCE_KIND, label + " source kind")
        require(value.get("backend_kind") == self.expected_backend_kind, label + " backend")
        require(value.get("synchronized") is True, label + " synchronization")
        self._register_global_id(value.get("observation_id"), label + " observation ID")
        self._register_global_id(value.get("sync_event_id"), label + " sync ID")

    def _load_logits(self, lane_root: Path, descriptor: Mapping[str, Any], call_index: int) -> bytes:
        exact_keys(descriptor, ("path", "sha256", "nbytes", "shape", "dtype"), "logit descriptor")
        expected_relative = Path("logits/call-%03d.f32le" % call_index)
        require(safe_relative_path(descriptor.get("path"), "logit path") == expected_relative, "logit path binding")
        path = require_regular_file_under(lane_root, expected_relative, "logit sidecar")
        raw = path.read_bytes()
        require(descriptor.get("dtype") == "float32-little-endian", "logit dtype")
        require(descriptor.get("shape") == [1, self.vocab_size], "logit shape")
        require(descriptor.get("nbytes") == self.vocab_size * 4 == len(raw), "logit size")
        require(require_sha256(descriptor.get("sha256"), "logit") == sha256_bytes(raw), "logit hash")
        values = array("f")
        values.frombytes(raw)
        if sys.byteorder != "little":
            values.byteswap()
        require(len(values) == self.vocab_size and all(math.isfinite(item) for item in values),
                "finite complete logits")
        return raw

    def _validate_receipt(self, value: Mapping[str, Any], expectation: LaneExpectation,
                          call_index: int, lane_root: Path) -> bytes:
        exact_keys(value, RECEIPT_FIELDS, "call receipt")
        verify_seal(value, "call receipt")
        require(value.get("schema_version") == "forkaudit-method-v3-call-receipt-v1", "receipt schema")
        for field, expected in self._identity_fields(expectation).items():
            require(value.get(field) == expected, "receipt " + field)
        schedule = self.authority.schedule[call_index]
        require(value.get("call_key") == schedule, "receipt exact call schedule")
        require(value.get("model_revision") == MODEL_REVISION, "receipt model revision")
        require(value.get("preregistration_sha256") == self.authority.preregistration_sha256,
                "receipt preregistration")
        require(value.get("schedule_sha256") == self.authority.schedule_sha256, "receipt schedule hash")
        require(value.get("atomic_policy_sha256") == self.authority.atomic_policy_sha256,
                "receipt atomic policy")
        require(value.get("designer_snapshot_manifest_sha256") == self.authority.designer_snapshot_manifest_sha256,
                "receipt designer snapshot")
        require(type(value.get("surfaced_token_id")) is int and value["surfaced_token_id"] >= 0,
                "receipt token")
        require(value.get("state_source") == "wrapper_bound_live_tensors_not_model_result", "receipt state source")
        pre = value.get("live_pre")
        post = value.get("live_post")
        self._validate_live(pre, schedule["request_id"], expectation, "live pre")
        self._validate_live(post, schedule["request_id"], expectation, "live post")
        return self._load_logits(lane_root, value.get("logits"), call_index)

    def _validate_allocator(self, value: Mapping[str, Any], expectation: LaneExpectation) -> Mapping[str, Any]:
        exact_keys(value, (
            "schema_version", "campaign_id", "run_id", "lane", "fault_id", "gpu_uuid",
            "device_index", "authoritative_config_sha256", "formal_config_sha256",
            "method_core_manifest_sha256", "peak_reset_before_h0", "endpoints",
            "payload_sha256",
        ), "allocator")
        verify_seal(value, "allocator")
        require(value.get("schema_version") == "forkaudit-method-v3-allocator-v1", "allocator schema")
        for field, expected in self._identity_fields(expectation).items():
            require(value.get(field) == expected, "allocator " + field)
        require(value.get("peak_reset_before_h0") is True, "allocator H0 peak reset")
        endpoints = value.get("endpoints")
        require(isinstance(endpoints, list) and len(endpoints) == 5, "allocator endpoints")
        require(all(isinstance(row, Mapping) for row in endpoints), "allocator endpoint row")
        require([row.get("phase") for row in endpoints] == list(ALLOCATOR_PHASES), "allocator phase order")
        previous_peak = -1
        for row in endpoints:
            exact_keys(row, (
                "phase", "current_allocated_bytes", "peak_allocated_bytes", "sync_event_id",
                "synchronized", "gpu_uuid", "device_index",
            ), "allocator endpoint")
            current = row.get("current_allocated_bytes")
            peak = row.get("peak_allocated_bytes")
            require(type(current) is int and current >= 0, "allocator current")
            require(type(peak) is int and peak >= current, "allocator peak")
            require(peak >= previous_peak, "allocator peak monotone")
            previous_peak = peak
            require(row.get("synchronized") is True, "allocator sync")
            require(row.get("gpu_uuid") == expectation.fault.gpu_uuid, "allocator endpoint GPU")
            require(row.get("device_index") == expectation.fault.device_index, "allocator endpoint device")
            self._register_global_id(row.get("sync_event_id"), "allocator sync ID")
        return value

    def _atomic_verdict(self, receipts: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        rows = []
        previous: dict[str, Mapping[str, Any]] = {}
        for call_index, receipt in enumerate(receipts):
            schedule = self.authority.schedule[call_index]
            pre = receipt["live_pre"]
            post = receipt["live_post"]
            checks = {
                "kv_length_delta": post["kv_logical_length"] - pre["kv_logical_length"] == schedule["input_token_count"],
                "kv_version_delta": post["kv_version"] - pre["kv_version"] == 1,
                "gdn_version_delta": post["gdn_version"] - pre["gdn_version"] == 1,
                "kv_epoch_delta": post["kv_commit_epoch"] - pre["kv_commit_epoch"] == 1,
                "gdn_epoch_delta": post["gdn_commit_epoch"] - pre["gdn_commit_epoch"] == 1,
                "pre_epoch_coherent": pre["kv_commit_epoch"] == pre["gdn_commit_epoch"],
                "post_epoch_coherent": post["kv_commit_epoch"] == post["gdn_commit_epoch"],
            }
            request_id = schedule["request_id"]
            if request_id in previous:
                checks["cross_call_continuity"] = all(
                    pre[field] == previous[request_id][field] for field in STRUCTURAL_FIELDS
                )
            else:
                checks["cross_call_continuity"] = True
            previous[request_id] = post
            rows.append({"call_index": call_index, "checks": checks, "passed": all(checks.values())})
        return {
            "schema_version": "forkaudit-method-v3-atomic-verdict-v1",
            "rows": rows,
            "passed": len(rows) == 16 and all(row["passed"] for row in rows),
            "attribution": "hybrid_atomic_coherence",
        }

    def read_lane(self, lane_root: Path, expectation: LaneExpectation) -> LaneData:
        require(lane_root.is_dir() and not lane_root.is_symlink(), "lane directory")
        required_files = {"lane-binding.json", "allocator.json"}
        required_files.update("receipts/call-%03d.json" % index for index in range(16))
        required_files.update("logits/call-%03d.f32le" % index for index in range(16))
        required_dirs = {"receipts", "logits"}
        observed_files = set()
        observed_dirs = set()
        for path in lane_root.rglob("*"):
            require(not path.is_symlink(), "symlink in lane inventory")
            relative = path.relative_to(lane_root).as_posix()
            if path.is_file():
                observed_files.add(relative)
            elif path.is_dir():
                observed_dirs.add(relative)
            else:
                raise ContractError("non-file inventory entry")
        require(observed_files == required_files, "exact lane file inventory")
        require(observed_dirs == required_dirs, "exact lane directory inventory")
        binding = load_json_file(lane_root / "lane-binding.json", "lane binding")
        self._validate_lane_binding(binding, expectation)
        receipts = []
        logits = []
        for call_index in range(16):
            receipt_path = require_regular_file_under(
                lane_root, Path("receipts/call-%03d.json" % call_index), "receipt")
            receipt = load_json_file(receipt_path, "call receipt")
            logits.append(self._validate_receipt(receipt, expectation, call_index, lane_root))
            receipts.append(receipt)
        allocator = self._validate_allocator(load_json_file(lane_root / "allocator.json", "allocator"), expectation)
        return LaneData(expectation, receipts, logits, allocator, self._atomic_verdict(receipts))

    @staticmethod
    def semantic_pair(reference: LaneData, candidate: LaneData) -> Mapping[str, Any]:
        rows = []
        for call_index, (ref_receipt, cand_receipt, ref_raw, cand_raw) in enumerate(zip(
                reference.receipts, candidate.receipts, reference.logits, candidate.logits)):
            token_exact = ref_receipt["surfaced_token_id"] == cand_receipt["surfaced_token_id"]
            logits_exact = ref_raw == cand_raw
            rows.append({
                "call_index": call_index, "token_exact": token_exact,
                "complete_fp32_logits_byte_exact": logits_exact,
                "passed": token_exact and logits_exact,
            })
        return {
            "schema_version": "forkaudit-method-v3-semantic-verdict-v1",
            "rows": rows, "passed": len(rows) == 16 and all(row["passed"] for row in rows),
            "attribution": "paired_complete_semantic_baseline",
        }

    @staticmethod
    def structural_pair(reference: LaneData, candidate: LaneData) -> Mapping[str, Any]:
        rows = []
        for call_index, (ref_receipt, cand_receipt) in enumerate(zip(reference.receipts, candidate.receipts)):
            checks = {}
            for endpoint in ("live_pre", "live_post"):
                for field in STRUCTURAL_FIELDS:
                    checks[endpoint + "." + field] = ref_receipt[endpoint][field] == cand_receipt[endpoint][field]
            rows.append({"call_index": call_index, "checks": checks, "passed": all(checks.values())})
        return {
            "schema_version": "forkaudit-method-v3-structural-pair-verdict-v1",
            "rows": rows, "passed": len(rows) == 16 and all(row["passed"] for row in rows),
            "attribution": "paired_live_structural_state",
        }

    @staticmethod
    def allocator_pair(reference: LaneData, candidate: LaneData) -> Mapping[str, Any]:
        ref_rows = reference.allocator["endpoints"]
        cand_rows = candidate.allocator["endpoints"]
        rows = []
        for ref, cand in zip(ref_rows, cand_rows):
            current_exact = ref["current_allocated_bytes"] == cand["current_allocated_bytes"]
            peak_exact = ref["peak_allocated_bytes"] == cand["peak_allocated_bytes"]
            rows.append({
                "phase": ref["phase"], "current_exact": current_exact,
                "peak_exact": peak_exact, "passed": current_exact and peak_exact,
            })
        reference_restored = ref_rows[-1]["current_allocated_bytes"] == ref_rows[0]["current_allocated_bytes"]
        candidate_restored = cand_rows[-1]["current_allocated_bytes"] == cand_rows[0]["current_allocated_bytes"]
        return {
            "schema_version": "forkaudit-method-v3-allocator-pair-verdict-v1",
            "rows": rows, "reference_restored": reference_restored,
            "candidate_restored": candidate_restored,
            "passed": reference_restored and candidate_restored and all(row["passed"] for row in rows),
            "attribution": "paired_allocator_baseline",
        }

    def verify_campaign(self) -> Mapping[str, Any]:
        artifacts = self.formal.output_root / "artifacts"
        require(artifacts.is_dir() and not artifacts.is_symlink(), "artifacts root")
        require({path.name for path in artifacts.iterdir()} == {row.fault_id for row in self.formal.faults},
                "exact fault-directory inventory")
        results = []
        for fault in self.formal.faults:
            fault_root = artifacts / fault.fault_id
            require(fault_root.is_dir() and not fault_root.is_symlink(), "fault directory")
            require({path.name for path in fault_root.iterdir()} == set(LANES), "exact lane-directory inventory")
            lanes = {}
            for lane in LANES:
                expectation = LaneExpectation(self.formal.run_id, lane, fault, self.formal.config_file_sha256)
                lanes[lane] = self.read_lane(fault_root / lane, expectation)
            clean_semantic = self.semantic_pair(lanes["reference"], lanes["clean"])
            clean_structural = self.structural_pair(lanes["reference"], lanes["clean"])
            clean_allocator = self.allocator_pair(lanes["reference"], lanes["clean"])
            clean_valid = all((
                lanes["reference"].atomic_verdict["passed"], lanes["clean"].atomic_verdict["passed"],
                clean_semantic["passed"], clean_structural["passed"], clean_allocator["passed"],
            ))
            mutant_semantic = self.semantic_pair(lanes["reference"], lanes["mutant"])
            mutant_structural = self.structural_pair(lanes["reference"], lanes["mutant"])
            mutant_allocator = self.allocator_pair(lanes["reference"], lanes["mutant"])
            mutant_gate_pass = all((
                lanes["mutant"].atomic_verdict["passed"], mutant_semantic["passed"],
                mutant_structural["passed"], mutant_allocator["passed"],
            ))
            results.append({
                "fault_id": fault.fault_id,
                "reference_clean_valid": clean_valid,
                "mutant_gate_pass": mutant_gate_pass,
                "detected_by_frozen_gates": clean_valid and not mutant_gate_pass,
                "reference_atomic": lanes["reference"].atomic_verdict,
                "clean_atomic": lanes["clean"].atomic_verdict,
                "mutant_atomic": lanes["mutant"].atomic_verdict,
                "clean_semantic": clean_semantic, "clean_structural": clean_structural,
                "clean_allocator": clean_allocator, "mutant_semantic": mutant_semantic,
                "mutant_structural": mutant_structural, "mutant_allocator": mutant_allocator,
            })
        return {
            "schema_version": "forkaudit-method-v3-campaign-verdict-v1",
            "campaign_id": CAMPAIGN_ID, "run_id": self.formal.run_id,
            "faults": results, "fault_count": 8,
            "population_detection_rate_computed": False,
            "all_observation_and_sync_ids_unique": True,
        }


def verify_fixed_campaign() -> Mapping[str, Any]:
    """The sole public verifier entry point.  It accepts no caller input."""

    authority = load_authority()
    formal_mapping = load_fixed_formal_config()
    formal_sha = sha256_file(FORMAL_CONFIG_PATH)
    formal = validate_formal_mapping(formal_mapping, formal_sha, authority)
    return _DiskCampaignVerifier(authority, formal).verify_campaign()

