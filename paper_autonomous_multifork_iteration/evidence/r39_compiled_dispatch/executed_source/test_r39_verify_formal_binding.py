from __future__ import annotations

import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SOURCE_DIR.parents[3]
sys.path.insert(0, str(SOURCE_DIR))

import r39_compiled_dispatch_receipts as receipts
import r39_verify_formal_binding as subject


R29_ROOT = REPO_ROOT / "paper_autonomous_multifork_iteration/evidence/r29_live_overhead"
R29_RESULT = R29_ROOT / "formal_run_20260825b/raw/formal-result.json"
R29_SIDECAR = R29_ROOT / "formal_run_20260825b/raw/audit/semantic-logits.fp32.bin"
R29_DESIGN = R29_ROOT / "preregistration.json"
R39_ROOT = SOURCE_DIR.parent
FIXTURE_LEDGER = R39_ROOT / "r29-fixtures.sha256"
EXPECTED_FIXTURE_HASHES = {
    "../r29_live_overhead/preregistration.json": (
        "2114d2cd85bedc1eafa5d1398fd0afd0d57819c0360c3be3f9ec20f1b2878939"
    ),
    "../r29_live_overhead/formal_run_20260825b/raw/formal-result.json": (
        "3ccf86e2233b560f003d965fdae05a8e3b0773e15976a05c8d70af881338bc22"
    ),
    "../r29_live_overhead/formal_run_20260825b/raw/audit/semantic-logits.fp32.bin": (
        "1c3e68ffb29ed3567c88b73757507590509c91d82b8641fca213eb39aabeaf07"
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_fixture_ledger() -> None:
    observed = {}
    for line in FIXTURE_LEDGER.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        observed[relative] = digest
    if observed != EXPECTED_FIXTURE_HASHES:
        raise AssertionError("R29 fixture ledger drift")
    for relative, expected in EXPECTED_FIXTURE_HASHES.items():
        path = R39_ROOT / relative
        if _sha256_file(path) != expected:
            raise AssertionError(f"R29 fixture hash drift: {relative}")


def _expanded_payload(root: Path) -> tuple[Path, Path, Path, dict]:
    cache, code, runtime, payload = receipts._demo_fixture(root)
    attention_template = payload["attention_calls"][0]
    gdn_template = payload["gdn_calls"][0]
    payload["attention_calls"] = []
    payload["gdn_calls"] = []
    for index in range(120):
        row = copy.deepcopy(attention_template)
        row["call_index"] = index
        payload["attention_calls"].append(row)
    linear_indices = [index for index in range(40) if index % 4 != 3]
    for _pair_index in range(6):
        for phase, sequence_length, has_previous in (
            ("document-prefill", 4033, False),
            ("request-cell", 16, True),
            ("request-cell", 16, True),
        ):
            for layer_idx in linear_indices:
                row = copy.deepcopy(gdn_template)
                row["call_index"] = len(payload["gdn_calls"])
                row["layer_idx"] = layer_idx
                row["sequence_length"] = sequence_length
                row["cache_has_previous_state"] = has_previous
                row["execution_phase"] = phase
                payload["gdn_calls"].append(row)
    return cache, code, runtime, payload


class FormalBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _verify_fixture_ledger()

    def test_archived_r29_closes_over_expanded_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, code, runtime, payload = _expanded_payload(root)
            receipt = root / "receipt.json"
            replay = root / "replay.json"
            controls = root / "controls.json"
            receipts._write_json(receipt, payload)
            receipts._write_json(
                replay,
                receipts.verify_payload(
                    payload,
                    cache_root=cache,
                    code_root=code,
                    runtime_root=runtime,
                ),
            )
            receipts.run_bound_negative_controls(
                receipt=receipt,
                cache_root=cache,
                code_root=code,
                runtime_root=runtime,
                output=controls,
            )
            aggregate = subject.verify_formal_binding(
                r29_result_path=R29_RESULT,
                semantic_sidecar=R29_SIDECAR,
                design=R29_DESIGN,
                receipt_path=receipt,
                replay_path=replay,
                controls_path=controls,
                cache_root=cache,
                code_root=code,
                runtime_root=runtime,
            )
            self.assertEqual(aggregate["status"], "pass")
            self.assertEqual(
                aggregate["r29_execution_binding"]["expected_attention_calls"], 120
            )
            self.assertEqual(
                aggregate["r29_execution_binding"]["expected_gdn_calls"], 540
            )
            self.assertEqual(
                aggregate["r29_execution_binding"][
                    "expected_document_prefill_gdn_calls"
                ],
                180,
            )
            self.assertEqual(
                aggregate["r29_execution_binding"]["expected_request_cell_gdn_calls"],
                360,
            )

    def test_missing_call_breaks_formal_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, code, runtime, payload = _expanded_payload(root)
            payload["attention_calls"].pop()
            receipt = root / "receipt.json"
            replay = root / "replay.json"
            controls = root / "controls.json"
            receipts._write_json(receipt, payload)
            receipts._write_json(
                replay,
                receipts.verify_payload(
                    payload,
                    cache_root=cache,
                    code_root=code,
                    runtime_root=runtime,
                ),
            )
            receipts.run_bound_negative_controls(
                receipt=receipt,
                cache_root=cache,
                code_root=code,
                runtime_root=runtime,
                output=controls,
            )
            with self.assertRaises(receipts.DispatchReceiptError):
                subject.verify_formal_binding(
                    r29_result_path=R29_RESULT,
                    semantic_sidecar=R29_SIDECAR,
                    design=R29_DESIGN,
                    receipt_path=receipt,
                    replay_path=replay,
                    controls_path=controls,
                    cache_root=cache,
                    code_root=code,
                    runtime_root=runtime,
                )

    def test_missing_gdn_call_breaks_phase_and_total_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, code, runtime, payload = _expanded_payload(root)
            payload["gdn_calls"].pop()
            receipt = root / "receipt.json"
            replay = root / "replay.json"
            controls = root / "controls.json"
            receipts._write_json(receipt, payload)
            receipts._write_json(
                replay,
                receipts.verify_payload(
                    payload,
                    cache_root=cache,
                    code_root=code,
                    runtime_root=runtime,
                ),
            )
            receipts.run_bound_negative_controls(
                receipt=receipt,
                cache_root=cache,
                code_root=code,
                runtime_root=runtime,
                output=controls,
            )
            with self.assertRaises(receipts.DispatchReceiptError):
                subject.verify_formal_binding(
                    r29_result_path=R29_RESULT,
                    semantic_sidecar=R29_SIDECAR,
                    design=R29_DESIGN,
                    receipt_path=receipt,
                    replay_path=replay,
                    controls_path=controls,
                    cache_root=cache,
                    code_root=code,
                    runtime_root=runtime,
                )


if __name__ == "__main__":
    unittest.main()
