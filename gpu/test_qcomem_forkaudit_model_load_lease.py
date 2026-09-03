from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import qcomem_forkaudit_model_load_lease as lease


class FakeLeaseOps:
    f_rdlck = 0

    def __init__(self, *, fail_name_index=None, release_callback=None):
        self.states = {}
        self.acquire_calls = 0
        self.fail_name_index = fail_name_index
        self.release_callback = release_callback

    def acquire_read(self, fd):
        if self.fail_name_index == self.acquire_calls:
            self.acquire_calls += 1
            raise lease.ModelLoadLeaseError("existing writer")
        self.acquire_calls += 1
        self.states[fd] = self.f_rdlck

    def get(self, fd):
        return self.states.get(fd, 2)

    def release(self, fd):
        if self.release_callback is not None:
            self.release_callback()
        self.states[fd] = 2


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ModelLoadLeaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rows = []
        for index in range(1, 15):
            name = f"model.safetensors-{index:05d}-of-00014.safetensors"
            path = self.root / name
            path.write_bytes((f"weight-{index}-" * 7).encode())
            path.chmod(0o400)
            self.rows.append({"logical_name": name, "sha256": _sha(path)})
        self.kwargs = {
            "model_view": self.root,
            "ledger_rows": self.rows,
            "run_id": "12" * 16,
            "weight_ledger_raw_sha256": "a" * 64,
            "model_artifact_ledger_raw_sha256": "b" * 64,
            "model_view_manifest_sha256": "c" * 64,
            "install_sigio_handler": True,
        }

    def tearDown(self):
        self.temp.cleanup()

    def _acquire(self, ops=None):
        guard = lease.ModelLoadLeaseSet(
            **self.kwargs, lease_ops=ops if ops is not None else FakeLeaseOps()
        )
        authority = guard.acquire_and_hash()
        return guard, authority

    def test_authority_rank_envelopes_and_closure(self):
        guard, authority = self._acquire()
        authority_sha = lease.sha256_bytes(lease.canonical_receipt_bytes(authority))
        before = lease.capture_rank_stat_envelope(
            model_view=self.root,
            authority=authority,
            authority_raw_sha256=authority_sha,
            rank=3,
            capture_point="immediately-before-from-pretrained",
        )
        after = lease.capture_rank_stat_envelope(
            model_view=self.root,
            authority=authority,
            authority_raw_sha256=authority_sha,
            rank=3,
            capture_point="immediately-after-from-pretrained",
        )
        self.assertEqual(before["rows"], after["rows"])
        self.assertEqual(before["payload_bytes_read"], 0)
        closure = guard.close_and_receipt()
        self.assertTrue(closure["passed"])
        lease.validate_closure(closure, authority=authority)

    def test_canonical_authority_raw_sha_is_external(self):
        guard, authority = self._acquire()
        raw = lease.canonical_receipt_bytes(authority)
        parsed = lease.authority_from_canonical_bytes(raw, lease.sha256_bytes(raw))
        self.assertEqual(parsed, authority)
        with self.assertRaisesRegex(lease.ModelLoadLeaseError, "raw SHA drift"):
            lease.authority_from_canonical_bytes(raw, "f" * 64)
        guard.close_and_receipt()

    def test_existing_writer_or_unsupported_lease_fails_closed(self):
        guard = lease.ModelLoadLeaseSet(
            **self.kwargs, lease_ops=FakeLeaseOps(fail_name_index=0)
        )
        with self.assertRaisesRegex(lease.ModelLoadLeaseError, "existing writer"):
            guard.acquire_and_hash()

    def test_unsupported_filesystem_fails_closed(self):
        class UnsupportedLeaseOps(FakeLeaseOps):
            def acquire_read(self, fd):
                raise lease.ModelLoadLeaseError("unsupported filesystem")

        guard = lease.ModelLoadLeaseSet(
            **self.kwargs, lease_ops=UnsupportedLeaseOps()
        )
        with self.assertRaisesRegex(lease.ModelLoadLeaseError, "unsupported filesystem"):
            guard.acquire_and_hash()

    def test_symlink_weight_is_rejected(self):
        target = self.root / self.rows[0]["logical_name"]
        outside = self.root / "outside"
        outside.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(outside)
        guard = lease.ModelLoadLeaseSet(**self.kwargs, lease_ops=FakeLeaseOps())
        with self.assertRaisesRegex(lease.ModelLoadLeaseError, "opened safely"):
            guard.acquire_and_hash()

    def test_write_permission_bit_is_rejected(self):
        (self.root / self.rows[0]["logical_name"]).chmod(0o600)
        guard = lease.ModelLoadLeaseSet(**self.kwargs, lease_ops=FakeLeaseOps())
        with self.assertRaisesRegex(lease.ModelLoadLeaseError, "write permission"):
            guard.acquire_and_hash()

    def test_content_mismatch_is_rejected(self):
        path = self.root / self.rows[0]["logical_name"]
        path.chmod(0o600)
        path.write_bytes(b"same authority must not accept this")
        path.chmod(0o400)
        guard = lease.ModelLoadLeaseSet(**self.kwargs, lease_ops=FakeLeaseOps())
        with self.assertRaisesRegex(lease.ModelLoadLeaseError, "content drift"):
            guard.acquire_and_hash()

    def test_rank_stat_replacement_is_rejected_without_payload_read(self):
        guard, authority = self._acquire()
        authority_sha = lease.sha256_bytes(lease.canonical_receipt_bytes(authority))
        path = self.root / self.rows[0]["logical_name"]
        replacement = self.root / "replacement"
        replacement.write_bytes(path.read_bytes())
        replacement.chmod(0o400)
        os.replace(replacement, path)
        with mock.patch.object(lease.os, "read", side_effect=AssertionError("payload read")):
            with self.assertRaisesRegex(lease.ModelLoadLeaseError, "stat differs"):
                lease.capture_rank_stat_envelope(
                    model_view=self.root,
                    authority=authority,
                    authority_raw_sha256=authority_sha,
                    rank=0,
                    capture_point="immediately-before-from-pretrained",
                )
        # The fake lease cannot protect the path; closing honestly records invalidity.
        closure = guard.close_and_receipt()
        self.assertFalse(closure["passed"])

    def test_sticky_break_invalidates_closure(self):
        guard, authority = self._acquire()
        guard.mark_breach_for_test()
        closure = guard.close_and_receipt()
        self.assertFalse(closure["passed"])
        self.assertEqual(closure["lease_break_count"], 1)
        self.assertFalse(closure["all_leases_remained_read"])
        with self.assertRaisesRegex(lease.ModelLoadLeaseError, "did not pass"):
            lease.validate_closure(closure, authority=authority)
        lease.validate_closure(closure, authority=authority, require_passed=False)

    def test_sigio_write_attempt_is_sticky(self):
        signal_module = __import__("signal")
        guard, authority = self._acquire()
        signal_module.raise_signal(signal_module.SIGIO)
        closure = guard.close_and_receipt()
        self.assertFalse(closure["passed"])
        self.assertGreaterEqual(closure["lease_break_count"], 1)
        lease.validate_closure(closure, authority=authority, require_passed=False)

    def test_handler_disabled_cannot_issue_authority(self):
        kwargs = dict(self.kwargs)
        kwargs["install_sigio_handler"] = False
        guard = lease.ModelLoadLeaseSet(**kwargs, lease_ops=FakeLeaseOps())
        with self.assertRaisesRegex(lease.ModelLoadLeaseError, "not installed"):
            guard.acquire_and_hash()

    def test_break_during_unlock_is_in_closure(self):
        ops = FakeLeaseOps()
        guard, authority = self._acquire(ops)
        ops.release_callback = guard.mark_breach_for_test
        closure = guard.close_and_receipt()
        self.assertFalse(closure["passed"])
        self.assertEqual(closure["lease_break_count"], 14)
        self.assertFalse(closure["all_leases_remained_read"])
        lease.validate_closure(closure, authority=authority, require_passed=False)

    def test_terminal_hash_exception_always_releases_everything(self):
        ops = FakeLeaseOps()
        guard, _authority = self._acquire(ops)
        with mock.patch.object(lease, "_fd_sha256", side_effect=OSError("terminal read")):
            with self.assertRaisesRegex(OSError, "terminal read"):
                guard.close_and_receipt()
        self.assertTrue(guard._closed)
        self.assertEqual(guard._fds, {})
        self.assertTrue(all(state == 2 for state in ops.states.values()))

    def test_changed_sigio_handler_invalidates_closure(self):
        ops = FakeLeaseOps()
        kwargs = dict(self.kwargs)
        kwargs["install_sigio_handler"] = True
        guard = lease.ModelLoadLeaseSet(**kwargs, lease_ops=ops)
        authority = guard.acquire_and_hash()
        signal_handler = __import__("signal")
        signal_handler.signal(signal_handler.SIGIO, signal_handler.SIG_IGN)
        closure = guard.close_and_receipt()
        self.assertFalse(closure["passed"])
        self.assertIn("sigio_handler_changed", closure["invalid_reasons"])
        lease.validate_closure(closure, authority=authority, require_passed=False)

    def test_exact_weight_shard_set_is_required(self):
        wrong = [dict(row) for row in self.rows]
        wrong[0] = {
            "logical_name": "model.safetensors-00000-of-00014.safetensors",
            "sha256": wrong[0]["sha256"],
        }
        wrong = sorted(wrong, key=lambda row: row["logical_name"].encode())
        kwargs = dict(self.kwargs)
        kwargs["ledger_rows"] = wrong
        with self.assertRaisesRegex(lease.ModelLoadLeaseError, "exact 00001"):
            lease.ModelLoadLeaseSet(**kwargs, lease_ops=FakeLeaseOps())

    def test_closure_canonical_raw_sha_is_external(self):
        guard, authority = self._acquire()
        closure = guard.close_and_receipt()
        raw = lease.canonical_receipt_bytes(closure)
        parsed = lease.closure_from_canonical_bytes(
            raw,
            lease.sha256_bytes(raw),
            authority=authority,
        )
        self.assertEqual(parsed, closure)
        with self.assertRaisesRegex(lease.ModelLoadLeaseError, "raw SHA drift"):
            lease.closure_from_canonical_bytes(
                raw,
                "f" * 64,
                authority=authority,
            )

    def test_authority_and_rank_receipt_tamper_fail(self):
        guard, authority = self._acquire()
        tampered = dict(authority)
        tampered["all_read_only"] = False
        with self.assertRaises(lease.ModelLoadLeaseError):
            lease.validate_authority(tampered)
        authority_sha = lease.sha256_bytes(lease.canonical_receipt_bytes(authority))
        receipt = lease.capture_rank_stat_envelope(
            model_view=self.root,
            authority=authority,
            authority_raw_sha256=authority_sha,
            rank=0,
            capture_point="immediately-before-from-pretrained",
        )
        receipt["payload_bytes_read"] = 1
        with self.assertRaisesRegex(lease.ModelLoadLeaseError, "read weight payload"):
            lease.validate_rank_stat_envelope(receipt, authority=authority)
        guard.close_and_receipt()

    def test_rank_capture_never_calls_payload_read(self):
        guard, authority = self._acquire()
        authority_sha = lease.sha256_bytes(lease.canonical_receipt_bytes(authority))
        with mock.patch.object(lease.os, "read", side_effect=AssertionError("payload read")):
            receipt = lease.capture_rank_stat_envelope(
                model_view=self.root,
                authority=authority,
                authority_raw_sha256=authority_sha,
                rank=7,
                capture_point="immediately-after-from-pretrained",
            )
        self.assertEqual(receipt["payload_bytes_read"], 0)
        guard.close_and_receipt()


if __name__ == "__main__":
    unittest.main()
