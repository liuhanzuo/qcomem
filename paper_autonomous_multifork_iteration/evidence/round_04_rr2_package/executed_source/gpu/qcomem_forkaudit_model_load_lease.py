"""Linux model-load read-lease authority for ForkAudit formal runs.

This module deliberately separates the expensive, authoritative weight read
from rank-local metadata checks.  A launcher-owned guard opens the fourteen
private-view weight shards with ``O_NOFOLLOW``, obtains Linux read leases,
hashes the fixed file descriptors once, and retains both descriptors and
leases throughout model loading.  Rank processes only compare an exact stat
envelope immediately before and after ``from_pretrained``.  The guard then
rehashes the same descriptors and emits a closure receipt before releasing the
leases.

Linux file leases protect normal VFS write-open/truncate operations and leave
a sticky SIGIO breach signal if a lease is challenged.  They do not claim to
protect against root/CAP_SYS_ADMIN, raw block-device writes, a malicious
kernel or filesystem server, or compromise of the guard process itself.
Unsupported filesystems fail closed; callers must not silently downgrade to
``flock``, POSIX record locks, chmod, or stat-only checking.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import stat
import sys
import threading
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


AUTHORITY_SCHEMA_VERSION = "qcomem-forkaudit-model-load-authority-v1"
CLOSURE_SCHEMA_VERSION = "qcomem-forkaudit-model-load-closure-v1"
RANK_STAT_SCHEMA_VERSION = "qcomem-forkaudit-model-load-rank-stat-v1"
THREAT_MODEL = (
    "normal-vfs-no-root-no-cap-lease-no-raw-device-no-malicious-fs-"
    "no-active-same-uid-pathname-adversary"
)
WEIGHT_COUNT = 14
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RUN_ID_RE = re.compile(r"[0-9a-f]{32}")
_WEIGHT_NAME_RE = re.compile(
    r"model\.safetensors-[0-9]{5}-of-00014\.safetensors"
)


class ModelLoadLeaseError(RuntimeError):
    """Raised when the formal model-load authority fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ModelLoadLeaseError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        "%s must be lowercase SHA-256" % label,
    )
    return value


def _require_run_id(value: Any) -> str:
    _require(
        isinstance(value, str) and _RUN_ID_RE.fullmatch(value) is not None,
        "run_id must be 32 lowercase hexadecimal characters",
    )
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def canonical_receipt_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _normalized_name(value: Any) -> str:
    _require(isinstance(value, str) and bool(value), "logical name must be nonempty")
    path = PurePosixPath(value)
    _require(
        not path.is_absolute()
        and ".." not in path.parts
        and "." not in path.parts
        and path.as_posix() == value
        and "\\" not in value
        and "\x00" not in value,
        "logical name is not normalized",
    )
    return value


def _fd_sha256(fd: int, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        payload = os.read(fd, chunk_bytes)
        if not payload:
            break
        digest.update(payload)
    os.lseek(fd, 0, os.SEEK_SET)
    return digest.hexdigest()


def _stat_identity(st: os.stat_result) -> Dict[str, int]:
    return {
        "bytes": int(st.st_size),
        "st_dev": int(st.st_dev),
        "st_ino": int(st.st_ino),
        "mode": int(stat.S_IMODE(st.st_mode)),
        "mtime_ns": int(st.st_mtime_ns),
        "ctime_ns": int(st.st_ctime_ns),
    }


def _process_thread_count() -> int:
    """Count Linux tasks; fall back only for non-production unit platforms."""

    task_root = Path("/proc/self/task")
    if sys.platform.startswith("linux"):
        try:
            return len(list(task_root.iterdir()))
        except OSError as exc:  # pragma: no cover - formal Linux failure path.
            raise ModelLoadLeaseError("cannot enumerate Linux process tasks") from exc
    return int(threading.active_count())


def _validate_stat_identity(value: Any, *, label: str) -> Dict[str, int]:
    fields = {"bytes", "st_dev", "st_ino", "mode", "mtime_ns", "ctime_ns"}
    _require(isinstance(value, dict) and set(value) == fields, "%s schema drift" % label)
    for field in fields:
        _require(_is_int(value[field]) and value[field] >= 0, "%s %s drift" % (label, field))
    _require(value["bytes"] > 0, "%s file is empty" % label)
    _require(value["st_dev"] > 0 and value["st_ino"] > 0, "%s identity is invalid" % label)
    _require(value["mode"] & 0o222 == 0, "%s has a write permission bit" % label)
    return dict(value)


class LinuxLeaseOps:
    """Small injectable wrapper around Linux ``F_SETLEASE``/``F_GETLEASE``."""

    def __init__(self) -> None:
        try:
            import fcntl
        except ImportError as exc:  # pragma: no cover - Linux formal path only.
            raise ModelLoadLeaseError("fcntl is unavailable") from exc
        _require(sys.platform.startswith("linux"), "Linux file leases are required")
        self._fcntl = fcntl
        self.f_setlease = getattr(fcntl, "F_SETLEASE", 1024)
        self.f_getlease = getattr(fcntl, "F_GETLEASE", 1025)
        self.f_setown = getattr(fcntl, "F_SETOWN", 8)
        self.f_rdlck = getattr(fcntl, "F_RDLCK", 0)
        self.f_unlck = getattr(fcntl, "F_UNLCK", 2)

    def acquire_read(self, fd: int) -> None:
        try:
            self._fcntl.fcntl(fd, self.f_setown, os.getpid())
            observed = self._fcntl.fcntl(fd, self.f_setlease, self.f_rdlck)
        except OSError as exc:
            raise ModelLoadLeaseError(
                "Linux read lease acquisition failed; filesystem/writer is unsupported"
            ) from exc
        _require(observed == 0, "F_SETLEASE returned an unexpected value")

    def get(self, fd: int) -> int:
        try:
            return int(self._fcntl.fcntl(fd, self.f_getlease))
        except OSError as exc:
            raise ModelLoadLeaseError("F_GETLEASE failed") from exc

    def release(self, fd: int) -> None:
        try:
            self._fcntl.fcntl(fd, self.f_setlease, self.f_unlck)
        except OSError as exc:
            raise ModelLoadLeaseError("F_SETLEASE unlock failed") from exc


class ModelLoadLeaseSet:
    """Own fixed weight FDs and Linux read leases for one formal run."""

    def __init__(
        self,
        *,
        model_view: Path,
        ledger_rows: Sequence[Mapping[str, str]],
        run_id: str,
        weight_ledger_raw_sha256: str,
        model_artifact_ledger_raw_sha256: str,
        model_view_manifest_sha256: str,
        lease_ops: Optional[Any] = None,
        install_sigio_handler: bool = True,
    ) -> None:
        self.model_view = Path(model_view)
        self.run_id = _require_run_id(run_id)
        self.weight_ledger_raw_sha256 = _require_sha256(
            weight_ledger_raw_sha256, "weight ledger raw SHA"
        )
        self.model_artifact_ledger_raw_sha256 = _require_sha256(
            model_artifact_ledger_raw_sha256, "model artifact ledger raw SHA"
        )
        self.model_view_manifest_sha256 = _require_sha256(
            model_view_manifest_sha256, "model-view manifest SHA"
        )
        _require(len(ledger_rows) == WEIGHT_COUNT, "weight ledger must contain 14 rows")
        normalized: List[Dict[str, str]] = []
        names: List[str] = []
        for row in ledger_rows:
            _require(
                isinstance(row, Mapping) and set(row) == {"logical_name", "sha256"},
                "weight ledger row schema drift",
            )
            name = _normalized_name(row["logical_name"])
            _require(_WEIGHT_NAME_RE.fullmatch(name) is not None, "unexpected weight name")
            names.append(name)
            normalized.append(
                {"logical_name": name, "sha256": _require_sha256(row["sha256"], name)}
            )
        _require(len(set(names)) == WEIGHT_COUNT, "weight ledger names are not unique")
        _require(
            names
            == [
                "model.safetensors-%05d-of-00014.safetensors" % index
                for index in range(1, WEIGHT_COUNT + 1)
            ],
            "weight ledger does not contain the exact 00001..00014 shard set",
        )
        _require(
            names == sorted(names, key=lambda item: item.encode("utf-8")),
            "weight ledger is not C-sorted",
        )
        self.ledger_rows = normalized
        self.lease_ops = lease_ops if lease_ops is not None else LinuxLeaseOps()
        self._fds: Dict[str, int] = {}
        self._authority: Optional[Dict[str, Any]] = None
        self._break_count = 0
        self._previous_sigio_handler: Any = None
        self._sigio_handler: Any = None
        self._handler_installed = False
        self._closed = False
        if install_sigio_handler:
            self._previous_sigio_handler = signal.getsignal(signal.SIGIO)
            self._sigio_handler = self._handle_sigio
            signal.signal(signal.SIGIO, self._sigio_handler)
            self._handler_installed = True

    def _handle_sigio(self, _signum: int, _frame: Any) -> None:
        self._break_count += 1

    def mark_breach_for_test(self) -> None:
        """Inject a sticky break in unit tests without a real Linux lease."""

        self._break_count += 1

    def acquire_and_hash(self) -> Dict[str, Any]:
        _require(not self._closed and not self._fds, "lease set is not fresh")
        _require(self._handler_installed, "formal SIGIO handler is not installed")
        process_thread_count = _process_thread_count()
        _require(
            process_thread_count == 1,
            "model-load keeper must be a single-threaded process",
        )
        _require(self.model_view.is_dir(), "private model view is missing")
        open_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        nofollow = getattr(os, "O_NOFOLLOW", None)
        _require(nofollow is not None, "O_NOFOLLOW is required")
        open_flags |= nofollow
        rows: List[Dict[str, Any]] = []
        try:
            for ledger_row in self.ledger_rows:
                name = ledger_row["logical_name"]
                path = self.model_view.joinpath(*PurePosixPath(name).parts)
                try:
                    fd = os.open(str(path), open_flags)
                except OSError as exc:
                    raise ModelLoadLeaseError("weight cannot be opened safely: %s" % name) from exc
                self._fds[name] = fd
                st = os.fstat(fd)
                _require(stat.S_ISREG(st.st_mode), "weight is not regular: %s" % name)
                identity = _stat_identity(st)
                _validate_stat_identity(identity, label=name)
                self.lease_ops.acquire_read(fd)
                _require(
                    self.lease_ops.get(fd) == self.lease_ops.f_rdlck,
                    "read lease is not active: %s" % name,
                )
                observed = _fd_sha256(fd)
                _require(observed == ledger_row["sha256"], "weight content drift: %s" % name)
                rows.append(
                    {
                        "logical_name": name,
                        "declared_sha256": ledger_row["sha256"],
                        "observed_sha256": observed,
                        "stat": identity,
                        "lease_state": "read",
                    }
                )
            _require(self._break_count == 0, "lease was challenged while authority was built")
        except BaseException:
            self._release_all(best_effort=True)
            raise
        authority = {
            "schema_version": AUTHORITY_SCHEMA_VERSION,
            "run_id": self.run_id,
            "threat_model": THREAT_MODEL,
            "view_policy": "private-independent-inode-no-symlink-reflink-or-copy",
            "weight_ledger_raw_sha256": self.weight_ledger_raw_sha256,
            "model_artifact_ledger_raw_sha256": self.model_artifact_ledger_raw_sha256,
            "model_view_manifest_sha256": self.model_view_manifest_sha256,
            "entry_count": WEIGHT_COUNT,
            "rows": rows,
            "rows_sha256": sha256_json(rows),
            "all_content_matches_ledger": True,
            "all_regular_no_symlink": True,
            "all_read_only": True,
            "all_linux_read_leases_active": True,
            "sigio_handler_installed": True,
            "process_thread_count_at_authority": process_thread_count,
            "lease_break_count_at_authority": 0,
        }
        validate_authority(authority)
        self._authority = authority
        return authority

    @property
    def authority(self) -> Dict[str, Any]:
        _require(self._authority is not None, "authority has not been built")
        return dict(self._authority)

    def close_and_receipt(self) -> Dict[str, Any]:
        _require(not self._closed, "lease set is already closed")
        _require(self._authority is not None, "authority has not been built")
        authority = self._authority
        authority_raw_sha256 = sha256_bytes(canonical_receipt_bytes(authority))
        rows: List[Dict[str, Any]] = []
        invalid_reasons: List[str] = []
        previous_mask: Any = None
        signal_barrier = False
        pending_sigio = False
        handler_unchanged = True
        lease_release_errors = 0
        fd_close_errors = 0
        try:
            if self._handler_installed:
                _require(
                    hasattr(signal, "pthread_sigmask")
                    and hasattr(signal, "sigpending"),
                    "pthread SIGIO barrier is required for terminal closure",
                )
                handler_unchanged = (
                    signal.getsignal(signal.SIGIO) is self._sigio_handler
                )
                previous_mask = signal.pthread_sigmask(
                    signal.SIG_BLOCK, {signal.SIGIO}
                )
                signal_barrier = True
                pending_sigio = signal.SIGIO in signal.sigpending()
            for authority_row in authority["rows"]:
                name = authority_row["logical_name"]
                fd = self._fds[name]
                try:
                    lease_state = self.lease_ops.get(fd)
                except ModelLoadLeaseError:
                    lease_state = -1
                identity = _stat_identity(os.fstat(fd))
                observed = _fd_sha256(fd)
                if lease_state != self.lease_ops.f_rdlck:
                    invalid_reasons.append("lease_not_read:%s" % name)
                if identity != authority_row["stat"]:
                    invalid_reasons.append("stat_changed:%s" % name)
                if observed != authority_row["observed_sha256"]:
                    invalid_reasons.append("content_changed:%s" % name)
                rows.append(
                    {
                        "logical_name": name,
                        "observed_sha256": observed,
                        "stat": identity,
                        "lease_state": (
                            "read"
                            if lease_state == self.lease_ops.f_rdlck
                            else "broken"
                        ),
                    }
                )
            lease_release_errors, fd_close_errors = self._release_fd_resources()
            if signal_barrier:
                pending_sigio = pending_sigio or signal.SIGIO in signal.sigpending()
        except BaseException:
            # Terminal I/O failures invalidate the run but must never strand a
            # fixed descriptor, lease, or process-global signal handler.
            self._release_fd_resources()
            self._finish_signal_barrier(previous_mask, signal_barrier)
            self._closed = True
            raise
        self._finish_signal_barrier(previous_mask, signal_barrier)
        self._closed = True

        final_break_count = int(self._break_count + (1 if pending_sigio else 0))
        if final_break_count:
            invalid_reasons.append("sticky_sigio_or_pending_lease_break")
        if not handler_unchanged:
            invalid_reasons.append("sigio_handler_changed")
        if lease_release_errors:
            invalid_reasons.append("lease_release_error")
        if fd_close_errors:
            invalid_reasons.append("fd_close_error")
        all_read = bool(
            final_break_count == 0
            and all(row["lease_state"] == "read" for row in rows)
        )
        all_stats = all(
            row["stat"] == authority_row["stat"]
            for row, authority_row in zip(rows, authority["rows"])
        )
        all_content = all(
            row["observed_sha256"] == authority_row["observed_sha256"]
            for row, authority_row in zip(rows, authority["rows"])
        )
        closure = {
            "schema_version": CLOSURE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "threat_model": THREAT_MODEL,
            "authority_raw_sha256": authority_raw_sha256,
            "entry_count": WEIGHT_COUNT,
            "rows": rows,
            "rows_sha256": sha256_json(rows),
            "lease_break_count": final_break_count,
            "sigio_pending_during_terminal": bool(pending_sigio),
            "sigio_handler_unchanged_until_release": bool(handler_unchanged),
            "all_leases_remained_read": all_read,
            "all_final_stats_equal_authority": all_stats,
            "all_final_content_equal_authority": all_content,
            "all_leases_released": lease_release_errors == 0,
            "all_fds_closed": fd_close_errors == 0,
            "lease_release_error_count": int(lease_release_errors),
            "fd_close_error_count": int(fd_close_errors),
            "terminal_full_content_rehash_performed": True,
            "invalid_reasons": sorted(set(invalid_reasons)),
            "passed": not invalid_reasons,
        }
        validate_closure(closure, authority=authority, require_passed=False)
        return closure

    def _release_fd_resources(self) -> tuple[int, int]:
        lease_errors = 0
        close_errors = 0
        for fd in list(self._fds.values()):
            try:
                self.lease_ops.release(fd)
            except BaseException:  # pragma: no cover - rare kernel failure.
                lease_errors += 1
            try:
                os.close(fd)
            except BaseException:  # pragma: no cover - rare kernel failure.
                close_errors += 1
        self._fds.clear()
        return lease_errors, close_errors

    def _finish_signal_barrier(self, previous_mask: Any, active: bool) -> None:
        if not self._handler_installed:
            return
        if active:
            # A pending guard-owned SIGIO is already recorded.  Discard it
            # instead of delivering it to the caller's previous handler.
            signal.signal(signal.SIGIO, signal.SIG_IGN)
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        signal.signal(signal.SIGIO, self._previous_sigio_handler)
        self._handler_installed = False

    def _release_all(self, *, best_effort: bool) -> None:
        lease_errors, close_errors = self._release_fd_resources()
        if self._handler_installed:
            signal.signal(signal.SIGIO, self._previous_sigio_handler)
            self._handler_installed = False
        self._closed = True
        if (lease_errors or close_errors) and not best_effort:
            raise ModelLoadLeaseError("one or more model-load leases failed to release")


def _validate_authority_row(value: Any) -> Dict[str, Any]:
    fields = {"logical_name", "declared_sha256", "observed_sha256", "stat", "lease_state"}
    _require(isinstance(value, dict) and set(value) == fields, "authority row schema drift")
    name = _normalized_name(value["logical_name"])
    _require(_WEIGHT_NAME_RE.fullmatch(name) is not None, "authority weight name drift")
    declared = _require_sha256(value["declared_sha256"], name)
    observed = _require_sha256(value["observed_sha256"], name)
    _require(declared == observed, "authority observed digest differs from ledger")
    _require(value["lease_state"] == "read", "authority lease is not read")
    return {
        "logical_name": name,
        "declared_sha256": declared,
        "observed_sha256": observed,
        "stat": _validate_stat_identity(value["stat"], label=name),
        "lease_state": "read",
    }


def validate_authority(value: Any) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "run_id",
        "threat_model",
        "view_policy",
        "weight_ledger_raw_sha256",
        "model_artifact_ledger_raw_sha256",
        "model_view_manifest_sha256",
        "entry_count",
        "rows",
        "rows_sha256",
        "all_content_matches_ledger",
        "all_regular_no_symlink",
        "all_read_only",
        "all_linux_read_leases_active",
        "sigio_handler_installed",
        "process_thread_count_at_authority",
        "lease_break_count_at_authority",
    }
    _require(isinstance(value, dict) and set(value) == fields, "authority schema drift")
    _require(value["schema_version"] == AUTHORITY_SCHEMA_VERSION, "authority version drift")
    _require_run_id(value["run_id"])
    _require(value["threat_model"] == THREAT_MODEL, "authority threat model drift")
    _require(
        value["view_policy"] == "private-independent-inode-no-symlink-reflink-or-copy",
        "authority view policy drift",
    )
    for field in (
        "weight_ledger_raw_sha256",
        "model_artifact_ledger_raw_sha256",
        "model_view_manifest_sha256",
        "rows_sha256",
    ):
        _require_sha256(value[field], field)
    _require(_is_int(value["entry_count"]) and value["entry_count"] == WEIGHT_COUNT, "authority count drift")
    _require(isinstance(value["rows"], list) and len(value["rows"]) == WEIGHT_COUNT, "authority rows drift")
    rows = [_validate_authority_row(row) for row in value["rows"]]
    names = [row["logical_name"] for row in rows]
    _require(len(set(names)) == WEIGHT_COUNT, "authority names are not unique")
    _require(
        names
        == [
            "model.safetensors-%05d-of-00014.safetensors" % index
            for index in range(1, WEIGHT_COUNT + 1)
        ],
        "authority does not contain the exact 00001..00014 shard set",
    )
    _require(names == sorted(names, key=lambda item: item.encode("utf-8")), "authority rows are not C-sorted")
    _require(sha256_json(rows) == value["rows_sha256"], "authority rows SHA drift")
    for field in (
        "all_content_matches_ledger",
        "all_regular_no_symlink",
        "all_read_only",
        "all_linux_read_leases_active",
        "sigio_handler_installed",
    ):
        _require(value[field] is True, "authority %s is not true" % field)
    _require(
        _is_int(value["process_thread_count_at_authority"])
        and value["process_thread_count_at_authority"] == 1,
        "authority process was not single-threaded",
    )
    _require(
        _is_int(value["lease_break_count_at_authority"])
        and value["lease_break_count_at_authority"] == 0,
        "authority has a lease break",
    )
    return dict(value)


def authority_from_canonical_bytes(payload: bytes, expected_sha256: str) -> Dict[str, Any]:
    _require(sha256_bytes(payload) == _require_sha256(expected_sha256, "authority raw SHA"), "authority raw SHA drift")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelLoadLeaseError("authority is not strict UTF-8 JSON") from exc
    _require(canonical_receipt_bytes(value) == payload, "authority is not canonical JSON plus LF")
    return validate_authority(value)


def closure_from_canonical_bytes(
    payload: bytes,
    expected_sha256: str,
    *,
    authority: Mapping[str, Any],
    require_passed: bool = True,
) -> Dict[str, Any]:
    _require(
        sha256_bytes(payload) == _require_sha256(expected_sha256, "closure raw SHA"),
        "closure raw SHA drift",
    )
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelLoadLeaseError("closure is not strict UTF-8 JSON") from exc
    _require(
        canonical_receipt_bytes(value) == payload,
        "closure is not canonical JSON plus LF",
    )
    return validate_closure(
        value, authority=authority, require_passed=require_passed
    )


def capture_rank_stat_envelope(
    *, model_view: Path, authority: Mapping[str, Any], authority_raw_sha256: str,
    rank: int, capture_point: str
) -> Dict[str, Any]:
    validated = validate_authority(dict(authority))
    _require(_is_int(rank) and 0 <= rank < 8, "rank stat rank drift")
    _require(
        capture_point in ("immediately-before-from-pretrained", "immediately-after-from-pretrained"),
        "rank stat capture point drift",
    )
    expected_authority_sha = sha256_bytes(canonical_receipt_bytes(validated))
    _require(expected_authority_sha == _require_sha256(authority_raw_sha256, "authority raw SHA"), "rank authority SHA drift")
    open_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    _require(nofollow is not None, "O_NOFOLLOW is required")
    rows: List[Dict[str, Any]] = []
    for expected in validated["rows"]:
        path = Path(model_view).joinpath(*PurePosixPath(expected["logical_name"]).parts)
        try:
            fd = os.open(str(path), open_flags | nofollow)
        except OSError as exc:
            raise ModelLoadLeaseError("rank cannot open fixed weight path") from exc
        try:
            observed = _stat_identity(os.fstat(fd))
        finally:
            os.close(fd)
        _require(observed == expected["stat"], "rank weight stat differs from authority")
        rows.append({"logical_name": expected["logical_name"], "stat": observed})
    receipt = {
        "schema_version": RANK_STAT_SCHEMA_VERSION,
        "run_id": validated["run_id"],
        "rank": rank,
        "capture_point": capture_point,
        "authority_raw_sha256": expected_authority_sha,
        "payload_bytes_read": 0,
        "rows": rows,
        "rows_sha256": sha256_json(rows),
        "all_stats_equal_authority": True,
    }
    return validate_rank_stat_envelope(receipt, authority=validated)


def validate_rank_stat_envelope(value: Any, *, authority: Mapping[str, Any]) -> Dict[str, Any]:
    validated = validate_authority(dict(authority))
    fields = {
        "schema_version", "run_id", "rank", "capture_point", "authority_raw_sha256",
        "payload_bytes_read", "rows", "rows_sha256", "all_stats_equal_authority",
    }
    _require(isinstance(value, dict) and set(value) == fields, "rank stat schema drift")
    _require(value["schema_version"] == RANK_STAT_SCHEMA_VERSION, "rank stat version drift")
    _require(value["run_id"] == validated["run_id"], "rank stat run ID drift")
    _require(_is_int(value["rank"]) and 0 <= value["rank"] < 8, "rank stat rank drift")
    _require(
        value["capture_point"]
        in ("immediately-before-from-pretrained", "immediately-after-from-pretrained"),
        "rank stat capture point drift",
    )
    _require(
        value["authority_raw_sha256"] == sha256_bytes(canonical_receipt_bytes(validated)),
        "rank stat authority SHA drift",
    )
    _require(_is_int(value["payload_bytes_read"]) and value["payload_bytes_read"] == 0, "rank read weight payload")
    _require(isinstance(value["rows"], list) and len(value["rows"]) == WEIGHT_COUNT, "rank stat rows drift")
    expected_rows = [
        {"logical_name": row["logical_name"], "stat": row["stat"]}
        for row in validated["rows"]
    ]
    _require(value["rows"] == expected_rows, "rank stat rows differ from authority")
    _require(value["rows_sha256"] == sha256_json(expected_rows), "rank stat rows SHA drift")
    _require(value["all_stats_equal_authority"] is True, "rank stat equality is false")
    return dict(value)


def validate_closure(
    value: Any, *, authority: Mapping[str, Any], require_passed: bool = True
) -> Dict[str, Any]:
    validated = validate_authority(dict(authority))
    fields = {
        "schema_version", "run_id", "threat_model", "authority_raw_sha256",
        "entry_count", "rows", "rows_sha256", "lease_break_count",
        "sigio_pending_during_terminal",
        "sigio_handler_unchanged_until_release",
        "all_leases_remained_read", "all_final_stats_equal_authority",
        "all_final_content_equal_authority", "all_leases_released",
        "all_fds_closed", "lease_release_error_count", "fd_close_error_count",
        "terminal_full_content_rehash_performed",
        "invalid_reasons", "passed",
    }
    _require(isinstance(value, dict) and set(value) == fields, "closure schema drift")
    _require(value["schema_version"] == CLOSURE_SCHEMA_VERSION, "closure version drift")
    _require(value["run_id"] == validated["run_id"], "closure run ID drift")
    _require(value["threat_model"] == THREAT_MODEL, "closure threat model drift")
    _require(
        value["authority_raw_sha256"] == sha256_bytes(canonical_receipt_bytes(validated)),
        "closure authority SHA drift",
    )
    _require(_is_int(value["entry_count"]) and value["entry_count"] == WEIGHT_COUNT, "closure count drift")
    _require(isinstance(value["rows"], list) and len(value["rows"]) == WEIGHT_COUNT, "closure rows drift")
    normalized_rows: List[Dict[str, Any]] = []
    for row, expected in zip(value["rows"], validated["rows"]):
        _require(
            isinstance(row, dict)
            and set(row) == {"logical_name", "observed_sha256", "stat", "lease_state"}
            and row["logical_name"] == expected["logical_name"],
            "closure row schema/order drift",
        )
        _require_sha256(row["observed_sha256"], "closure observed SHA")
        normalized_rows.append(
            {
                "logical_name": row["logical_name"],
                "observed_sha256": row["observed_sha256"],
                "stat": _validate_stat_identity(row["stat"], label=row["logical_name"]),
                "lease_state": row["lease_state"],
            }
        )
    _require(value["rows_sha256"] == sha256_json(normalized_rows), "closure rows SHA drift")
    _require(_is_int(value["lease_break_count"]) and value["lease_break_count"] >= 0, "closure break count drift")
    _require(
        isinstance(value["sigio_pending_during_terminal"], bool)
        and isinstance(value["sigio_handler_unchanged_until_release"], bool),
        "closure SIGIO fields drift",
    )
    _require(
        _is_int(value["lease_release_error_count"])
        and value["lease_release_error_count"] >= 0
        and _is_int(value["fd_close_error_count"])
        and value["fd_close_error_count"] >= 0,
        "closure release counts drift",
    )
    _require(isinstance(value["invalid_reasons"], list) and all(isinstance(x, str) for x in value["invalid_reasons"]), "closure invalid reasons drift")
    all_read = bool(
        value["lease_break_count"] == 0
        and all(row["lease_state"] == "read" for row in normalized_rows)
    )
    all_stats = all(
        row["stat"] == expected["stat"]
        for row, expected in zip(normalized_rows, validated["rows"])
    )
    all_content = all(
        row["observed_sha256"] == expected["observed_sha256"]
        for row, expected in zip(normalized_rows, validated["rows"])
    )
    expected_invalid_reasons: List[str] = []
    for row, expected in zip(normalized_rows, validated["rows"]):
        name = row["logical_name"]
        if row["lease_state"] != "read":
            expected_invalid_reasons.append("lease_not_read:%s" % name)
        if row["stat"] != expected["stat"]:
            expected_invalid_reasons.append("stat_changed:%s" % name)
        if row["observed_sha256"] != expected["observed_sha256"]:
            expected_invalid_reasons.append("content_changed:%s" % name)
    if value["lease_break_count"]:
        expected_invalid_reasons.append("sticky_sigio_or_pending_lease_break")
    if not value["sigio_handler_unchanged_until_release"]:
        expected_invalid_reasons.append("sigio_handler_changed")
    if value["lease_release_error_count"]:
        expected_invalid_reasons.append("lease_release_error")
    if value["fd_close_error_count"]:
        expected_invalid_reasons.append("fd_close_error")
    expected_invalid_reasons = sorted(set(expected_invalid_reasons))
    _require(
        value["invalid_reasons"] == expected_invalid_reasons,
        "closure invalid reasons do not replay",
    )
    expected_passed = bool(
        all_read
        and all_stats
        and all_content
        and value["sigio_pending_during_terminal"] is False
        and value["sigio_handler_unchanged_until_release"] is True
        and value["lease_release_error_count"] == 0
        and value["fd_close_error_count"] == 0
        and not value["invalid_reasons"]
    )
    _require(value["all_leases_remained_read"] is all_read, "closure lease summary drift")
    _require(value["all_final_stats_equal_authority"] is all_stats, "closure stat summary drift")
    _require(value["all_final_content_equal_authority"] is all_content, "closure content summary drift")
    _require(value["all_leases_released"] is (value["lease_release_error_count"] == 0), "closure release summary drift")
    _require(value["all_fds_closed"] is (value["fd_close_error_count"] == 0), "closure FD summary drift")
    _require(value["terminal_full_content_rehash_performed"] is True, "closure omitted terminal rehash")
    _require(value["passed"] is expected_passed, "closure passed summary drift")
    if require_passed:
        _require(expected_passed, "model-load closure did not pass")
    return dict(value)


__all__ = [
    "AUTHORITY_SCHEMA_VERSION", "CLOSURE_SCHEMA_VERSION", "RANK_STAT_SCHEMA_VERSION",
    "THREAT_MODEL", "WEIGHT_COUNT", "LinuxLeaseOps", "ModelLoadLeaseError",
    "ModelLoadLeaseSet", "authority_from_canonical_bytes", "closure_from_canonical_bytes", "canonical_json_bytes",
    "canonical_receipt_bytes", "capture_rank_stat_envelope", "sha256_bytes",
    "sha256_json", "validate_authority", "validate_closure",
    "validate_rank_stat_envelope",
]
