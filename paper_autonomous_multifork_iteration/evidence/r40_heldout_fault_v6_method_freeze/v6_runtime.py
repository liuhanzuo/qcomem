from __future__ import annotations

import os
import signal
import stat
import threading
from pathlib import Path

from v6_guard import (
    HASH_KEYS,
    TERM_IDS,
    Reject,
    canonical_bytes,
    lifecycle_gate,
    measure_hashes,
    need,
    validate_hash_map,
)


class ProtectedParent:
    """No-replace publisher bound to one retained directory inode."""

    def __init__(self, path: os.PathLike[str] | str):
        self.path = Path(path).absolute()
        before = os.lstat(self.path)
        need(stat.S_ISDIR(before.st_mode) and not stat.S_ISLNK(before.st_mode), "parent directory")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        self.fd = os.open(self.path, flags)
        opened = os.fstat(self.fd)
        need((opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino), "parent open race")
        self.key = (opened.st_dev, opened.st_ino)
        self.closed = False

    def check(self) -> None:
        need(not self.closed, "closed parent")
        retained = os.fstat(self.fd)
        need((retained.st_dev, retained.st_ino) == self.key, "retained parent changed")
        try:
            named = os.stat(self.path, follow_symlinks=False)
        except OSError as exc:
            raise Reject("parent path absent") from exc
        need(stat.S_ISDIR(named.st_mode), "parent path not directory")
        need((named.st_dev, named.st_ino) == self.key, "parent path replaced")

    @staticmethod
    def _name(name: object) -> str:
        need(
            type(name) is str
            and bool(name)
            and "/" not in name
            and "\\" not in name
            and name not in (".", ".."),
            "publish name",
        )
        return name  # type: ignore[return-value]

    def _absent(self, name: str) -> None:
        try:
            os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise Reject("publish preflight") from exc
        raise Reject("no replace")

    def publish_many(self, outputs: dict[str, bytes], mode: int = 0o600) -> None:
        need(type(outputs) is dict and bool(outputs), "publish batch")
        need(type(mode) is int and mode in (0o444, 0o600), "publish mode")
        names = [self._name(name) for name in outputs]
        need(len(set(names)) == len(names), "duplicate publish name")
        for name in names:
            need(type(outputs[name]) is bytes, "publish bytes")
        self.check()
        for name in names:
            self._absent(name)

        token = f"{os.getpid()}.{threading.get_ident()}"
        staged: list[str] = []
        linked: list[str] = []
        try:
            for ordinal, name in enumerate(names):
                temporary = f".{name}.stage.{token}.{ordinal}"
                fd = os.open(
                    temporary,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    mode,
                    dir_fd=self.fd,
                )
                staged.append(temporary)
                try:
                    raw = outputs[name]
                    offset = 0
                    while offset < len(raw):
                        written = os.write(fd, raw[offset:])
                        need(written > 0, "publish short write")
                        offset += written
                    os.fsync(fd)
                finally:
                    os.close(fd)

            for name, temporary in zip(names, staged):
                try:
                    os.link(
                        temporary,
                        name,
                        src_dir_fd=self.fd,
                        dst_dir_fd=self.fd,
                        follow_symlinks=False,
                    )
                except FileExistsError as exc:
                    raise Reject("no replace") from exc
                linked.append(name)

            for temporary in list(staged):
                os.unlink(temporary, dir_fd=self.fd)
                staged.remove(temporary)
            os.fsync(self.fd)
            # A failure here must remove all names already linked into the retained
            # directory, even if the pathname now designates a replacement inode.
            self.check()
        except BaseException:
            for name in reversed(linked):
                try:
                    os.unlink(name, dir_fd=self.fd)
                except FileNotFoundError:
                    pass
            for temporary in list(staged):
                try:
                    os.unlink(temporary, dir_fd=self.fd)
                except FileNotFoundError:
                    pass
            try:
                os.fsync(self.fd)
            except OSError:
                pass
            raise

    def publish_json_many(self, outputs: dict[str, object]) -> None:
        self.publish_many({name: canonical_bytes(value) for name, value in outputs.items()})

    def close(self) -> None:
        if not self.closed:
            os.close(self.fd)
            self.closed = True

    def __enter__(self) -> "ProtectedParent":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def signal_exit(signum: object) -> int:
    need(type(signum) is int and signum in (signal.SIGINT, signal.SIGTERM), "signal")
    return 128 + signum  # type: ignore[operator]


class Lifecycle:
    def __init__(
        self,
        terminal_parent: ProtectedParent,
        archive_path: os.PathLike[str] | str,
        source_ledger_path: os.PathLike[str] | str,
        snapshot_root: os.PathLike[str] | str,
        runner_root: os.PathLike[str] | str,
        runner_manifest: object,
    ):
        self.parent = terminal_parent
        self.archive_path = Path(archive_path)
        self.source_ledger_path = Path(source_ledger_path)
        self.snapshot_root = Path(snapshot_root)
        self.runner_root = Path(runner_root)
        self.runner_manifest = runner_manifest
        self.started = False
        self.done = False
        self.handlers_installed = False
        self.pre_hashes: dict | None = None
        self.gate: dict | None = None
        self._old_handlers: dict[int, object] = {}
        self._handling_signal = False
        self._lock = threading.RLock()
        self.exit_code: int | None = None

    def install_signal_handlers(self) -> None:
        with self._lock:
            need(not self.handlers_installed and not self.started and not self.done, "handler install order")
            need(threading.current_thread() is threading.main_thread(), "signal handlers require main thread")
            for signum in (signal.SIGINT, signal.SIGTERM):
                self._old_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle_signal)
            self.handlers_installed = True

    def restore_signal_handlers(self) -> None:
        with self._lock:
            if self.handlers_installed:
                for signum, handler in self._old_handlers.items():
                    signal.signal(signum, handler)
                self._old_handlers.clear()
                self.handlers_installed = False

    def start(self) -> None:
        with self._lock:
            need(self.handlers_installed, "handlers not installed")
            need(not self.started and not self.done, "lifecycle start")
            self.pre_hashes = measure_hashes(
                self.archive_path,
                self.source_ledger_path,
                self.snapshot_root,
                self.runner_root,
                self.runner_manifest,
            )
            validate_hash_map(self.pre_hashes)
            self.started = True

    def set_gate(self, gate: object) -> None:
        with self._lock:
            need(self.started and not self.done and self.gate is None, "gate order")
            self.gate = lifecycle_gate(gate)

    def _post_hashes(self) -> tuple[dict, str | None]:
        try:
            measured = measure_hashes(
                self.archive_path,
                self.source_ledger_path,
                self.snapshot_root,
                self.runner_root,
                self.runner_manifest,
            )
            return validate_hash_map(measured), None
        except BaseException as exc:
            return {key: None for key in HASH_KEYS}, f"{type(exc).__name__}:{exc}"

    def finalize(self, reason: str, signum: int | None = None) -> int:
        with self._lock:
            if self.done:
                return self.exit_code if self.exit_code is not None else 1
            need(type(reason) is str and bool(reason), "finalize reason")
            need(signum is None or signum in (signal.SIGINT, signal.SIGTERM), "finalize signal")
            if self.pre_hashes is None:
                try:
                    self.pre_hashes = measure_hashes(
                        self.archive_path,
                        self.source_ledger_path,
                        self.snapshot_root,
                        self.runner_root,
                        self.runner_manifest,
                    )
                except BaseException:
                    self.pre_hashes = {key: None for key in HASH_KEYS}
            post_hashes, rehash_error = self._post_hashes()
            success = False
            if reason == "success" and signum is None and self.started and rehash_error is None:
                try:
                    lifecycle_gate(self.gate)
                    success = self.pre_hashes == post_hashes
                except BaseException:
                    success = False
            status = "success" if success else "failure"
            terminal_reason = "success" if success else reason
            if rehash_error is not None:
                terminal_reason = f"{reason};post-rehash={rehash_error}"
            outputs = {}
            for fault_id in TERM_IDS:
                outputs[f"{fault_id}.terminal.json"] = {
                    "fault_id": fault_id,
                    "post_hashes": post_hashes,
                    "pre_hashes": self.pre_hashes,
                    "reason": terminal_reason,
                    "schema_version": "forkaudit-v6-terminal-v1",
                    "signal": signum,
                    "status": status,
                }
            try:
                self.parent.publish_json_many(outputs)
            except BaseException:
                self.done = True
                self.exit_code = signal_exit(signum) if signum is not None else 1
                return self.exit_code
            self.done = True
            self.exit_code = (
                signal_exit(signum)
                if signum is not None
                else (0 if success else 1)
            )
            return self.exit_code

    def _handle_signal(self, signum: int, _frame: object) -> None:
        if self._handling_signal:
            os._exit(signal_exit(signum))
        self._handling_signal = True
        code = self.finalize("signal", signum)
        raise SystemExit(code)

