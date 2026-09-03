from __future__ import annotations

import copy
import os
import signal
import stat
import subprocess
import threading
import time
from pathlib import Path

from v8_guard import (
    HASH_KEYS,
    SPAWN_PROVENANCE_KEYS,
    TERM_IDS,
    Reject,
    _read_regular,
    canonical_bytes,
    derive_spawned_specs,
    digest_bytes,
    digest_file,
    isolated_torch_probe,
    lifecycle_gate,
    measure_hashes,
    need,
    validate_execution_contract,
    validate_hash_map,
    validate_runtime_expectation,
    verify_operator_binding,
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


_AUTHORIZED_PLAN_TOKEN = object()


class AuthorizedPlan:
    """Opaque result of the complete signed pre-spawn authorization chain."""

    __slots__ = (
        "_artifact_hashes",
        "_binding_path",
        "_contract_path",
        "_expectation_path",
        "_provenance",
        "_runner_manifest_path",
        "_runner_root",
        "_spawned_specs",
        "_timeout_seconds",
    )

    def __init__(
        self,
        token: object,
        *,
        artifact_hashes: dict,
        binding_path: Path,
        contract_path: Path,
        expectation_path: Path,
        provenance: dict,
        runner_manifest_path: Path,
        runner_root: Path,
        spawned_specs: dict,
        timeout_seconds: int,
    ):
        need(token is _AUTHORIZED_PLAN_TOKEN, "authorized plan construction")
        need(set(provenance) == set(SPAWN_PROVENANCE_KEYS), "authorized provenance schema")
        object.__setattr__(self, "_artifact_hashes", copy.deepcopy(validate_hash_map(artifact_hashes)))
        object.__setattr__(self, "_binding_path", binding_path)
        object.__setattr__(self, "_contract_path", contract_path)
        object.__setattr__(self, "_expectation_path", expectation_path)
        object.__setattr__(self, "_provenance", copy.deepcopy(provenance))
        object.__setattr__(self, "_runner_manifest_path", runner_manifest_path)
        object.__setattr__(self, "_runner_root", runner_root)
        object.__setattr__(self, "_spawned_specs", copy.deepcopy(spawned_specs))
        object.__setattr__(self, "_timeout_seconds", timeout_seconds)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise Reject("authorized plan is immutable")

    @property
    def artifact_hashes(self) -> dict:
        return copy.deepcopy(self._artifact_hashes)

    @property
    def binding_path(self) -> Path:
        return self._binding_path

    @property
    def contract_path(self) -> Path:
        return self._contract_path

    @property
    def expectation_path(self) -> Path:
        return self._expectation_path

    @property
    def provenance(self) -> dict:
        return copy.deepcopy(self._provenance)

    @property
    def runner_manifest_path(self) -> Path:
        return self._runner_manifest_path

    @property
    def runner_root(self) -> Path:
        return self._runner_root

    @property
    def spawned_specs(self) -> dict:
        return copy.deepcopy(self._spawned_specs)

    @property
    def timeout_seconds(self) -> int:
        return self._timeout_seconds


def prepare_authorized_plan(
    binding_path: os.PathLike[str] | str,
    archive_path: os.PathLike[str] | str,
    source_ledger_path: os.PathLike[str] | str,
    snapshot_root: os.PathLike[str] | str,
    execution_contract_path: os.PathLike[str] | str,
    runtime_expectation_path: os.PathLike[str] | str,
    runner_root: os.PathLike[str] | str,
    runner_manifest_path: os.PathLike[str] | str,
) -> AuthorizedPlan:
    """Verify every signed input and the isolated probe before worker spawn."""
    binding = Path(binding_path)
    archive = Path(archive_path)
    ledger = Path(source_ledger_path)
    snapshot = Path(snapshot_root)
    contract_path = Path(execution_contract_path)
    expectation_path = Path(runtime_expectation_path)
    runner = Path(runner_root)
    manifest = Path(runner_manifest_path)

    binding_raw, _ = _read_regular(binding)
    payload = verify_operator_binding(
        binding_raw,
        archive,
        ledger,
        snapshot,
        contract_path,
        expectation_path,
        runner,
        manifest,
    )
    expectation_raw, _ = _read_regular(expectation_path)
    validate_runtime_expectation(
        expectation_raw,
        payload["approved_runtime_expectation_sha256"],
        runner,
        manifest,
        payload["approved_runner_manifest_sha256"],
        payload["approved_runner_inventory_sha256"],
    )
    contract_raw, _ = _read_regular(contract_path)
    contract = validate_execution_contract(
        contract_raw,
        payload["approved_execution_contract_sha256"],
        expectation_path,
        runner,
        manifest,
        payload["approved_runner_manifest_sha256"],
        payload["approved_runner_inventory_sha256"],
        payload["approved_runtime_expectation_sha256"],
    )
    probe_report = isolated_torch_probe(
        expectation_raw,
        payload["approved_runtime_expectation_sha256"],
        runner,
        manifest,
        payload["approved_runner_manifest_sha256"],
        payload["approved_runner_inventory_sha256"],
    )
    spawned_specs = derive_spawned_specs(contract, runner)
    provenance = {
        "binding_sha256": digest_bytes(binding_raw),
        "execution_contract_sha256": digest_bytes(contract_raw),
        "probe_report_sha256": digest_bytes(canonical_bytes(probe_report)),
        "runtime_expectation_sha256": digest_bytes(expectation_raw),
        "spawned_specs_sha256": digest_bytes(canonical_bytes(spawned_specs)),
    }
    for key in SPAWN_PROVENANCE_KEYS:
        need(type(provenance[key]) is str and len(provenance[key]) == 64, "provenance digest")
    artifact_hashes = measure_hashes(archive, ledger, snapshot, runner, manifest)
    need(
        artifact_hashes
        == {
            "archive_sha256": payload["approved_archive_sha256"],
            "runner_inventory_sha256": payload["approved_runner_inventory_sha256"],
            "runner_manifest_sha256": payload["approved_runner_manifest_sha256"],
            "snapshot_inventory_sha256": payload["approved_snapshot_inventory_sha256"],
            "snapshot_sha256": payload["approved_snapshot_sha256"],
            "source_ledger_sha256": payload["approved_source_ledger_sha256"],
        },
        "signed artifact hash map",
    )
    return AuthorizedPlan(
        _AUTHORIZED_PLAN_TOKEN,
        artifact_hashes=artifact_hashes,
        binding_path=binding.resolve(strict=True),
        contract_path=contract_path.resolve(strict=True),
        expectation_path=expectation_path.resolve(strict=True),
        provenance=provenance,
        runner_manifest_path=manifest.resolve(strict=True),
        runner_root=runner.resolve(strict=True),
        spawned_specs=spawned_specs,
        timeout_seconds=contract["timeout_seconds"],
    )


class Lifecycle:
    def __init__(
        self,
        terminal_parent: ProtectedParent,
        authorized_plan: AuthorizedPlan,
        archive_path: os.PathLike[str] | str,
        source_ledger_path: os.PathLike[str] | str,
        snapshot_root: os.PathLike[str] | str,
    ):
        need(type(authorized_plan) is AuthorizedPlan, "lifecycle requires authorized plan")
        self.parent = terminal_parent
        self.plan = authorized_plan
        self.archive_path = Path(archive_path)
        self.source_ledger_path = Path(source_ledger_path)
        self.snapshot_root = Path(snapshot_root)
        self.runner_root = authorized_plan.runner_root
        self.runner_manifest_path = authorized_plan.runner_manifest_path
        self.started = False
        self.done = False
        self.handlers_installed = False
        self.pre_hashes: dict | None = None
        self._old_handlers: dict[int, object] = {}
        self._handling_signal = False
        self._lock = threading.RLock()
        self.exit_code: int | None = None
        self._processes: dict[str, subprocess.Popen] = {}
        self._receipts: dict[str, dict] = {
            fault_id: {
                "death_confirmed": False,
                "exit_code": None,
                "fault_id": fault_id,
                "kill_completed": True,
                "kill_required": False,
                "kill_sent": False,
                "pid": None,
                "spawned": False,
                "terminate_sent": False,
                "wait_completed": False,
            }
            for fault_id in TERM_IDS
        }
        self._kill_errors: list[str] = []

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

    def _provenance_files_unchanged(self) -> bool:
        try:
            return (
                digest_file(self.plan.binding_path) == self.plan.provenance["binding_sha256"]
                and digest_file(self.plan.contract_path)
                == self.plan.provenance["execution_contract_sha256"]
                and digest_file(self.plan.expectation_path)
                == self.plan.provenance["runtime_expectation_sha256"]
                and digest_bytes(canonical_bytes(self.plan.spawned_specs))
                == self.plan.provenance["spawned_specs_sha256"]
            )
        except BaseException:
            return False

    def start(self) -> None:
        with self._lock:
            need(self.handlers_installed, "handlers not installed")
            need(not self.started and not self.done, "lifecycle start")
            self.pre_hashes = measure_hashes(
                self.archive_path,
                self.source_ledger_path,
                self.snapshot_root,
                self.runner_root,
                self.runner_manifest_path,
            )
            validate_hash_map(self.pre_hashes)
            need(self.pre_hashes == self.plan.artifact_hashes, "authorized artifacts changed before start")
            need(self._provenance_files_unchanged(), "authorized provenance changed before start")
            self.started = True

    def _spawn_worker(self, spec: dict) -> int:
        fault_id = spec["fault_id"]
        with self._lock:
            need(self.started and not self.done, "worker spawn order")
            need(fault_id in TERM_IDS and fault_id not in self._processes, "worker id/reuse")
            need(type(spec["argv"]) is list and bool(spec["argv"]), "worker argv")
            need(type(spec["env"]) is dict, "worker env")
            cwd_path = Path(spec["cwd"])
            need(str(cwd_path.resolve(strict=True)) == spec["cwd"], "derived absolute cwd")
            try:
                cwd_path.relative_to(self.runner_root)
            except ValueError as exc:
                raise Reject("worker cwd outside runner") from exc
            try:
                process = subprocess.Popen(
                    list(spec["argv"]),
                    cwd=spec["cwd"],
                    env=dict(spec["env"]),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                    start_new_session=True,
                )
            except OSError as exc:
                raise Reject("worker spawn") from exc
            self._processes[fault_id] = process
            receipt = self._receipts[fault_id]
            receipt["spawned"] = True
            receipt["pid"] = process.pid
            return process.pid

    def spawn_workers(self) -> None:
        """Spawn exactly V8F01--V8F08 from the signed/validated contract."""
        need(self._provenance_files_unchanged(), "provenance changed before spawn")
        specs = self.plan.spawned_specs
        need(
            digest_bytes(canonical_bytes(specs))
            == self.plan.provenance["spawned_specs_sha256"],
            "local spawned specs commitment",
        )
        need(
            type(specs) is dict
            and specs.get("schema_version") == "forkaudit-v8-spawned-specs-v1"
            and type(specs.get("workers")) is list
            and [row.get("fault_id") for row in specs["workers"]] == list(TERM_IDS),
            "authorized spawned specs",
        )
        try:
            for spec in specs["workers"]:
                self._spawn_worker(spec)
        except BaseException:
            self._cleanup_workers()
            raise

    def _record_wait(self, fault_id: str, returncode: int) -> None:
        need(type(returncode) is int, "actual worker return code")
        receipt = self._receipts[fault_id]
        receipt["exit_code"] = returncode
        receipt["wait_completed"] = True
        receipt["death_confirmed"] = self._processes[fault_id].poll() is not None

    def wait_workers(self) -> list[int]:
        with self._lock:
            need(self.started and not self.done, "worker wait order")
            need(set(self._processes) == set(TERM_IDS), "all workers must be spawned")
            deadline = time.monotonic() + self.plan.timeout_seconds
            codes = []
            for fault_id in TERM_IDS:
                process = self._processes[fault_id]
                remaining = max(0.001, deadline - time.monotonic())
                try:
                    code = process.wait(timeout=remaining)
                except subprocess.TimeoutExpired as exc:
                    raise Reject("worker wait timeout") from exc
                self._record_wait(fault_id, code)
                codes.append(code)
            return codes

    def _cleanup_workers(self) -> None:
        for fault_id in TERM_IDS:
            process = self._processes.get(fault_id)
            if process is None:
                continue
            receipt = self._receipts[fault_id]
            current = process.poll()
            if current is not None:
                self._record_wait(fault_id, current)
                continue
            receipt["kill_required"] = True
            receipt["kill_completed"] = False
            receipt["terminate_sent"] = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (OSError, ProcessLookupError) as exc:
                self._kill_errors.append(f"{fault_id}:terminate:{type(exc).__name__}")
            try:
                code = process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                receipt["kill_sent"] = True
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (OSError, ProcessLookupError) as exc:
                    self._kill_errors.append(f"{fault_id}:kill:{type(exc).__name__}")
                try:
                    code = process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._kill_errors.append(f"{fault_id}:wait:TimeoutExpired")
                    continue
            self._record_wait(fault_id, code)
            receipt["kill_completed"] = receipt["death_confirmed"]

    def _derived_gate(
        self,
        *,
        inventory_verified: bool,
        post_rehash_verified: bool,
        provenance_verified: bool,
    ) -> dict:
        workers = [dict(self._receipts[fault_id]) for fault_id in TERM_IDS]
        kill_required = any(row["kill_required"] for row in workers)
        kill_completed = all(row["kill_completed"] for row in workers) and not self._kill_errors
        receipts_verified = all(
            (not row["spawned"])
            or (
                row["wait_completed"]
                and row["death_confirmed"]
                and type(row["exit_code"]) is int
            )
            for row in workers
        )
        verification_complete = (
            inventory_verified
            and post_rehash_verified
            and provenance_verified
            and receipts_verified
            and kill_completed
        )
        gate = {
            "inventory_verified": inventory_verified,
            "kill_completion": {
                "completed": kill_completed,
                "errors": list(self._kill_errors),
                "required": kill_required,
            },
            "post_rehash_verified": post_rehash_verified,
            "provenance": copy.deepcopy(self.plan.provenance),
            "provenance_verified": provenance_verified,
            "receipts_verified": receipts_verified,
            "schema_version": "forkaudit-v8-lifecycle-gate-v1",
            "verification_complete": verification_complete,
            "workers": workers,
        }
        return lifecycle_gate(gate)

    def _post_hashes(self) -> tuple[dict, str | None]:
        try:
            measured = measure_hashes(
                self.archive_path,
                self.source_ledger_path,
                self.snapshot_root,
                self.runner_root,
                self.runner_manifest_path,
            )
            return validate_hash_map(measured), None
        except BaseException as exc:
            return {key: None for key in HASH_KEYS}, f"{type(exc).__name__}:{exc}"

    def finalize(self, reason: str, signum: int | None = None) -> int:
        with self._lock:
            if self.done:
                return self.exit_code if self.exit_code is not None else 1
            need(type(reason) is str and bool(reason), "finalize reason")
            need(
                signum is None
                or (type(signum) is int and signum in (signal.SIGINT, signal.SIGTERM)),
                "finalize signal",
            )
            self._cleanup_workers()
            if self.pre_hashes is None:
                try:
                    self.pre_hashes = measure_hashes(
                        self.archive_path,
                        self.source_ledger_path,
                        self.snapshot_root,
                        self.runner_root,
                        self.runner_manifest_path,
                    )
                except BaseException:
                    self.pre_hashes = {key: None for key in HASH_KEYS}
            post_hashes, rehash_error = self._post_hashes()
            inventory_verified = self.pre_hashes is not None and all(
                self.pre_hashes.get(key) is not None for key in HASH_KEYS
            )
            post_verified = rehash_error is None and self.pre_hashes == post_hashes
            provenance_verified = self._provenance_files_unchanged()
            gate = self._derived_gate(
                inventory_verified=inventory_verified,
                post_rehash_verified=post_verified,
                provenance_verified=provenance_verified,
            )
            success = False
            if reason == "success" and signum is None and self.started and rehash_error is None:
                try:
                    lifecycle_gate(gate, require_success=True)
                    success = self.pre_hashes == post_hashes
                except BaseException:
                    success = False
            status = "success" if success else "failure"
            terminal_reason = (
                "success"
                if success
                else ("success-gate-rejected" if reason == "success" else reason)
            )
            if rehash_error is not None:
                terminal_reason = f"{reason};post-rehash={rehash_error}"
            provenance = copy.deepcopy(self.plan.provenance)
            provenance_sha = digest_bytes(canonical_bytes(provenance))
            outputs = {}
            for fault_id in TERM_IDS:
                outputs[f"{fault_id}.terminal.json"] = {
                    "fault_id": fault_id,
                    "lifecycle_receipt": gate,
                    "lifecycle_receipt_sha256": digest_bytes(canonical_bytes(gate)),
                    "post_hashes": post_hashes,
                    "pre_hashes": self.pre_hashes,
                    "provenance": provenance,
                    "provenance_sha256": provenance_sha,
                    "reason": terminal_reason,
                    "schema_version": "forkaudit-v8-terminal-v1",
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
            self.exit_code = signal_exit(signum) if signum is not None else (0 if success else 1)
            return self.exit_code

    def _handle_signal(self, signum: int, _frame: object) -> None:
        if self._handling_signal:
            os._exit(signal_exit(signum))
        self._handling_signal = True
        code = self.finalize("signal", signum)
        raise SystemExit(code)
