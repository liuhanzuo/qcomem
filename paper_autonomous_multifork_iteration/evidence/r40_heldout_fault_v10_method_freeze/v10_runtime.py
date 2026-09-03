from __future__ import annotations

import os
import stat
import threading
from pathlib import Path

from v10_guard import Reject, canonical_bytes, need


class ProtectedParent:
    """No-replace publisher bound to one retained directory inode."""

    __slots__ = ("_closed", "_fd", "_key", "_path")

    def __init__(self, path: os.PathLike[str] | str):
        absolute = Path(path).absolute()
        before = os.lstat(absolute)
        need(stat.S_ISDIR(before.st_mode) and not stat.S_ISLNK(before.st_mode), "parent directory")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(absolute, flags)
        opened = os.fstat(fd)
        need((opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino), "parent open race")
        object.__setattr__(self, "_path", absolute)
        object.__setattr__(self, "_fd", fd)
        object.__setattr__(self, "_key", (opened.st_dev, opened.st_ino))
        object.__setattr__(self, "_closed", False)

    def check(self) -> None:
        need(not self._closed, "closed parent")
        retained = os.fstat(self._fd)
        need((retained.st_dev, retained.st_ino) == self._key, "retained parent changed")
        try:
            named = os.stat(self._path, follow_symlinks=False)
        except OSError as exc:
            raise Reject("parent path absent") from exc
        need(stat.S_ISDIR(named.st_mode), "parent path not directory")
        need((named.st_dev, named.st_ino) == self._key, "parent path replaced")

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
            os.stat(name, dir_fd=self._fd, follow_symlinks=False)
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
                    dir_fd=self._fd,
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
                        src_dir_fd=self._fd,
                        dst_dir_fd=self._fd,
                        follow_symlinks=False,
                    )
                except FileExistsError as exc:
                    raise Reject("no replace") from exc
                linked.append(name)

            for temporary in list(staged):
                os.unlink(temporary, dir_fd=self._fd)
                staged.remove(temporary)
            os.fsync(self._fd)
            self.check()
        except BaseException:
            for name in reversed(linked):
                try:
                    os.unlink(name, dir_fd=self._fd)
                except FileNotFoundError:
                    pass
            for temporary in list(staged):
                try:
                    os.unlink(temporary, dir_fd=self._fd)
                except FileNotFoundError:
                    pass
            try:
                os.fsync(self._fd)
            except OSError:
                pass
            raise

    def publish_json_many(self, outputs: dict[str, object]) -> None:
        self.publish_many({name: canonical_bytes(value) for name, value in outputs.items()})

    def close(self) -> None:
        if not self._closed:
            os.close(self._fd)
            object.__setattr__(self, "_closed", True)

    def __enter__(self) -> "ProtectedParent":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
