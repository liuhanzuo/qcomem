from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


GPU_DIR = Path(__file__).resolve().parent
BUILDER = GPU_DIR / "qcomem_forkaudit_execution_identity.py"


def freeze_tree(root: Path) -> None:
    for directory, directory_names, file_names in os.walk(root, topdown=False):
        for name in file_names:
            os.chmod(Path(directory) / name, 0o444)
        for name in directory_names:
            os.chmod(Path(directory) / name, 0o555)
    os.chmod(root, 0o555)


def thaw_tree(root: Path) -> None:
    if not root.exists():
        return
    os.chmod(root, 0o755)
    for directory, directory_names, file_names in os.walk(root):
        for name in directory_names:
            os.chmod(Path(directory) / name, 0o755)
        for name in file_names:
            os.chmod(Path(directory) / name, 0o644)


class ExecutionIdentityTest(unittest.TestCase):
    def capture_command(
        self,
        *,
        source: Path,
        cache_root: Path,
        output: Path,
        require_empty: bool = False,
    ) -> list[str]:
        command = [
            sys.executable,
            "-I",
            "-B",
            os.fspath(BUILDER),
            "capture",
            "--source-root",
            os.fspath(source),
            "--python",
            sys.executable,
            "--cache",
            f"triton={cache_root / 'triton'}",
            "--cache",
            f"torchinductor={cache_root / 'torchinductor'}",
            "--cache",
            f"cuda={cache_root / 'cuda'}",
            "--command-file",
            f"runner={source / 'runner.py'}",
            "--command-template",
            "python -I -B runner.py --stage shard --rank 0",
            "--output",
            os.fspath(output),
        ]
        if require_empty:
            command.append("--require-empty-caches")
        return command

    def test_stable_source_environment_and_terminal_cache_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "runner.py").write_text("VALUE = 1\n", encoding="utf-8")
            cache_root = root / "cache"
            before = root / "before.json"
            after = root / "after.json"
            verification = root / "verification.json"
            freeze_tree(source)
            try:
                first = subprocess.run(
                    self.capture_command(
                        source=source,
                        cache_root=cache_root,
                        output=before,
                        require_empty=True,
                    ),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(first.returncode, 0, first.stderr)
                (cache_root / "triton" / "kernel").mkdir(parents=True)
                (cache_root / "triton" / "kernel" / "compiled.so").write_bytes(
                    b"compiled-kernel"
                )
                (cache_root / "triton" / "autotune-best_config.json").write_text(
                    '{"block":128}\n', encoding="utf-8"
                )
                second = subprocess.run(
                    self.capture_command(
                        source=source,
                        cache_root=cache_root,
                        output=after,
                    ),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(second.returncode, 0, second.stderr)
                verified = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-B",
                        os.fspath(BUILDER),
                        "verify-stable",
                        "--before",
                        os.fspath(before),
                        "--after",
                        os.fspath(after),
                        "--output",
                        os.fspath(verification),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(verified.returncode, 0, verified.stderr)
                value = json.loads(verification.read_text(encoding="utf-8"))
                self.assertEqual(value["status"], "execution_identity_stable")
                self.assertEqual(value["compiled_kernel_artifact_count"], 1)
                self.assertEqual(value["autotune_artifact_count"], 1)
                self.assertIsNone(value["autotune_identity_limitation"])
                terminal = json.loads(after.read_text(encoding="utf-8"))
                self.assertEqual(
                    terminal["source"]["closure_sha256"],
                    json.loads(before.read_text(encoding="utf-8"))["source"][
                        "closure_sha256"
                    ],
                )
                self.assertTrue(terminal["executable"]["sha256"])
                self.assertTrue(terminal["environment"]["identity_sha256"])
                self.assertTrue(terminal["command"]["identity_sha256"])
            finally:
                thaw_tree(source)

    def test_source_tamper_is_rejected_by_terminal_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            runner = source / "runner.py"
            runner.write_text("VALUE = 1\n", encoding="utf-8")
            cache_root = root / "cache"
            before = root / "before.json"
            after = root / "after.json"
            freeze_tree(source)
            try:
                subprocess.run(
                    self.capture_command(
                        source=source,
                        cache_root=cache_root,
                        output=before,
                        require_empty=True,
                    ),
                    check=True,
                )
                os.chmod(runner, 0o644)
                runner.write_text("VALUE = 2\n", encoding="utf-8")
                os.chmod(runner, 0o444)
                subprocess.run(
                    self.capture_command(
                        source=source,
                        cache_root=cache_root,
                        output=after,
                    ),
                    check=True,
                )
                rejected = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-B",
                        os.fspath(BUILDER),
                        "verify-stable",
                        "--before",
                        os.fspath(before),
                        "--after",
                        os.fspath(after),
                        "--output",
                        os.fspath(root / "verification.json"),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn("terminal execution identity drift", rejected.stderr)
            finally:
                thaw_tree(source)

    def test_pycache_in_source_tree_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            pycache = source / "__pycache__"
            pycache.mkdir(parents=True)
            (source / "runner.py").write_text("VALUE = 1\n", encoding="utf-8")
            (pycache / "runner.pyc").write_bytes(b"stale")
            freeze_tree(source)
            try:
                rejected = subprocess.run(
                    self.capture_command(
                        source=source,
                        cache_root=root / "cache",
                        output=root / "receipt.json",
                        require_empty=True,
                    ),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn("Python bytecode cache in source tree", rejected.stderr)
            finally:
                thaw_tree(source)


if __name__ == "__main__":
    unittest.main()
