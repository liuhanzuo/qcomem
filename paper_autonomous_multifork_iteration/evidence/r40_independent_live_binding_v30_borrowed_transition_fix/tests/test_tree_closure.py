from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "executed_source/r40_tree_closure.py"
sys.path.insert(0, str(ROOT / "executed_source"))
from r40_tree_closure import (  # noqa: E402
    AGGREGATE_FIELDS,
    CUDA_FIELDS,
    EXPECTATION_FIELDS,
    FORMAL_TOP_LEVELS,
    PREFLIGHT_EXACT_PATHS,
    PRIMARY_EXACT_LOGS,
    PRIMARY_EXACT_STAGES,
    PRIMARY_CODE_PYCACHE_PREFIX,
    PRIMARY_PYCACHE_TAG,
    TREE_FIELDS,
    lexical_tree,
    prepare_terminal_expectation,
    publish_empty_exclusive,
    write_terminal_ledger,
)


SOURCE_SHA = "a" * 64


def reseal(value: dict[str, object]) -> dict[str, object]:
    value = dict(value)
    value["payload_sha256"] = None
    value["payload_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return value


class TreeClosure(unittest.TestCase):
    def fixture(self) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        (root / "regular.json").write_text('{"schema_version":"fixture-v1","value":1}\n')
        (root / "dir").mkdir()
        return root

    def prepare(self, root: Path, *, publish_complete: bool = True) -> tuple[Path, Path, Path]:
        expectation = root / "terminal-closure.json"
        complete = root / "COMPLETE"
        ledger = root / "terminal-tree.json"
        prepare_terminal_expectation(
            root,
            expectation,
            terminal_tree_output=ledger,
            complete_output=complete,
            source_ledger_sha256=SOURCE_SHA,
            profile="fixture",
            expected_existing_paths=("dir", "regular.json"),
        )
        if publish_complete:
            publish_empty_exclusive(root, complete)
        return expectation, complete, ledger

    def command(self, *arguments: object) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *(str(argument) for argument in arguments)],
            capture_output=True,
            text=True,
        )

    def formal_fixture(self) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        for top_level in FORMAL_TOP_LEVELS:
            (root / top_level).mkdir()

        preflight_directories = {
            "preflight",
            "preflight/logs",
            "preflight/pycache",
            "preflight/stages",
        }
        for relative in preflight_directories:
            (root / relative).mkdir(parents=True, exist_ok=True)
        for relative in PREFLIGHT_EXACT_PATHS - preflight_directories:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n" if path.suffix == ".json" else "ok\n")

        for directory in ("logs", "model-view", "preregistration", "pycache", "raw/shards", "receipts", "stages"):
            (root / "primary" / directory).mkdir(parents=True, exist_ok=True)
        for name in PRIMARY_EXACT_LOGS:
            path = root / "primary/logs" / name
            path.write_text("{}\n" if path.suffix == ".json" else "ok\n")
        for name in PRIMARY_EXACT_STAGES:
            (root / "primary/stages" / name).write_text("ok\n")
        (root / "primary/forkaudit-summary.json").write_text("{}\n")
        (root / "primary/gpus-before.csv").write_text("gpu\n")

        code_rows = []
        code_targets = [f"source-{index:02d}.py" for index in range(31)] + [
            "FORKAUDIT_REVIEW_REVISION_PROTOCOL_ZH.md",
            "launch_qcomem_qwen35_forkaudit_review_revision_8gpu.sh",
            "qcomem_qwen35_vllm_paged_multifork_resident_protocol_manifest.json",
        ]
        for target in code_targets:
            code_rows.append(f"{'a' * 64}  ./{target}\n")
        code_ledger = root / "primary/preregistration/code.sha256"
        code_ledger.write_text("".join(code_rows))
        for target in code_targets[:31]:
            pycache = root / f"{PRIMARY_CODE_PYCACHE_PREFIX}/{target[:-3]}{PRIMARY_PYCACHE_TAG}"
            pycache.parent.mkdir(parents=True, exist_ok=True)
            pycache.write_bytes(b"fixture-cpython-311-bytecode\n")

        model_rows = []
        for index in range(14):
            relative = f"model-{index:02d}.bin"
            data = f"model-{index}\n".encode()
            (root / "primary/model-view" / relative).write_bytes(data)
            model_rows.append(
                {
                    "relative_path": relative,
                    "ledger_roles": ["model_weight"],
                    "declared_sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                    "copy_mode": "byte-copy",
                    "source_device": 1,
                    "source_inode": index + 1,
                    "view_device": 1,
                    "view_inode": index + 101,
                    "source_and_view_inode_distinct": True,
                }
            )
        manifest = {
            "schema_version": "qcomem-forkaudit-private-model-view-v1",
            "model_id": "fixture",
            "model_revision": "fixture",
            "model_artifact_ledger_raw_sha256": "a" * 64,
            "model_weight_ledger_raw_sha256": "b" * 64,
            "copy_policy": "ficlone-then-byte-copy;hardlink-and-symlink-forbidden",
            "file_count": len(model_rows),
            "weight_file_count": 14,
            "all_source_and_view_inodes_distinct": True,
            "all_view_files_regular": True,
            "all_view_files_read_only": True,
            "rows": model_rows,
            "generated_before_candidate_outputs": True,
            "cuda_initialized": False,
        }
        manifest_path = root / "primary/preregistration/private-model-view-manifest.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        digest_receipt = root / "primary/receipts/run-id-receipt.canonical-json.sha256"
        digest_receipt.write_text("c" * 64 + "\n")
        scientific_targets = (
            "forkaudit-summary.json",
            "preregistration/code.sha256",
            "preregistration/private-model-view-manifest.json",
            "receipts/run-id-receipt.canonical-json.sha256",
        )
        scientific_rows = []
        for target in scientific_targets:
            digest = hashlib.sha256((root / "primary" / target).read_bytes()).hexdigest()
            scientific_rows.append(f"{digest}  {target}\n")
        (root / "primary/scientific-artifacts.sha256").write_text("".join(scientific_rows))

        formal_targets = ["formal-aggregate.json"]
        (root / "formal-binding/formal-aggregate.json").write_text("{}\n")
        for rank in range(8):
            rank_root = root / f"formal-binding/rank-{rank}"
            for name in (
                "negative-controls.json",
                "primary-shard-replay.json",
                "replay.json",
                "source-snapshot-manifest.json",
            ):
                path = rank_root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n")
                formal_targets.append(f"rank-{rank}/{name}")
            (rank_root / "runtime-cache/triton").mkdir(parents=True)
            (rank_root / "runtime-cache/triton/kernel.bin").write_bytes(b"kernel\n")
            formal_targets.append(f"rank-{rank}/runtime-cache/triton/kernel.bin")
            (rank_root / "source-snapshot/code").mkdir(parents=True)
            (rank_root / "source-snapshot/code/source.py").write_text("# source\n")
            formal_targets.append(f"rank-{rank}/source-snapshot/code/source.py")

            capture = root / f"compiled-dispatch-capture/rank-{rank}"
            (capture / "runtime-cache/triton").mkdir(parents=True)
            (capture / "runtime-cache/triton/kernel.bin").write_bytes(b"kernel\n")
            (capture / "raw").mkdir()
            runner_argv = ["--rank", str(rank), "--fixture"]
            runner_sha256 = hashlib.sha256(f"runner-{rank}".encode()).hexdigest()
            primary_shard_sha256 = hashlib.sha256(f"shard-{rank}".encode()).hexdigest()
            runner_argv_sha256 = hashlib.sha256(
                json.dumps(
                    runner_argv,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode()
            ).hexdigest()
            receipt = {
                "execution_binding": {
                    "runner_sha256": runner_sha256,
                    "runner_argv": runner_argv,
                    "runner_argv_sha256": runner_argv_sha256,
                    "primary_shard_sha256": primary_shard_sha256,
                }
            }
            receipt_bytes = (
                json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                + "\n"
            )
            (rank_root / "primary-compiled-dispatch-receipt.json").write_text(receipt_bytes)
            formal_targets.append(f"rank-{rank}/primary-compiled-dispatch-receipt.json")
            (capture / "raw/primary-compiled-dispatch-receipt.json").write_text(receipt_bytes)
            invocation = {
                "schema_version": "forkaudit-r39-primary-rank-invocation-v2",
                "rank": rank,
                "runner_sha256": runner_sha256,
                "runner_argv": runner_argv,
                "primary_shard_sha256": primary_shard_sha256,
            }
            (capture / "invocation.json").write_text(
                json.dumps(invocation, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                + "\n"
            )
        ledger_rows = []
        for target in sorted(formal_targets):
            digest = hashlib.sha256((root / "formal-binding" / target).read_bytes()).hexdigest()
            ledger_rows.append(f"{digest}  {target}\n")
        (root / "formal-binding/terminal-files.sha256").write_text("".join(ledger_rows))
        (root / "formal-binding/COMPLETE").write_bytes(b"")

        for rank in range(8):
            raw = root / f"r40-clean-binding/rank-{rank}/raw"
            raw.mkdir(parents=True)
            (raw / "real-binding.json").write_text("{}\n")
            (raw / "global-absence.json").write_text("{}\n")
        cuda = {field: None for field in CUDA_FIELDS}
        cuda["schema_version"] = "forkaudit-r40-v16-cuda-smoke-v1"
        aggregate = {field: None for field in AGGREGATE_FIELDS}
        aggregate["schema_version"] = "forkaudit-r40-v16-clean-aggregate-v1"
        (root / "r40-formal/cuda-smoke.json").write_text(json.dumps(cuda) + "\n")
        (root / "r40-formal/aggregate.json").write_text(json.dumps(aggregate) + "\n")
        return root

    def test_terminal_ledger_has_exact_whitelist_counts_and_per_file_schema(self):
        root = self.fixture()
        expectation, _, ledger = self.prepare(root)
        payload = write_terminal_ledger(root, ledger, expectation)
        self.assertEqual(set(payload), TREE_FIELDS)
        self.assertEqual(payload["expected_paths"], ["COMPLETE", "dir", "regular.json", "terminal-closure.json"])
        self.assertEqual(
            (
                payload["expected_node_count"],
                payload["expected_regular_file_count"],
                payload["expected_directory_count"],
                payload["final_node_count"],
                payload["final_regular_file_count"],
                payload["final_directory_count"],
            ),
            (4, 3, 1, 5, 4, 1),
        )
        self.assertEqual(payload["nodes"]["dir"], {"kind": "directory"})
        for relative in ("COMPLETE", "regular.json", "terminal-closure.json"):
            self.assertEqual(
                set(payload["nodes"][relative]),
                {"kind", "bytes", "sha256", "content_schema"},
            )
        self.assertEqual(
            payload["nodes"]["regular.json"]["content_schema"],
            {
                "format": "json",
                "top_level_type": "object",
                "exact_fields": ["schema_version", "value"],
                "schema_version": "fixture-v1",
            },
        )
        self.assertEqual(set(json.loads(expectation.read_text())), EXPECTATION_FIELDS)
        self.assertEqual(set(lexical_tree(root)), {"COMPLETE", "dir", "regular.json", "terminal-closure.json", "terminal-tree.json"})

    def test_root_symlink_and_symlinked_parent_rejected_before_resolve_or_write(self):
        outer = self.fixture()
        real = outer / "dir" / "real-root"
        real.mkdir()
        (real / "regular.json").write_text("{}\n")
        link = outer / "root-link"
        link.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "root symlink"):
            lexical_tree(link)
        result = self.command(
            "prepare",
            "--root",
            link,
            "--output",
            link / "terminal-closure.json",
            "--terminal-tree-output",
            link / "terminal-tree.json",
            "--complete-output",
            link / "COMPLETE",
            "--source-ledger-sha256",
            SOURCE_SHA,
            "--profile",
            "fixture",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("root symlink", result.stderr)
        self.assertFalse((real / "terminal-closure.json").exists())

        parent_link = outer / "parent-link"
        parent_link.symlink_to(outer / "dir", target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "canonical tree root differs"):
            lexical_tree(parent_link / "real-root")

    def test_dotdot_output_rejected_by_command_with_no_outside_side_effect(self):
        root = self.fixture()
        outside = root.parent / f"{root.name}-outside.json"
        result = self.command(
            "prepare",
            "--root",
            root,
            "--output",
            root / ".." / outside.name,
            "--terminal-tree-output",
            root / "terminal-tree.json",
            "--complete-output",
            root / "COMPLETE",
            "--source-ledger-sha256",
            SOURCE_SHA,
            "--profile",
            "fixture",
            "--expected-existing-path",
            "dir",
            "--expected-existing-path",
            "regular.json",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("dotdot", result.stderr)
        self.assertFalse(outside.exists())
        self.assertEqual(set(lexical_tree(root)), {"dir", "regular.json"})

    def test_extra_regular_file_fails_exact_whitelist_before_ledger_write(self):
        root = self.fixture()
        expectation, _, ledger = self.prepare(root)
        (root / "extra.txt").write_text("unexpected")
        result = self.command("close", "--root", root, "--output", ledger, "--expectation", expectation)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exact whitelist", result.stderr)
        self.assertFalse(ledger.exists())

    def test_extra_directory_fails_exact_whitelist_before_ledger_write(self):
        root = self.fixture()
        expectation, _, ledger = self.prepare(root)
        (root / "extra-dir").mkdir()
        result = self.command("close", "--root", root, "--output", ledger, "--expectation", expectation)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exact whitelist", result.stderr)
        self.assertFalse(ledger.exists())

    def test_formal_prepare_rejects_preexisting_extra_regular_and_directory(self):
        for kind in ("regular", "directory"):
            root = self.formal_fixture()
            unexpected = root / f"primary/UNEXPECTED-{'file.txt' if kind == 'regular' else 'directory'}"
            if kind == "regular":
                unexpected.write_text("must not be blessed\n")
            else:
                unexpected.mkdir()

            listed = self.command("expected-paths", "--root", root)
            self.assertNotEqual(listed.returncode, 0)
            self.assertIn("exact expected path whitelist", listed.stderr)
            observed = sorted(lexical_tree(root))
            arguments: list[object] = [
                "prepare",
                "--root",
                root,
                "--output",
                root / "r40-formal/terminal-closure.json",
                "--terminal-tree-output",
                root / "r40-formal/terminal-tree.json",
                "--complete-output",
                root / "COMPLETE",
                "--source-ledger-sha256",
                SOURCE_SHA,
                "--profile",
                "r40-v16-formal",
            ]
            for relative in observed:
                arguments.extend(("--expected-existing-path", relative))
            result = self.command(*arguments)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exact expected path whitelist", result.stderr)
            self.assertFalse((root / "r40-formal/terminal-closure.json").exists())

    def test_formal_predetermined_path_plan_and_terminal_close_succeed(self):
        root = self.formal_fixture()
        listed = self.command("expected-paths", "--root", root)
        self.assertEqual(listed.returncode, 0, listed.stderr)
        expected = listed.stdout.splitlines()
        self.assertEqual(expected, sorted(lexical_tree(root)))
        invocation_paths = [
            relative
            for relative in expected
            if relative.startswith("compiled-dispatch-capture/rank-")
            and relative.endswith("/invocation.json")
        ]
        self.assertEqual(
            invocation_paths,
            [f"compiled-dispatch-capture/rank-{rank}/invocation.json" for rank in range(8)],
        )
        arguments: list[object] = [
            "prepare",
            "--root",
            root,
            "--output",
            root / "r40-formal/terminal-closure.json",
            "--terminal-tree-output",
            root / "r40-formal/terminal-tree.json",
            "--complete-output",
            root / "COMPLETE",
            "--source-ledger-sha256",
            SOURCE_SHA,
            "--profile",
            "r40-v16-formal",
        ]
        for relative in expected:
            arguments.extend(("--expected-existing-path", relative))
        self.assertEqual(self.command(*arguments).returncode, 0)
        self.assertEqual(
            self.command("complete", "--root", root, "--output", root / "COMPLETE").returncode,
            0,
        )
        closed = self.command(
            "close",
            "--root",
            root,
            "--output",
            root / "r40-formal/terminal-tree.json",
            "--expectation",
            root / "r40-formal/terminal-closure.json",
        )
        self.assertEqual(closed.returncode, 0, closed.stderr)

    def test_primary_result_pycache_exact_31_file_13_directory_projection(self):
        root = self.formal_fixture()
        listed = self.command("expected-paths", "--root", root)
        self.assertEqual(listed.returncode, 0, listed.stderr)
        observed = lexical_tree(root)
        files = [
            path for path, row in observed.items()
            if path.startswith("primary/pycache/") and row["kind"] == "regular"
        ]
        directories = [
            path for path, row in observed.items()
            if path.startswith("primary/pycache/") and row["kind"] == "directory"
        ]
        self.assertEqual((len(files), len(directories)), (31, 13))

        for mutation in ("missing", "orphan", "wrong-abi", "wrong-suffix", "extra-directory"):
            with self.subTest(mutation=mutation):
                candidate = self.formal_fixture()
                prefix = candidate / PRIMARY_CODE_PYCACHE_PREFIX
                first = prefix / f"source-00{PRIMARY_PYCACHE_TAG}"
                if mutation == "missing":
                    first.unlink()
                elif mutation == "orphan":
                    (prefix / f"orphan{PRIMARY_PYCACHE_TAG}").write_bytes(b"orphan-bytecode-payload\n")
                elif mutation == "wrong-abi":
                    first.rename(prefix / "source-00.cpython-310.pyc")
                elif mutation == "wrong-suffix":
                    first.rename(prefix / "source-00.cpython-311.bin")
                else:
                    (prefix / "extra-directory").mkdir()
                rejected = self.command("expected-paths", "--root", candidate)
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn("primary pycache", rejected.stderr)
                self.assertFalse((candidate / "r40-formal/terminal-closure.json").exists())

    def test_formal_invocation_schema_and_receipt_binding_fail_closed(self):
        mutations = (
            ("missing", None),
            ("extra-field", lambda value: value.__setitem__("extra", True)),
            ("schema-version", lambda value: value.__setitem__("schema_version", "wrong")),
            ("rank", lambda value: value.__setitem__("rank", 7)),
            ("runner-sha", lambda value: value.__setitem__("runner_sha256", "0" * 64)),
            ("runner-argv", lambda value: value["runner_argv"].append("--drift")),
            ("shard-sha", lambda value: value.__setitem__("primary_shard_sha256", "0" * 64)),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                root = self.formal_fixture()
                invocation = root / "compiled-dispatch-capture/rank-0/invocation.json"
                if mutate is None:
                    invocation.unlink()
                else:
                    value = json.loads(invocation.read_text())
                    mutate(value)
                    invocation.write_text(
                        json.dumps(
                            value,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
                listed = self.command("expected-paths", "--root", root)
                self.assertNotEqual(listed.returncode, 0)
                self.assertIn("invocation", listed.stderr)
                self.assertFalse((root / "r40-formal/terminal-closure.json").exists())
                self.assertFalse((root / "r40-formal/terminal-tree.json").exists())
                self.assertFalse((root / "COMPLETE").exists())

    def test_symlink_fifo_socket_and_hardlink_nodes_fail_closed(self):
        creators = (
            lambda root: os.mkfifo(root / "bad"),
            lambda root: (root / "bad").symlink_to(root / "regular.json"),
            lambda root: os.link(root / "regular.json", root / "bad"),
        )
        for create in creators:
            root = self.fixture()
            create(root)
            with self.assertRaisesRegex(RuntimeError, "special|symlink|hardlink"):
                lexical_tree(root)
        root = self.fixture()
        sock = socket.socket(socket.AF_UNIX)
        self.addCleanup(sock.close)
        sock.bind(str(root / "bad"))
        with self.assertRaisesRegex(RuntimeError, "special"):
            lexical_tree(root)

    def test_all_preexisting_terminal_output_node_types_fail_closed(self):
        for kind in ("regular", "directory", "symlink", "fifo"):
            root = self.fixture()
            output = root / "terminal-closure.json"
            if kind == "regular":
                output.write_text("sentinel")
            elif kind == "directory":
                output.mkdir()
            elif kind == "symlink":
                output.symlink_to(root / "missing")
            else:
                os.mkfifo(output)
            with self.assertRaisesRegex(FileExistsError, "overwrite|special"):
                prepare_terminal_expectation(
                    root,
                    output,
                    terminal_tree_output=root / "terminal-tree.json",
                    complete_output=root / "COMPLETE",
                    source_ledger_sha256=SOURCE_SHA,
                    profile="fixture",
                )

    def test_resealed_count_and_per_file_schema_tamper_fail_closed(self):
        for mutation, pattern in (
            (lambda value: value.__setitem__("expected_node_count", value["expected_node_count"] + 1), "node count"),
            (
                lambda value: value["expected_nodes"]["regular.json"]["content_schema"].__setitem__("extra", True),
                "exact schema",
            ),
        ):
            root = self.fixture()
            expectation, _, ledger = self.prepare(root)
            value = json.loads(expectation.read_text())
            mutation(value)
            expectation.write_text(json.dumps(reseal(value), sort_keys=True, separators=(",", ":")) + "\n")
            result = self.command("close", "--root", root, "--output", ledger, "--expectation", expectation)
            self.assertNotEqual(result.returncode, 0)
            self.assertRegex(result.stderr, pattern)
            self.assertFalse(ledger.exists())

    def test_terminal_closure_command_is_exclusive_nonoverwrite(self):
        root = self.fixture()
        args = (
            "prepare",
            "--root",
            root,
            "--output",
            root / "terminal-closure.json",
            "--terminal-tree-output",
            root / "terminal-tree.json",
            "--complete-output",
            root / "COMPLETE",
            "--source-ledger-sha256",
            SOURCE_SHA,
            "--profile",
            "fixture",
            "--expected-existing-path",
            "dir",
            "--expected-existing-path",
            "regular.json",
        )
        self.assertEqual(self.command(*args).returncode, 0)
        before = (root / "terminal-closure.json").read_bytes()
        self.assertNotEqual(self.command(*args).returncode, 0)
        self.assertEqual((root / "terminal-closure.json").read_bytes(), before)

    def test_complete_command_is_exclusive_nonoverwrite(self):
        root = self.fixture()
        expectation, _, _ = self.prepare(root, publish_complete=False)
        self.assertTrue(expectation.is_file())
        args = ("complete", "--root", root, "--output", root / "COMPLETE")
        self.assertEqual(self.command(*args).returncode, 0)
        result = self.command(*args)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((root / "COMPLETE").read_bytes(), b"")

    def test_terminal_tree_command_is_exclusive_nonoverwrite(self):
        root = self.fixture()
        expectation, _, ledger = self.prepare(root)
        args = ("close", "--root", root, "--output", ledger, "--expectation", expectation)
        self.assertEqual(self.command(*args).returncode, 0)
        before = ledger.read_bytes()
        self.assertNotEqual(self.command(*args).returncode, 0)
        self.assertEqual(ledger.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

