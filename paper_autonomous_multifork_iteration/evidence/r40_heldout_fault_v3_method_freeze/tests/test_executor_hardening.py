from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.helpers import digest, fake_formal
from v3_common import ContractError
from v3_executor import (
    ExecutionFinalizer, OneShotLocks, _precreate_pending_terminals,
    execute_fixed_campaign, parse_gpu_inventory, rehash_fixed_sources_and_config,
    validate_empty_h20_node,
)
from v3_formal import FaultBinding


def eight_faults():
    return tuple(
        FaultBinding("V3F%02d" % (index + 1), "GPU-test-%d" % index, index)
        for index in range(8)
    )


def gpu_stdout(memory: int = 0, family: str = "NVIDIA H20") -> str:
    return "".join(
        "%d, %s, GPU-test-%d, %d\n" % (index, family, index, memory)
        for index in range(8)
    )


class FakeProcess:
    def __init__(self, pid: int = 12345) -> None:
        self.pid = pid

    def poll(self):
        return None


class ExecutorHardeningTests(unittest.TestCase):
    def test_exact_empty_specified_eight_h20_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            formal = fake_formal(Path(temporary) / "output", eight_faults())
            records = parse_gpu_inventory(gpu_stdout())
            validate_empty_h20_node(records, "", formal)
            for bad_gpu, processes in (
                (gpu_stdout(family="NVIDIA A100"), ""),
                (gpu_stdout(memory=300), ""),
                (gpu_stdout(), "GPU-test-0, 999\n"),
            ):
                with self.assertRaises(ContractError):
                    validate_empty_h20_node(parse_gpu_inventory(bad_gpu), processes, formal)
            wrong = list(records)
            wrong[0] = replace(wrong[0], gpu_uuid="GPU-other")
            with self.assertRaises(ContractError):
                validate_empty_h20_node(wrong, "", formal)

    def test_campaign_global_lock_blocks_changed_config_or_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            formal = fake_formal(parent / "output", eight_faults())
            first = OneShotLocks(formal)
            first.close_descriptors_only()
            changed = replace(
                formal, config_file_sha256=digest("changed-config"),
                output_root=parent / "changed-output",
            )
            with self.assertRaises(FileExistsError):
                OneShotLocks(changed)
            self.assertTrue((parent / ".R40_V3_CAMPAIGN_GLOBAL.lock").is_file())
            self.assertTrue((parent / ".locks" / (formal.config_file_sha256 + ".lock")).is_file())

    def test_all_eight_pending_terminals_exist_before_workers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            output.mkdir()
            formal = fake_formal(output, eight_faults())
            _precreate_pending_terminals(formal, {"method": digest("method")})
            pending = sorted((output / "terminals").glob("*.pending.json"))
            self.assertEqual(len(pending), 8)
            self.assertEqual([path.stem.split(".")[0] for path in pending], list("V3F%02d" % i for i in range(1, 9)))

    def test_single_idempotent_finalizer_kills_groups_and_writes_all_terminals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            output.mkdir()
            formal = fake_formal(output, eight_faults())
            _precreate_pending_terminals(formal, {"method": digest("method")})
            finalizer = ExecutionFinalizer(formal, {"method": digest("method")})
            process = FakeProcess()
            finalizer.register_process(process)
            with patch("v3_executor.os.killpg") as killpg, patch(
                    "v3_executor.rehash_fixed_sources_and_config",
                    return_value={"method": digest("method")}) as rehash:
                finalizer.finalize("signal", 15)
                finalizer.finalize("process_exit", None)
            killpg.assert_called_once_with(process.pid, 9)
            rehash.assert_called_once()
            self.assertEqual(len(list((output / "terminals").glob("*.terminal.json"))), 8)
            self.assertEqual(len(list((output / "terminals").glob("*.pending.json"))), 8)
            self.assertTrue((output / "execution-terminal.json").is_file())

    def test_pre_post_rehash_rejects_formal_config_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            formal = fake_formal(Path(temporary) / "output", eight_faults())
            with patch("v3_executor.load_authority"), patch(
                    "v3_executor.sha256_file", return_value=digest("wrong")):
                with self.assertRaises(ContractError):
                    rehash_fixed_sources_and_config(formal)

    def test_formal_executor_fails_before_any_config_or_gpu_access_without_authorization(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch("v3_executor.load_authority") as authority:
            with self.assertRaises(ContractError):
                execute_fixed_campaign()
            authority.assert_not_called()

    def test_launcher_has_no_config_or_output_arguments_and_source_has_handlers(self) -> None:
        root = Path(__file__).resolve().parents[1]
        launcher = (root / "formal/launch_fixed_h20.sh").read_text()
        self.assertIn('"$#" -ne 0', launcher)
        self.assertNotIn("$1", launcher)
        source = (root / "executed_source/v3_executor.py").read_text()
        self.assertIn("signal.SIGINT", source)
        self.assertIn("signal.SIGTERM", source)
        self.assertIn("atexit.register(finalizer.finalize", source)
        self.assertIn("os.killpg", source)
        self.assertIn("start_new_session=True", source)


if __name__ == "__main__":
    unittest.main()

