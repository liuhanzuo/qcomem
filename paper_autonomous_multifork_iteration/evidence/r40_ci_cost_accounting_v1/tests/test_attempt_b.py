from __future__ import annotations

from pathlib import Path, PurePosixPath
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import measure_replays_attempt_b as attempt_b


class AttemptBTests(unittest.TestCase):
    def test_manifest_path_validation(self) -> None:
        self.assertEqual(attempt_b.validate_relative_path("replay/run_replay.sh"), PurePosixPath("replay/run_replay.sh"))
        for value in ("", ".", "/absolute", "../escape", "a/../escape"):
            with self.assertRaises(attempt_b.base.MeasurementError):
                attempt_b.validate_relative_path(value)

    def test_copy_exclusive_preserves_bytes_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "nested/destination"
            source.write_bytes(b"manifest-bound\x00bytes")
            attempt_b.copy_exclusive(source, destination)
            self.assertEqual(source.read_bytes(), destination.read_bytes())
            with self.assertRaises(FileExistsError):
                attempt_b.copy_exclusive(source, destination)

    def test_attempt_b_isolated_outputs(self) -> None:
        self.assertNotEqual(attempt_b.RAW, attempt_b.HERE / "raw")
        self.assertEqual(attempt_b.RAW.name, "raw_attempt_b")
        self.assertIn("prepared_inputs", attempt_b.CLEAN_PRIMARY.parts)

    def test_exact_three_repetitions_remains_frozen(self) -> None:
        with self.assertRaises(attempt_b.base.MeasurementError):
            attempt_b.run_profiles(2)


if __name__ == "__main__":
    unittest.main()

