from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import build_terminal_manifest


class TerminalManifestTests(unittest.TestCase):
    def test_bulk_selection_boundary(self) -> None:
        self.assertTrue(build_terminal_manifest.is_bulk_prepared_member("prepared_inputs/dual_formal/a.json"))
        self.assertTrue(build_terminal_manifest.is_bulk_prepared_member("prepared_inputs/primary_manifest_view/upstream/raw/x.bin"))
        self.assertFalse(build_terminal_manifest.is_bulk_prepared_member("prepared_inputs/primary_manifest_view/MANIFEST.json"))

    def test_manifest_contains_raw_and_aggregate_bindings(self) -> None:
        value = build_terminal_manifest.build_manifest()
        paths = {row["relative_path"] for row in value["files"]}
        self.assertIn("raw/RUN_COMPLETE.json", paths)
        self.assertIn("raw_attempt_b/RUN_COMPLETE.json", paths)
        self.assertIn("aggregate.json", paths)
        self.assertNotIn("TERMINAL_MANIFEST.json", paths)
        self.assertTrue(value["bulk_prepared_input_bindings"]["primary_manifest_view"]["all_members_verified_before_and_after_copy"])


if __name__ == "__main__":
    unittest.main()

