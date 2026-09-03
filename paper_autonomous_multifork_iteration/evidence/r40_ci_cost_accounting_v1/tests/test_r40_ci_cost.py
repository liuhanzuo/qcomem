from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import aggregate_results
import audit_inventory
import measure_replays


class R40CostTests(unittest.TestCase):
    def test_frozen_component_order(self) -> None:
        self.assertEqual(len(measure_replays.COMPONENTS), 6)
        self.assertEqual(measure_replays.COMPONENTS[0], "primary_rr2")
        self.assertNotIn("falcon", " ".join(measure_replays.COMPONENTS))

    def test_parse_darwin_time_report(self) -> None:
        payload = (
            "        1.25 real         1.00 user         0.20 sys\n"
            "             1234567  maximum resident set size\n"
            "              765432  peak memory footprint\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "time.txt"
            path.write_text(payload, encoding="utf-8")
            value = measure_replays.parse_time_report(path)
        self.assertEqual(value["maximum_resident_set_size_raw"], 1234567)
        self.assertEqual(value["maximum_resident_set_size_unit"], "bytes_on_darwin")
        self.assertEqual(value["time_real_seconds_rounded"], 1.25)

    def test_safe_extract_rejects_parent_member(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                info = tarfile.TarInfo("../escape")
                data = b"x"
                info.size = len(data)
                handle.addfile(info, io.BytesIO(data))
            with self.assertRaises(measure_replays.MeasurementError):
                measure_replays.safe_extract(archive, root / "out")

    def test_statistics_are_exact_for_three_rows(self) -> None:
        value = aggregate_results.stats([3, 1, 2])
        self.assertEqual(value, {"count": 3.0, "minimum": 1.0, "median": 2.0, "maximum": 3.0})

    def test_inventory_marks_non_measurements(self) -> None:
        value = audit_inventory.build_inventory()
        falcon = value["packages"]["r39_falcon_h1_v2"]
        blind = value["packages"]["r39_pdf_only_blind_faults"]
        self.assertEqual(falcon["entrypoints"][0]["status"], "blocked_unmeasured")
        self.assertEqual(blind["entrypoints"][0]["status"], "blocked_unmeasured_full_replay")
        self.assertIn("gpu_perturbation_or_slowdown", value["unmeasured"])

    def test_protocol_forbids_h20_overhead_substitution(self) -> None:
        text = (ROOT / "PROTOCOL.md").read_text(encoding="utf-8")
        self.assertIn("does not measure H20 capture overhead", text)
        self.assertIn("engineering effort", text)
        self.assertIn("Falcon-H1 v2", text)


if __name__ == "__main__":
    unittest.main()

