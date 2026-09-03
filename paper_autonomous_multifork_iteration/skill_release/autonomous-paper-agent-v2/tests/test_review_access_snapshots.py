from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_review_snapshot.py"


class ReviewAccessSnapshotTest(unittest.TestCase):
    def test_pdf_only_view_excludes_repository_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paper = root / "paper.pdf"
            rubric = root / "rubric.md"
            evidence = root / "private_evidence.json"
            paper.write_bytes(b"%PDF-1.4\nreview fixture\n")
            rubric.write_text("rubric", encoding="utf-8")
            evidence.write_text('{"verified": true}\n', encoding="utf-8")
            output = root / "review"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--round",
                    "3",
                    "--paper",
                    str(paper),
                    "--pdf-only-include",
                    str(rubric),
                    "--include",
                    str(evidence),
                    "--output-root",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            submission = output / "round_03" / "submission"
            pdf_only = submission / "pdf_only"
            artifact = submission / "pdf_plus_repository"
            self.assertFalse(any(path.name == evidence.name for path in pdf_only.rglob("*")))
            self.assertTrue(any(path.name == evidence.name for path in artifact.rglob("*")))
            self.assertEqual(
                (pdf_only / "manuscript" / paper.name).read_bytes(),
                (artifact / "manuscript" / paper.name).read_bytes(),
            )
            self.assertEqual(
                payload["default_panel_access_mix"],
                {"pdf_only": 3, "pdf_plus_repository": 2},
            )
            manifests = [
                json.loads((view / "MANIFEST.json").read_text(encoding="utf-8"))
                for view in (pdf_only, artifact)
            ]
            self.assertEqual(
                [manifest["access_mode"] for manifest in manifests],
                ["pdf_only", "pdf_plus_repository"],
            )
            manifest_paths = [
                submission / "MANIFEST.json",
                pdf_only / "MANIFEST.json",
                artifact / "MANIFEST.json",
            ]
            for manifest_path in manifest_paths:
                self.assertNotIn(str(root), manifest_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
