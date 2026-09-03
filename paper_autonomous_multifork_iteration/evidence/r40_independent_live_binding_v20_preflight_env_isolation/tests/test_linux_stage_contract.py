from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import stage_v6_clean as stage  # noqa: E402
import verify_frozen_science as frozen_science  # noqa: E402


V6_SHA256 = "306daba7b79b045a306f0b22d6434143dd568cf1f3b6af7114ad1a4ebe1d6f82"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def v6_archive() -> Path:
    configured = os.environ.get("R40_V20_CANONICAL_V6_ARCHIVE")
    candidate = (
        Path(configured)
        if configured
        else ROOT.parent / "r39_primary_compiled_dispatch_v6/packages/r39-primary-compiled-dispatch-20260827f.tar.gz"
    )
    if not candidate.is_file():
        raise AssertionError("canonical v6 archive is mandatory for zero-skip v20 staging tests")
    return candidate.resolve(strict=True)


def overlay_archive() -> Path:
    configured = os.environ.get("R40_V20_OVERLAY_ARCHIVE")
    candidate = Path(configured) if configured else ROOT / "packages/r40-independent-live-binding-v20-preflight-env-isolation-20260828a.tar.gz"
    if not candidate.is_file():
        raise AssertionError("v20 overlay archive is mandatory for zero-skip v20 staging tests")
    return candidate.resolve(strict=True)


def stage_keywords() -> dict[str, object]:
    overlay = overlay_archive()
    return {
        "v6_archive": v6_archive(),
        "overlay_archive": overlay,
        "clean_ledger": ROOT / "v6-clean-members.json",
        "exclusion_ledger": ROOT / "v6-appledouble-exclusions.json",
        "expected_v6_sha256": V6_SHA256,
        "expected_overlay_sha256": digest(overlay),
    }


def write_tar(path: Path, rows: list[tuple[str, str, bytes, int, str | None]]) -> None:
    with tarfile.open(path, "w:gz", format=tarfile.USTAR_FORMAT) as archive:
        for name, kind, data, mode, linkname in rows:
            info = tarfile.TarInfo(name)
            info.mode = mode
            info.uid = info.gid = 0
            info.mtime = 0
            if kind == "regular":
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            elif kind == "directory":
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = linkname or "target"
                archive.addfile(info)
            elif kind == "hardlink":
                info.type = tarfile.LNKTYPE
                info.linkname = linkname or "target"
                archive.addfile(info)
            elif kind == "fifo":
                info.type = tarfile.FIFOTYPE
                archive.addfile(info)
            else:
                raise AssertionError(kind)


class LinuxStageContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v6 = v6_archive()
        cls.overlay = overlay_archive()
        with tarfile.open(cls.v6, "r:gz") as archive:
            member = next(item for item in archive.getmembers() if PurePosixPath(item.name).name.startswith("._"))
            handle = archive.extractfile(member)
            assert handle is not None
            cls.appledouble = handle.read()

    def test_cross_platform_inventory_is_exact_and_science_preserving(self):
        retained, excluded, _ = stage.archive_inventory(self.v6)
        self.assertEqual((len(retained), len(excluded)), (130, 130))
        self.assertEqual({row["size"] for row in excluded}, {163})
        self.assertEqual({row["sha256"] for row in excluded}, {stage.EXPECTED_APPLEDOUBLE_SHA256})
        self.assertEqual(sum(row["type"] == "regular" for row in retained), 98)
        self.assertEqual(sum(row["type"] == "directory" for row in retained), 32)
        retained_paths = {row["path"] for row in retained}
        self.assertTrue(set(stage.REQUIRED_SCIENCE_PATHS) <= retained_paths)
        self.assertTrue(all(row["companion_path"] in retained_paths for row in excluded))

    def test_python_gnu_equivalent_raw_extraction_materializes_v16_blocker(self):
        with tempfile.TemporaryDirectory() as temporary:
            with tarfile.open(self.v6, "r:gz") as archive:
                archive.extractall(temporary)
            apple_paths = [path for path in Path(temporary).rglob("._*") if path.is_file()]
            self.assertEqual(len(apple_paths), 130)
            calibration = Path(temporary) / "paper_autonomous_multifork_iteration/evidence/round_04_rr2_package/executed_source/results/gpu-qwen35-vllm-paged-fair-v2-20260814c"
            exact = {path.relative_to(calibration).as_posix() for path in calibration.rglob("._*")}
            self.assertEqual(
                exact,
                {
                    "._pg19-gate-shards",
                    "._scientific-artifacts.sha256",
                    *(f"pg19-gate-shards/._pg19-fair-v2-shard-{rank}.json" for rank in range(8)),
                },
            )

    def test_frozen_ledgers_are_exact_recomputations(self):
        retained, excluded, _, _, _ = stage.verified_ledgers(
            self.v6, ROOT / "v6-clean-members.json", ROOT / "v6-appledouble-exclusions.json"
        )
        self.assertEqual(stage.clean_ledger_value(retained), stage.load_json_regular(ROOT / "v6-clean-members.json", label="clean")[0])
        self.assertEqual(stage.exclusion_ledger_value(excluded), stage.load_json_regular(ROOT / "v6-appledouble-exclusions.json", label="excluded")[0])

    def test_current_payload_is_self_contained_and_matches_controlled_v18_diff(self):
        opened: list[Path] = []
        real_open = os.open

        def observed_open(path, flags, *args, **kwargs):
            opened.append(Path(path))
            return real_open(path, flags, *args, **kwargs)

        with mock.patch.object(frozen_science.os, "open", side_effect=observed_open):
            receipt = frozen_science.verify_scientific_payload(root=ROOT, repo_root=ROOT.parents[2])
        self.assertEqual(receipt["current_payload_file_count"], 10)
        self.assertEqual(receipt["byte_identical_file_count"], 9)
        self.assertEqual(receipt["controlled_change_file_count"], 1)
        self.assertFalse(receipt["sibling_source_accessed"])
        self.assertFalse(any(path.name == "r40_independent_live_binding_v18_self_contained_stage" for path in opened))

    def test_overlay_has_no_appledouble_link_special_or_unsafe_member(self):
        rows, _ = stage.overlay_inventory(self.overlay, digest(self.overlay))
        self.assertTrue(rows)
        self.assertFalse(any(PurePosixPath(row["path"]).name.startswith("._") for row in rows))
        self.assertEqual({row["type"] for row in rows}, {"regular"})

    def test_prepare_is_nonoverwriting_and_verify_has_zero_appledouble(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve() / "qcomem_r40_v20_clean_20260828a"
            receipt = stage.prepare_stage(output_root=output, **stage_keywords())
            self.assertTrue(receipt["zero_appledouble_paths"])
            self.assertEqual(list(output.rglob("._*")), [])
            before = stage.lexical_stage_tree(output)
            with self.assertRaises(stage.StageContractError):
                stage.prepare_stage(output_root=output, **stage_keywords())
            self.assertEqual(stage.lexical_stage_tree(output), before)
            self.assertEqual(stage.verify_stage(stage_root=output, **stage_keywords()), receipt)

    def test_atomic_directory_publication_refuses_even_empty_existing_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "private-source"
            destination = root / "existing-destination"
            source.mkdir()
            destination.mkdir()
            (source / "science").write_bytes(b"source")
            with self.assertRaises(stage.StageContractError):
                stage.rename_directory_noreplace(source, destination)
            self.assertEqual((source / "science").read_bytes(), b"source")
            self.assertEqual(list(destination.iterdir()), [])

    def test_prepare_dotdot_and_preexisting_nodes_have_zero_side_effect(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            outside = root.parent / f"{root.name}-outside"
            with self.assertRaises(stage.StageContractError):
                stage.prepare_stage(output_root=root / ".." / outside.name, **stage_keywords())
            self.assertFalse(os.path.lexists(outside))
            for kind in ("regular", "directory", "symlink", "fifo"):
                output = root / kind
                if kind == "regular":
                    output.write_bytes(b"sentinel")
                elif kind == "directory":
                    output.mkdir()
                elif kind == "symlink":
                    output.symlink_to(root)
                else:
                    os.mkfifo(output)
                before = os.lstat(output)
                with self.assertRaises((stage.StageContractError, FileNotFoundError)):
                    stage.prepare_stage(output_root=output, **stage_keywords())
                after = os.lstat(output)
                self.assertEqual((before.st_mode, before.st_size, before.st_ino), (after.st_mode, after.st_size, after.st_ino))

    def test_root_symlink_and_symlinked_parent_fail_before_stage_action(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / "stage"
            stage.prepare_stage(output_root=output, **stage_keywords())
            alias = root / "stage-alias"
            alias.symlink_to(output)
            with self.assertRaises(stage.StageContractError):
                stage.verify_stage(stage_root=alias, **stage_keywords())
            parent_alias = root / "parent-alias"
            real_parent = root / "real-parent"
            real_parent.mkdir()
            parent_alias.symlink_to(real_parent)
            with self.assertRaises(stage.StageContractError):
                stage.prepare_stage(output_root=parent_alias / "new-stage", **stage_keywords())
            self.assertFalse(os.path.lexists(real_parent / "new-stage"))

    def test_verify_rejects_extra_regular_directory_symlink_hardlink_fifo_and_appledouble(self):
        for kind in ("regular", "directory", "symlink", "hardlink", "fifo", "appledouble"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary).resolve() / "stage"
                stage.prepare_stage(output_root=output, **stage_keywords())
                bad = output / ("._evil" if kind == "appledouble" else f"extra-{kind}")
                source = output / "paper_autonomous_multifork_iteration/evidence/r40_independent_live_binding_v20_preflight_env_isolation/README.md"
                if kind in {"regular", "appledouble"}:
                    bad.write_bytes(b"extra")
                elif kind == "directory":
                    bad.mkdir()
                elif kind == "symlink":
                    bad.symlink_to(source)
                elif kind == "hardlink":
                    os.link(source, bad)
                else:
                    os.mkfifo(bad)
                with self.assertRaises(stage.StageContractError):
                    stage.verify_stage(stage_root=output, **stage_keywords())

    def test_verify_rejects_retained_byte_mode_and_receipt_tamper(self):
        mutations = ("byte", "mode", "receipt", "receipt-mode")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary).resolve() / "stage"
                stage.prepare_stage(output_root=output, **stage_keywords())
                target = output / "paper_autonomous_multifork_iteration/evidence/round_04_rr2_package/executed_source/results/gpu-qwen35-vllm-paged-fair-v2-20260814c/pg19-gate-shards/pg19-fair-v2-shard-0.json"
                if mutation == "byte":
                    target.write_bytes(target.read_bytes() + b"\n")
                elif mutation == "mode":
                    target.chmod(0o600)
                elif mutation == "receipt":
                    receipt = output / stage.STAGE_RECEIPT
                    receipt.write_bytes(receipt.read_bytes() + b" ")
                else:
                    (output / stage.STAGE_RECEIPT).chmod(0o644)
                with self.assertRaises(stage.StageContractError):
                    stage.verify_stage(stage_root=output, **stage_keywords())

    def test_archive_counterexamples_fail_closed(self):
        root = "paper_autonomous_multifork_iteration/foo"
        valid = [(root, "regular", b"science", 0o644, None), ("paper_autonomous_multifork_iteration/._foo", "regular", self.appledouble, 0o644, None)]
        cases = {
            "absolute": [("/escape", "regular", b"x", 0o644, None)],
            "dotdot": [("../escape", "regular", b"x", 0o644, None)],
            "symlink": [(root, "symlink", b"", 0o777, "target")],
            "hardlink": [(root, "hardlink", b"", 0o644, "target")],
            "fifo": [(root, "fifo", b"", 0o644, None)],
            "false-appledouble": [(root, "regular", b"science", 0o644, None), ("paper_autonomous_multifork_iteration/._foo", "regular", b"not metadata", 0o644, None)],
            "missing-companion": [("paper_autonomous_multifork_iteration/._foo", "regular", self.appledouble, 0o644, None)],
            "mode-mismatch": [(root, "regular", b"science", 0o600, None), ("paper_autonomous_multifork_iteration/._foo", "regular", self.appledouble, 0o644, None)],
            "duplicate": [valid[0], valid[0], valid[1]],
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            for label, rows in cases.items():
                with self.subTest(label=label):
                    archive = directory / f"{label}.tar.gz"
                    write_tar(archive, rows)
                    with mock.patch.multiple(
                        stage,
                        V6_ARCHIVE_SHA256=digest(archive),
                        EXPECTED_ARCHIVE_MEMBERS=len(rows),
                        EXPECTED_RETAINED_MEMBERS=sum(not PurePosixPath(row[0]).name.startswith("._") for row in rows),
                        EXPECTED_EXCLUDED_MEMBERS=sum(PurePosixPath(row[0]).name.startswith("._") for row in rows),
                        REQUIRED_SCIENCE_PATHS=(root,),
                    ):
                        with self.assertRaises(stage.StageContractError):
                            stage.archive_inventory(archive)

    def test_archive_exact_appledouble_pair_succeeds_under_synthetic_contract(self):
        root = "paper_autonomous_multifork_iteration/foo"
        rows = [(root, "regular", b"science", 0o644, None), ("paper_autonomous_multifork_iteration/._foo", "regular", self.appledouble, 0o644, None)]
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary).resolve() / "valid.tar.gz"
            write_tar(archive, rows)
            with mock.patch.multiple(
                stage,
                V6_ARCHIVE_SHA256=digest(archive),
                EXPECTED_ARCHIVE_MEMBERS=2,
                EXPECTED_RETAINED_MEMBERS=1,
                EXPECTED_EXCLUDED_MEMBERS=1,
                REQUIRED_SCIENCE_PATHS=(root,),
            ):
                retained, excluded, _ = stage.archive_inventory(archive)
            self.assertEqual([row["path"] for row in retained], [root])
            self.assertEqual([row["companion_path"] for row in excluded], [root])

    def test_overlay_archive_counterexamples_fail_closed(self):
        root = stage.V20_PACKAGE_NAME
        cases = {
            "unsafe": [("../escape", "regular", b"x", 0o644, None)],
            "appledouble": [(f"{root}/._science.py", "regular", self.appledouble, 0o644, None)],
            "symlink": [(f"{root}/bad", "symlink", b"", 0o777, "target")],
            "hardlink": [(f"{root}/bad", "hardlink", b"", 0o644, "target")],
            "fifo": [(f"{root}/bad", "fifo", b"", 0o644, None)],
            "duplicate": [(f"{root}/science.py", "regular", b"x", 0o644, None)] * 2,
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            for label, rows in cases.items():
                with self.subTest(label=label):
                    archive = directory / f"{label}.tar.gz"
                    write_tar(archive, rows)
                    with self.assertRaises(stage.StageContractError):
                        stage.overlay_inventory(archive, digest(archive))

    def test_self_consistently_resealed_clean_and_exclusion_ledger_tamper_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            for source_name, key in (("v6-clean-members.json", "mode"), ("v6-appledouble-exclusions.json", "size")):
                with self.subTest(source=source_name):
                    source = ROOT / source_name
                    value = json.loads(source.read_text())
                    value["rows"][0][key] += 1
                    value = stage.seal(value)
                    target = directory / source_name
                    target.write_bytes(stage.canonical_json_bytes(value))
                    clean = target if source_name == "v6-clean-members.json" else ROOT / "v6-clean-members.json"
                    exclusion = target if source_name == "v6-appledouble-exclusions.json" else ROOT / "v6-appledouble-exclusions.json"
                    with self.assertRaises(stage.StageContractError):
                        stage.verified_ledgers(self.v6, clean, exclusion)


if __name__ == "__main__":
    unittest.main()
