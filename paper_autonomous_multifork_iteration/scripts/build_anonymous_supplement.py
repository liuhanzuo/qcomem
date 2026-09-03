#!/usr/bin/env python3
"""Build a self-contained offline-replay, double-blind derivative of the evidence.

The original run mirror remains untouched.  This builder copies the exact
executed source closure, redacts identifying path/cluster metadata from JSON
evidence, rebinds the redacted shard hashes in a derivative summary, runs the
CPU governance suite from the copied closure, and emits a relative-path
SHA-256 manifest.  Original byte digests are retained without original paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
GPU = REPO / "gpu"
PRIMARY = REPO / "results/gpu-qwen35-vllm-paged-multifork-resident-20260814a"
FAIR = REPO / "results/gpu-qwen35-vllm-paged-fair-v2-20260814c/fair-v2-summary.json"
NEGATIVE = REPO / "results/gpu-qwen35-vllm-paged-q16-formal-negative-20260814b/failure-summary.json"

CODE_FILES = [
    "qcomem_joint_policy.py",
    "qcomem_paged_attention.py",
    "qcomem_qwen35_functional_stack.py",
    "qcomem_qwen35_gdn_functional.py",
    "qcomem_qwen35_native_cache.py",
    "qcomem_qwen35_paged_integration.py",
    "qcomem_qwen35_vllm_paged_integration.py",
    "qcomem_torch.py",
    "qcomem_vllm_paged_fair_control.py",
    "qcomem_vllm_paged_kernel.py",
    "qcomem_vllm_paged_multifork_resident.py",
    "run_downstream.py",
    "run_qcomem_qwen35_vllm_paged_multifork_resident.py",
    "test_launch_qcomem_qwen35_vllm_paged_multifork_resident.py",
    "test_qcomem_vllm_paged_multifork_resident.py",
    "test_run_qcomem_qwen35_vllm_paged_multifork_resident.py",
    "launch_qcomem_qwen35_vllm_paged_multifork_resident_8gpu.sh",
    "MULTIFORK_RESIDENT_PROTOCOL_ZH.md",
]

TEXT_REPLACEMENTS = [
    (re.compile(r"/Users/[^/\s\"']+"), "/home/anonymous"),
    (re.compile(r"/mnt/tidal-alsh-hilab/dataset/diandian/user/[^/\s\"']+"), "/mnt/anonymous"),
    (re.compile(r"https://qs2\.devops\.[^/\s\"']+/[^\s\"']+"), "<INTERNAL_JOB_URL_REDACTED>"),
    (re.compile(r"\bliuhanzuo\b", re.IGNORECASE), "anonymous-user"),
    (re.compile(r"\bxiaohongshu\b", re.IGNORECASE), "anonymous-organization"),
    (re.compile(r"\b(?:Job\s*)?237580\b", re.IGNORECASE), "JOB_REDACTED"),
    (re.compile(r"\b(?:Trial\s*)?1840837\b", re.IGNORECASE), "TRIAL_REDACTED"),
    (re.compile(r"\b(?:Job\s*)?237468\b", re.IGNORECASE), "SUPPORT_JOB_REDACTED"),
    (re.compile(r"\b(?:Trial\s*)?1840486\b", re.IGNORECASE), "SUPPORT_TRIAL_REDACTED"),
    (re.compile(r"\b(?:Job\s*)?237281\b", re.IGNORECASE), "NEGATIVE_JOB_REDACTED"),
    (re.compile(r"\b(?:Trial\s*)?1840009\b", re.IGNORECASE), "NEGATIVE_TRIAL_REDACTED"),
    (re.compile(r"\bqueue436\b", re.IGNORECASE), "QUEUE_REDACTED"),
    (re.compile(r"\bcluster53\b", re.IGNORECASE), "CLUSTER_REDACTED"),
]

FORBIDDEN_SCAN = [
    re.compile(r"/Users/[^/\s\"']+"),
    re.compile(r"/mnt/tidal-alsh-hilab/dataset/diandian/user/[^/\s\"']+"),
    re.compile(r"qs2\.devops", re.IGNORECASE),
    re.compile(r"liuhanzuo", re.IGNORECASE),
    re.compile(r"xiaohongshu", re.IGNORECASE),
]

SENSITIVE_METADATA_KEYS = {
    "job_id",
    "trial_id",
    "queue_id",
    "cluster_id",
    "package_id",
    "pod_name",
    "node_name",
    "namespace",
    "run_url",
    "job_url",
    "trial_url",
    "web_ui_url",
}
REDACTED_METADATA_VALUE = "<REDACTED>"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize_text(value: str) -> str:
    for pattern, replacement in TEXT_REPLACEMENTS:
        value = pattern.sub(replacement, value)
    return value


def sanitize_json(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    if isinstance(value, dict):
        return {
            key: (
                REDACTED_METADATA_VALUE
                if key.lower() in SENSITIVE_METADATA_KEYS
                else sanitize_json(item)
            )
            for key, item in value.items()
        }
    return value


def sensitive_metadata_violations(value: Any, pointer: str = "$") -> list[str]:
    violations: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{pointer}.{key}"
            if key.lower() in SENSITIVE_METADATA_KEYS and item != REDACTED_METADATA_VALUE:
                violations.append(child)
            violations.extend(sensitive_metadata_violations(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            violations.extend(sensitive_metadata_violations(item, f"{pointer}[{index}]"))
    return violations


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_expected_code_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for line in (PRIMARY / "code.sha256").read_text(encoding="utf-8").splitlines():
        digest, original_path = line.split("  ", 1)
        result[Path(original_path).name] = digest
    return result


def original_code_ledger_record() -> dict[str, Any]:
    ledger = PRIMARY / "code.sha256"
    lines = ledger.read_text(encoding="utf-8").splitlines()
    return {
        "sha256": sha256(ledger),
        "bytes": ledger.stat().st_size,
        "entry_count": len(lines),
        "disclosure": (
            "The original absolute-path ledger is not redistributed. Each copied "
            "closure file is verified against its corresponding digest; the "
            "path-normalized 18-file derivative ledger has a different file hash."
        ),
    }


def copy_code_closure(output: Path) -> list[dict[str, Any]]:
    expected = load_expected_code_hashes()
    records = []
    code_dir = output / "code"
    code_dir.mkdir(parents=True)
    for name in CODE_FILES:
        source = GPU / name
        destination = code_dir / name
        actual = sha256(source)
        if expected.get(name) != actual:
            raise RuntimeError(f"executed source hash drift: {name}")
        shutil.copyfile(source, destination)
        records.append({"file": f"code/{name}", "sha256": actual, "bytes": destination.stat().st_size})
    ledger = "".join(f"{row['sha256']}  {Path(row['file']).name}\n" for row in records)
    (code_dir / "EXECUTED_SOURCE_SHA256").write_text(ledger, encoding="utf-8")
    return records


def copy_redacted_primary(output: Path) -> list[dict[str, Any]]:
    destination = output / "raw_primary"
    shard_dir = destination / "resident-shards"
    shard_dir.mkdir(parents=True)
    originals: list[dict[str, Any]] = []
    redacted_shards: list[dict[str, Any]] = []
    for source in sorted((PRIMARY / "resident-shards").glob("multifork-resident-shard-*.json")):
        originals.append({"role": "primary_raw_shard", "file": source.name,
                          "sha256": sha256(source), "bytes": source.stat().st_size})
        target = shard_dir / source.name
        write_json(target, sanitize_json(json.loads(source.read_text(encoding="utf-8"))))
        redacted_shards.append({
            "path": f"resident-shards/{source.name}",
            "sha256": sha256(target),
            "bytes": target.stat().st_size,
        })

    source_summary = PRIMARY / "multifork-resident-summary.json"
    originals.append({"role": "primary_summary", "file": source_summary.name,
                      "sha256": sha256(source_summary), "bytes": source_summary.stat().st_size})
    summary = sanitize_json(json.loads(source_summary.read_text(encoding="utf-8")))
    if "raw_shard_artifacts" not in summary:
        raise RuntimeError("primary summary has no raw_shard_artifacts binding")
    summary["raw_shard_artifacts"] = redacted_shards
    summary["anonymous_derivative"] = {
        "redaction_only": True,
        "numeric_and_boolean_measurements_unchanged": True,
        "raw_shard_hashes_rebound_after_metadata_redaction": True,
        "original_byte_digests": "../provenance/original_evidence_digests.json",
    }
    write_json(destination / source_summary.name, summary)

    static = PRIMARY / "static-dry-run.json"
    originals.append({"role": "primary_static", "file": static.name,
                      "sha256": sha256(static), "bytes": static.stat().st_size})
    write_json(destination / static.name,
               sanitize_json(json.loads(static.read_text(encoding="utf-8"))))
    return originals


def copy_supporting_summaries(output: Path) -> list[dict[str, Any]]:
    destination = output / "raw_supporting"
    destination.mkdir(parents=True)
    records = []
    for role, source, target_name in (
        ("same_kernel_single_request_summary", FAIR, "fair-v2-summary.json"),
        ("cross_backend_negative_summary", NEGATIVE, "cross-backend-negative-summary.json"),
    ):
        records.append({"role": role, "file": target_name,
                        "sha256": sha256(source), "bytes": source.stat().st_size})
        write_json(destination / target_name,
                   sanitize_json(json.loads(source.read_text(encoding="utf-8"))))
    return records


def run_governance_tests(output: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "unittest",
        "-v",
        "test_qcomem_vllm_paged_multifork_resident",
        "test_run_qcomem_qwen35_vllm_paged_multifork_resident",
        "test_launch_qcomem_qwen35_vllm_paged_multifork_resident",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(output / "code")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        command,
        cwd=output / "code",
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
        check=False,
    )
    log = output / "provenance/governance_tests.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "Command: python -m unittest -v "
        "test_qcomem_vllm_paged_multifork_resident "
        "test_run_qcomem_qwen35_vllm_paged_multifork_resident "
        "test_launch_qcomem_qwen35_vllm_paged_multifork_resident\n"
        f"Exit code: {process.returncode}\n\n{sanitize_text(process.stdout)}",
        encoding="utf-8",
    )
    if process.returncode != 0 or "Ran 17 tests" not in process.stdout or "OK" not in process.stdout:
        raise RuntimeError("governance suite failed in anonymous source closure")


def write_readme(output: Path) -> None:
    (output / "README.md").write_text(
        """# ForkAudit anonymous supplementary evidence

This directory is a deterministic, double-blind derivative of the frozen
research artifacts. The original run mirror is not modified.

- `code/` contains the 18 source/protocol files needed for offline replay and
  the 17 CPU governance tests. Every copied file matches a published per-file
  digest from the executed source set; the private 135-entry absolute-path
  ledger preimage is not redistributed, so completeness against that ledger is
  not independently reconstructible here.
- `raw_primary/` contains the aggregate, eight raw shards, and static audit.
  Identifying filesystem, cluster, and private-service metadata was redacted;
  the derivative summary rebinds the resulting shard hashes. Numeric values,
  digests of model outputs/state, and pass/fail booleans are unchanged.
- `raw_supporting/` contains the same-kernel single-request control summary and
  the cross-backend negative-gate summary under the same metadata redaction.
- `provenance/original_evidence_digests.json` binds the unredacted source bytes
  by role, basename, size, and SHA-256 without redistributing identifying paths.
  It also binds the complete original execution code-ledger file hash while
  explaining why the path-normalized derivative ledger has a different hash.
- `provenance/governance_tests.log` records a clean run from the copied source
  closure. Model weights and PG-19 text are not redistributed.

Deep-replay the raw cross-arm and cross-fan-out relations, storage/accounting
equations, and paper figures/tables from the anonymous primary evidence with:

```bash
python scripts/generate_paper_artifacts.py \\
  --results supplement_anonymous/raw_primary \\
  --output-root /tmp/forkaudit-replay
```

The evidence stores full-tensor SHA-256 digests and runtime `torch.equal`
booleans, not the full logits/KV/GDN tensors. Independent numerical replay
therefore still requires the pinned model and PG-19 inputs. This package is an
offline evidence-replay artifact, not a clean GPU-rerun artifact. The original
code ledger used absolute private paths and is not redistributed; the copied
source files are checked against their published per-file digests, while the
relative-path derivative ledger necessarily has a different whole-file hash.
""",
        encoding="utf-8",
    )


def scan_anonymity(output: Path) -> None:
    checked = 0
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        checked += 1
        for pattern in FORBIDDEN_SCAN:
            if pattern.search(text):
                raise RuntimeError(f"anonymity scan failed: {path.relative_to(output)}")
        if path.suffix.lower() == ".json":
            value = json.loads(text)
            violations = sensitive_metadata_violations(value)
            if violations:
                raise RuntimeError(
                    "sensitive metadata was not redacted: "
                    f"{path.relative_to(output)}:{','.join(violations)}"
                )
    write_json(output / "provenance/anonymization_audit.json", {
        "status": "passed",
        "text_files_checked_before_manifest": checked,
        "checks": [
            "no personal home-directory path",
            "no private remote-user path",
            "no private job-service hostname",
            "no personal username token",
            "no organization-name token",
            "all sensitive infrastructure metadata values are <REDACTED>",
        ],
    })


def write_manifest(output: Path) -> None:
    paths = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name != "ANONYMOUS_MANIFEST.sha256"
    )
    (output / "ANONYMOUS_MANIFEST.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output).as_posix()}\n" for path in paths),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PAPER / "supplement_anonymous")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == Path("/") or PAPER not in output.parents:
        raise RuntimeError("output must be a generated directory inside the paper workspace")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    code_records = copy_code_closure(output)
    evidence_records = copy_redacted_primary(output) + copy_supporting_summaries(output)
    write_json(output / "provenance/original_evidence_digests.json", {
        "schema_version": "1.0.0",
        "code": code_records,
        "original_execution_code_ledger": original_code_ledger_record(),
        "evidence": evidence_records,
        "note": "Original paths are intentionally omitted from the anonymous derivative.",
    })
    (output / "scripts").mkdir(parents=True)
    shutil.copyfile(PAPER / "scripts/generate_paper_artifacts.py",
                    output / "scripts/generate_paper_artifacts.py")
    write_readme(output)
    run_governance_tests(output)
    scan_anonymity(output)
    write_manifest(output)
    # Scan once more including the audit and manifest themselves.
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in FORBIDDEN_SCAN:
            if pattern.search(text):
                raise RuntimeError(f"post-manifest anonymity scan failed: {path.relative_to(output)}")
        if path.suffix.lower() == ".json":
            violations = sensitive_metadata_violations(json.loads(text))
            if violations:
                raise RuntimeError(
                    "post-manifest sensitive metadata scan failed: "
                    f"{path.relative_to(output)}:{','.join(violations)}"
                )

    print(json.dumps({
        "output": str(output),
        "files": sum(path.is_file() for path in output.rglob("*")),
        "bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
        "manifest_sha256": sha256(output / "ANONYMOUS_MANIFEST.sha256"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
