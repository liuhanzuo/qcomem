from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def run_command(
    command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path
) -> dict[str, Any]:
    started = dt.datetime.now(dt.timezone.utc)
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=900,
    )
    ended = dt.datetime.now(dt.timezone.utc)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise FileExistsError(f"refusing to overwrite {log_path}")
    log_path.write_text(
        completed.stdout + ("\n[stderr]\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    receipt = {
        "command": command,
        "started_at_utc": started.isoformat(),
        "completed_at_utc": ended.isoformat(),
        "returncode": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
        "log_path": str(log_path),
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    (output / "artifacts").mkdir(parents=True)
    (output / "logs").mkdir()
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(root)
    prereg = root / "preregistration.json"
    ledger = root / "frozen/source_ledger.json"
    commands: list[dict[str, Any]] = []
    commands.append(
        run_command(
            [
                sys.executable,
                str(root / "scripts/verify_frozen_sources.py"),
                "--root",
                str(root),
                "--ledger",
                str(ledger),
            ],
            cwd=root,
            env=env,
            log_path=output / "logs/source-verification.log",
        )
    )
    commands.append(
        run_command(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(root / "tests"),
                "-p",
                "test_*.py",
                "-v",
            ],
            cwd=root,
            env=env,
            log_path=output / "logs/unit-tests.log",
        )
    )
    campaign_path = output / "artifacts/campaign-result.json"
    commands.append(
        run_command(
            [
                sys.executable,
                str(root / "scripts/run_campaign.py"),
                "--root",
                str(root),
                "--preregistration",
                str(prereg),
                "--source-ledger",
                str(ledger),
                "--output",
                str(campaign_path),
            ],
            cwd=root,
            env=env,
            log_path=output / "logs/campaign.log",
        )
    )
    verification_path = output / "artifacts/verification.json"
    commands.append(
        run_command(
            [
                sys.executable,
                str(root / "scripts/verify_campaign.py"),
                "--root",
                str(root),
                "--preregistration",
                str(prereg),
                "--source-ledger",
                str(ledger),
                "--campaign",
                str(campaign_path),
                "--output",
                str(verification_path),
            ],
            cwd=root,
            env=env,
            log_path=output / "logs/independent-replay.log",
        )
    )
    receipt = {
        "schema_version": "forkaudit-r40-independent-live-binding-execution-receipt-v1",
        "experiment_id": "R40-INDEPENDENT-LIVE-BINDING-20260827A",
        "status": "completed",
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "gpu_or_qs_resource_touched": False,
        },
        "preregistration_sha256": sha256_file(prereg),
        "source_ledger_sha256": sha256_file(ledger),
        "campaign_raw_sha256": sha256_file(campaign_path),
        "verification_raw_sha256": sha256_file(verification_path),
        "commands": commands,
    }
    write_json_new(output / "artifacts/execution-receipt.json", receipt)
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "terminal-files.sha256"
    )
    ledger_lines = [
        f"{sha256_file(path)}  {path.relative_to(output).as_posix()}" for path in files
    ]
    terminal = output / "terminal-files.sha256"
    terminal.write_text("\n".join(ledger_lines) + "\n", encoding="utf-8")
    print(f"PASS output={output}")
    print(f"campaign_sha256={receipt['campaign_raw_sha256']}")
    print(f"verification_sha256={receipt['verification_raw_sha256']}")
    print(f"terminal_ledger_sha256={sha256_file(terminal)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

