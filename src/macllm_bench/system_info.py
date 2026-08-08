from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import psutil


def _run(*args: str) -> str | None:
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def collect() -> dict[str, object]:
    hardware = _run("system_profiler", "SPHardwareDataType", "-json")
    power = _run("pmset", "-g", "custom")
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "macos_version": platform.mac_ver()[0],
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "unified_memory_bytes": psutil.virtual_memory().total,
        "hardware": json.loads(hardware) if hardware else None,
        "power_settings": power,
    }


def main() -> None:
    payload = collect()
    output = Path("results/system_info.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
