from __future__ import annotations

import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from typing import Any, Mapping

import psutil


_BYTE_UNITS = {
    "K": 1024,
    "M": 1024**2,
    "G": 1024**3,
    "T": 1024**4,
}


def _run(*args: str, timeout: float = 20) -> str | None:
    try:
        result = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def parse_battery_status(output: str | None) -> dict[str, Any]:
    """Parse the stable parts of ``pmset -g batt`` output."""

    output = output or ""
    source_match = re.search(r"Now drawing from '([^']+)'", output)
    percent_match = re.search(r"\b(\d{1,3})%;", output)
    state_match = re.search(
        r"\b(charging|discharging|charged|finishing charge|not charging)\b",
        output,
        flags=re.IGNORECASE,
    )
    source_name = source_match.group(1) if source_match else None
    normalized_source = None
    if source_name:
        normalized_source = (
            "ac" if "AC" in source_name or "UPS" in source_name else "battery"
        )
    return {
        "source": normalized_source,
        "source_name": source_name,
        "battery_percent": int(percent_match.group(1)) if percent_match else None,
        "battery_state": state_match.group(1).lower() if state_match else None,
        "raw": output,
    }


def parse_power_profiles(output: str | None) -> dict[str, dict[str, int]]:
    """Parse numeric settings in each section of ``pmset -g custom``."""

    profiles: dict[str, dict[str, int]] = {}
    current: str | None = None
    for raw_line in (output or "").splitlines():
        line = raw_line.strip()
        if line.endswith(":") and not raw_line.startswith((" ", "\t")):
            current = line[:-1]
            profiles[current] = {}
            continue
        if current is None:
            continue
        match = re.match(r"([A-Za-z0-9_]+)\s+(-?\d+)$", line)
        if match:
            profiles[current][match.group(1)] = int(match.group(2))
    return profiles


def active_power_mode(
    battery: Mapping[str, Any], profiles: Mapping[str, Mapping[str, int]]
) -> int | None:
    section = "AC Power" if battery.get("source") == "ac" else "Battery Power"
    value = profiles.get(section, {}).get("powermode")
    return int(value) if value is not None else None


def parse_swap_bytes(output: str | None) -> dict[str, int | None]:
    values: dict[str, int | None] = {"total": None, "used": None, "free": None}
    for name, number, unit in re.findall(
        r"(total|used|free)\s*=\s*([0-9.]+)([KMGT])", output or ""
    ):
        values[name] = round(float(number) * _BYTE_UNITS[unit])
    return values


def parse_thermal_output(output: str | None) -> dict[str, Any]:
    lines = [line.strip() for line in (output or "").splitlines() if line.strip()]
    warnings = [
        line
        for line in lines
        if "warning" in line.lower()
        and not line.lower().startswith("note: no ")
    ]
    return {"warnings": warnings, "raw": output}


def _thermal_state() -> dict[str, Any]:
    raw = _run(
        "/usr/bin/swift",
        "-e",
        "import Foundation; print(ProcessInfo.processInfo.thermalState.rawValue)",
        timeout=30,
    )
    labels = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}
    try:
        value = int(raw) if raw is not None else None
    except ValueError:
        value = None
    return {"value": value, "label": labels.get(value, "unknown")}


def _top_processes(limit: int = 8) -> list[dict[str, Any]]:
    output = _run(
        "ps",
        "-A",
        "-o",
        "pid=,comm=,%cpu=,rss=",
        "-r",
    )
    rows = []
    for line in (output or "").splitlines()[:limit]:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        try:
            rows.append(
                {
                    "pid": int(parts[0]),
                    "command": " ".join(parts[1:-2]),
                    "cpu_percent": float(parts[-2]),
                    "rss_bytes": int(parts[-1]) * 1024,
                }
            )
        except ValueError:
            continue
    return rows


def collect_experiment_snapshot() -> dict[str, Any]:
    """Collect conditions that can materially change laptop benchmark results."""

    battery = parse_battery_status(_run("pmset", "-g", "batt"))
    power_profiles_raw = _run("pmset", "-g", "custom")
    power_profiles = parse_power_profiles(power_profiles_raw)
    mode = active_power_mode(battery, power_profiles)
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=0.25)
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "macos_version": platform.mac_ver()[0],
        "power": {
            **battery,
            "active_power_mode": mode,
            "low_power_mode": mode == 0 if mode is not None else None,
            "profiles": power_profiles,
            "profiles_raw": power_profiles_raw,
        },
        "thermal_state": _thermal_state(),
        "pmset_thermal": parse_thermal_output(_run("pmset", "-g", "therm")),
        "swap_bytes": parse_swap_bytes(_run("sysctl", "vm.swapusage")),
        "memory": {
            "total_bytes": memory.total,
            "available_bytes": memory.available,
            "percent": memory.percent,
        },
        "load_average": list(os.getloadavg()),
        "cpu_percent": cpu_percent,
        "top_processes": _top_processes(),
    }


def assess_preflight(
    snapshot: Mapping[str, Any],
    *,
    max_cpu_percent: float = 35.0,
) -> dict[str, Any]:
    """Return eligibility independent of whether the caller enforces it."""

    reasons: list[str] = []
    power = snapshot.get("power", {})
    if power.get("source") != "ac":
        reasons.append("not_connected_to_ac_power")
    if power.get("low_power_mode") is True:
        reasons.append("low_power_mode_enabled")
    thermal = snapshot.get("thermal_state", {})
    if thermal.get("value") not in (0, None):
        reasons.append(f"thermal_state_{thermal.get('label', 'unknown')}")
    if snapshot.get("pmset_thermal", {}).get("warnings"):
        reasons.append("pmset_reported_performance_or_thermal_warning")
    cpu_percent = snapshot.get("cpu_percent")
    if cpu_percent is not None and float(cpu_percent) > max_cpu_percent:
        reasons.append("background_cpu_above_limit")
    return {
        "formal_result_eligible": not reasons,
        "reasons": reasons,
        "limits": {"max_preflight_cpu_percent": max_cpu_percent},
    }


def assess_completed_run(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    max_cpu_percent: float = 35.0,
    max_swap_growth_bytes: int = 128 * 1024**2,
) -> dict[str, Any]:
    assessment = assess_preflight(before, max_cpu_percent=max_cpu_percent)
    reasons = list(assessment["reasons"])
    after_thermal = after.get("thermal_state", {})
    if after_thermal.get("value") not in (0, None):
        reasons.append(f"post_run_thermal_state_{after_thermal.get('label', 'unknown')}")
    if after.get("pmset_thermal", {}).get("warnings"):
        reasons.append("post_run_pmset_warning")
    before_swap = before.get("swap_bytes", {}).get("used")
    after_swap = after.get("swap_bytes", {}).get("used")
    swap_growth = None
    if before_swap is not None and after_swap is not None:
        swap_growth = int(after_swap) - int(before_swap)
        if swap_growth > max_swap_growth_bytes:
            reasons.append("swap_growth_above_limit")
    return {
        "formal_result_eligible": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "limits": {
            "max_preflight_cpu_percent": max_cpu_percent,
            "max_swap_growth_bytes": max_swap_growth_bytes,
        },
        "observed_swap_growth_bytes": swap_growth,
    }
