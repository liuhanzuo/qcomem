from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "source-code.sha256"
INCLUDED = [
    "README.md",
    "DESIGN.md",
    "preregistration.json",
    "acceptance.json",
    "formal-blocker.json",
    "absorbed-lineage.json",
    "local-static-audit.json",
    "local-validation.json",
    "package-manifest.json",
    "linux-stage-regression.json",
    "v6-clean-members.json",
    "v6-appledouble-exclusions.json",
    "v19-current-payload.sha256",
    "v18-v19-controlled-diff.json",
]
INCLUDED += [str(path.relative_to(ROOT)) for folder in ("executed_source", "formal", "scripts", "tests") for path in sorted((ROOT / folder).glob("*")) if path.is_file() and path.name != "source-code.sha256"]


def main() -> int:
    lines = []
    for relative in sorted(INCLUDED):
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative}")
    with OUTPUT.open("x", encoding="ascii", errors="strict") as stream:
        stream.write("\n".join(lines) + "\n")
    print(hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
