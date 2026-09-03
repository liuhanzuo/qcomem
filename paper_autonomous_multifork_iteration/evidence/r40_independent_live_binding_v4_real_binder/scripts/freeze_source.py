from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "source-code.sha256"
INCLUDED = ["README.md", "DESIGN.md", "preregistration.json", "acceptance.json", "local-static-audit.json"]
INCLUDED += [str(path.relative_to(ROOT)) for folder in ("executed_source", "scripts", "tests") for path in sorted((ROOT / folder).glob("*")) if path.is_file() and path.name != "source-code.sha256"]


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    lines = []
    for relative in sorted(INCLUDED):
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative}")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
