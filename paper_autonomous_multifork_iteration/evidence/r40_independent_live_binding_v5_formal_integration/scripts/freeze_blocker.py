from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "source-code.sha256"

def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError("refusing to overwrite blocker ledger")
    files = [ROOT / name for name in ("README.md","acceptance.json","blocker.json","tests/test_blocker.py","scripts/freeze_blocker.py")]
    OUTPUT.write_text("\n".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT)}" for path in files) + "\n")
    print(hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
