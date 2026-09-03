#!/usr/bin/env bash
set -euo pipefail

AUDIT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$AUDIT_ROOT"
export PYTHONDONTWRITEBYTECODE=1

python3 scripts/verify_hashes.py
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 probes/probe_v4_counterexamples.py --format text | diff -u raw/probe.stdout.txt -
python3 probes/probe_v4_counterexamples.py --format json | diff -u raw/probe.results.json -

if find . -type d -name '__pycache__' -print | grep -q .; then
  echo "unexpected __pycache__ directory" >&2
  exit 1
fi

printf '%s\n' 'R40_V4_POSTHOC_REPLAY=PASS'
