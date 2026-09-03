# CPU-only reproduction commands

Run from this directory:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_local_validation.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/freeze_method.py
```

Both commands are non-overwriting.  They do not invoke GPU, QS, SSH, or formal
execution.  After `METHOD_FROZEN.json` exists, changes require a new versioned
directory and a new fresh-audit/fresh-designer sequence.

