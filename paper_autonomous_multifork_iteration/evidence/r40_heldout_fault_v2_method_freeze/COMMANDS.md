# Reproduction commands

Run from this directory with the system Python.  Neither command contacts a
cluster or invokes a GPU.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_local_validation.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/freeze_method.py
```

The first command is non-overwriting and emits the complete local test/static
audit records.  The second refuses to run unless validation passed, then emits
the source ledger, freeze record, deterministic archive, package manifest, and
terminal hashes.  Once `METHOD_FROZEN.json` exists, no file covered by
`source-code.sha256` may be edited; a changed method requires a new versioned
directory and a fresh designer separation process.

