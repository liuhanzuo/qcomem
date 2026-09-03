# V28 published phase-artifact receipt path

V28 changes one pathname binding in the real-binding hook. Scientific runner
inputs, model inputs, measurement code, ownership verifier predicates,
producer receipt equalities, allocator measurement, Python invocation hygiene,
and terminal closure rules are unchanged.

## Failure mechanism

The immutable runner writes each rank into
`primary/raw/.forkaudit-rank-X-*/rank-X/...`, atomically renames the inner
`rank-X` directory to `primary/raw/rank-X`, and removes the staging directory.
V27 validated the phase bytes in the staging tree but recorded that transient
pathname in `real-binding.json`. The R40 finalizer correctly failed when it
reread the now-absent path after publication cleanup.

## Controlled repair

For each phase reference the hook now verifies two related paths:

```text
validated now: artifact_root / reference.relative_path
re-read later: artifact_root.parent / reference.relative_path
```

The hook requires the artifact root to be the exact rank-specific temporary
root under the current result's `primary/raw`, requires a normalized relative
path whose first component is the same `rank-X`, and requires the derived
published target to remain contained and fresh. Only the second, stable path is
written to the receipt. Artifact payload, hashes, GDN digest, and finalizer
verification remain unchanged.
