# R33 Attempt-B harness amendment

Attempt A terminated identically on ranks 0--4 before construction of any
matched clean case, at the pre-build allocator equality assertion.  No fault
injection, mutant execution, detector predicate, semantic oracle, or
scientific classification was reached.  Its non-overwriting output directory
is retained as an operational-invalid record.

The root cause was an executor-only Python lifetime defect.  The discarded
warm-up retained its final loop-local `request` and `ledger` aliases while it
captured the allocator baseline.  Those aliases were released only after the
warm-up function returned, so the next model-only snapshot could not equal the
baseline that still included the warm-up arena.

Attempt B changes only the warm-up disposal statement: it clears `request`,
`ledger`, and `logits` before the existing garbage collection, CUDA cache
cleanup, synchronization, and allocator snapshot.  This restores the disposal
order already used by the inherited R29 executor.  All author-frozen fault
definitions, row hashes, rank and policy assignments, injection payloads,
predicate precedence, expected primary gates, thresholds, clean requirements,
and fail-closed classification rules remain unchanged.

Attempt B uses run ID `R33-FRESH-FAULTS-20260825B`, a newly frozen protocol and
package, and a new output root.  All five ranks are rerun; there is no selective
replacement of Attempt-A ranks.

Regression validation before freezing:

```text
PYTHONPATH=scripts python3 -m unittest \
  scripts/r33_test_fault_mapping.py scripts/r33_test_executor_core.py
Ran 15 tests in 0.655s
OK
```

The added test parses the formal executor and requires the warm-up loop aliases
`request`, `ledger`, and `logits` to be cleared before `_cleanup_allocator`
freezes the baseline.

