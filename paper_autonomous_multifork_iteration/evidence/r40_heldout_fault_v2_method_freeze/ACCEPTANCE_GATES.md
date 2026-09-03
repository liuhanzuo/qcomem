# Method-freeze acceptance gates

All local gates below must pass before the method bytes are frozen.  A local
pass is an engineering result only; it is not a fault-detection result.

| ID | Frozen requirement | Failure action |
|---|---|---|
| L01 | Every Python file parses and every JSON file loads; the launcher passes `bash -n`. | HOLD; do not freeze. |
| L02 | The complete local unit and integration suite passes, including negative tests for missing calls, corrupt sidecars, unsynchronized allocator rows, torn hybrid commits, forged receipt state, overwrite attempts, and unauthorized formal launch. | HOLD; fix before freeze, then rerun the full suite. |
| L03 | Predicate source contains no historical case ID or case-ID dispatch and applies the same atomic relation to every receipt. | HOLD; remove case-specific logic. |
| L04 | Every public designer-snapshot member matches `designer_snapshot/SHA256SUMS`, and a forbidden-content scan finds no historical campaign identifier, outcome, review comment, or private detector implementation. | HOLD; rebuild the snapshot before any designer sees it. |
| L05 | Historical artifacts are read-only development inputs, their hashes are recorded, all rows are retained, and the regression is labeled non-held-out/non-scoring. | HOLD; no historical result may enter a method-v2 score. |
| L06 | The formal template is unsealed and null-bound, explicit authorization is mandatory, and the driver enforces eight H20 UUIDs, 900 seconds per case, ordered reference--clean--mutant lanes, a new output root, zero retry, zero tuning, and terminal retention. | HOLD; formal execution remains impossible. |
| L07 | The source ledger covers all contract, capture, predicate, integration, test, snapshot, executor, and local-validation bytes; its deterministic archive verifies. | HOLD; do not report a freeze hash. |

After L01--L07 pass, the local status is `PASS_METHOD_FREEZE_ONLY`.  The
scientific campaign status remains `HOLD` until a separate auditor approves
the frozen bytes, a genuinely fresh isolated designer seals eight cases from
only the public snapshot, an injection/runner bundle is independently audited
and frozen before outcomes, and an explicitly authorized one-shot execution
retains all terminals.

