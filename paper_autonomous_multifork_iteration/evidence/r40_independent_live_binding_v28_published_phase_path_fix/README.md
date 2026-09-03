# R40 v28 published phase-path successor

Status: **HOLD_PENDING_FRESH_AUDIT_AND_H20**. This directory is the minimal
finalizer-path successor to v27. Local validation and packaging are not H20
evidence and cannot authorize claims. No v28 GPU result marker has been
produced.

V27 completed all eight scientific shards, the primary aggregate, stage 06,
and formal binding. R40 finalization then rejected `phase artifact missing on
finalizer reread`. Each real-binding receipt named the rank staging directory
`primary/raw/.forkaudit-rank-X-*/rank-X/...`; the immutable runner had already
atomically published that subtree as `primary/raw/rank-X/...` and removed the
temporary directory. The artifact bytes and hashes were correct, but the
receipt retained a pre-publication pathname. V27 did not close terminally and
is not paper evidence.

## Minimal v28 correction

The immutable Round-04 runner and launcher, resident builder, Qwen3.5 cache
adapter, R39 proxy, R40 verifier predicates, producer accounting, scientific
protocol, selected cell, allocator measurements, no-bytecode wrapper, and
terminal predicates remain unchanged. V28 changes only the real-binding hook's
phase-artifact pathname receipt.

After validating the actual artifact inside the immutable runner's exact
`.forkaudit-rank-X-*` staging root, the hook derives the immutable runner's
fixed publication target from `artifact_root.parent / reference.relative_path`.
It requires the temporary root to be under the same result's `primary/raw`, the
reference's first normalized component to be `rank-X`, strict containment, and
a fresh publication target. The receipt then records
`primary/raw/rank-X/...`, which remains valid after the unchanged atomic rank
publish and staging cleanup.

The inherited `_prepare_request_gdn_base` path directly compact-clones 60 tensors
for materialized-final requests; delegated borrowed construction calls equal all wrapped requests.
The exact `_convert_persistent` scope still binds
the returned persistent object, and an unmarked cache outside that scope fails
closed. The v27 source-ledger-bound interpreter wrapper still prepends `-B`.

A regression constructs the exact temporary rank layout, captures all three
phase receipts, performs the immutable temp-to-published rename, removes the
temporary directory, and proves the unchanged finalizer rereads and verifies
all three published artifacts.

The non-overwriting release archive basename is
`r40-independent-live-binding-v28-published-phase-path-fix-20260902a.tar.gz`.

## Remaining gates

- independently rebuild and audit the frozen v28 overlay and clean stage;
- explicitly approve the exact v28 source, archive, and canonical-v6 hashes;
- run one new non-overwriting eight-rank H20 execution; and
- accept nothing until scientific, cleanup, R40 finalization, and terminal
  closure gates pass.

All previous diagnostics and nonterminal executions remain preserved.
