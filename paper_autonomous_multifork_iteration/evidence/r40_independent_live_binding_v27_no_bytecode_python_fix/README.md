# R40 v27 no-bytecode Python successor

Status: **HOLD_PENDING_FRESH_AUDIT_AND_H20**. This directory is the minimal
terminal-cleanliness successor to v26. Local validation is not H20 evidence;
packaging are not H20 evidence and cannot authorize claims. No v27 GPU result marker has
been produced.

V26 completed all eight scientific shards and the R40 aggregate successfully.
Terminal closure then failed after stage 06 because the immutable primary code
snapshot contained a writable `__pycache__`. The model-load lease command uses
the R39 Python proxy's transparent branch with upstream `-I -c`; isolated mode
ignores `PYTHON*` controls, so the actual interpreter could write bytecode after
the preflight cleanliness check. V26 did not close terminally and is not paper
evidence.

## Minimal v27 correction

The immutable Round-04 runner and launcher, resident builder, Qwen3.5 cache
adapter, R39 proxy, R40 verifier, producer accounting, scientific protocol,
selected cell, and terminal predicates remain unchanged. V27 adds one
source-ledger-bound executable, `executed_source/r40_no_bytecode_python`.

The inherited `_prepare_request_gdn_base` path directly compact-clones 60 tensors
for materialized-final requests; delegated borrowed construction calls equal all wrapped requests.
The exact `_convert_persistent` scope still
binds the returned persistent object, and an unmarked cache outside that scope
fails closed.

The generated launcher captures the actual environment interpreter in
`R40_ACTUAL_REAL_PYTHON`, then points `R39_PRIMARY_REAL_PYTHON` at this wrapper.
The wrapper executes the actual interpreter with a leading command-line `-B`
before all upstream arguments. Thus the transparent lease command becomes
`real-python -B -I -c ...`; shard routing receives the same leading `-B`, with
the inherited later `-B` harmlessly redundant. No immutable source is edited.

A regression routes an isolated transparent command through the exact R39
proxy and v27 wrapper, proves the actual interpreter receives `-B` before
`-I`, imports a module from a source directory, and proves no `__pycache__` is
created there.

The non-overwriting release archive basename is
`r40-independent-live-binding-v27-no-bytecode-python-fix-20260901a.tar.gz`.

## Remaining gates

- independently rebuild and audit the frozen v27 overlay and clean stage;
- explicitly approve the exact v27 source, archive, and canonical-v6 hashes;
- run one new non-overwriting eight-rank H20 execution; and
- accept nothing until scientific, cleanup, and terminal-closure gates pass.

All previous diagnostics and nonterminal executions remain preserved.
