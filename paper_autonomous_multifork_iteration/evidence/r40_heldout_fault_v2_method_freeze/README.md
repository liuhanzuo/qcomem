# ForkAudit method-v2 local freeze

Status: **local method freeze only; no v2 fault set and no H20 scientific
result**.  The authoritative local status and hashes are in
`METHOD_FROZEN.json`, `method-freeze.json`, and `TERMINAL_SHA256SUMS`; the
scientific campaign remains on HOLD in `formal-blocker.json`.

This package implements three general, always-on method-v2 gates:

1. paired call-cardinality, token, and complete-vocabulary FP32-logit semantic
   comparison under an exact or predeclared-tolerance policy;
2. paired synchronized allocator current/peak endpoints plus exact cleanup
   restoration; and
3. per-call hybrid KV/GDN atomic-version coherence from independent live
   pre/post state reads bound to the surfaced call receipt.

R39 artifacts are used only by a development regression test.  Every R39 fault
and outcome remains design knowledge and is prohibited from future method-v2
held-out scoring.  The detector implementation contains no R39/BF03-specific
branch.

`designer_snapshot/` is the only input intended for the future fresh isolated
fault designer.  It contains the frozen public contract and injection interface
only; it contains no R39/v1 fault list, result, reviewer comment, detector
source, or private predicate implementation.  This package does not design any
method-v2 fault.

`executor_skeleton/` is node-local scaffolding for a later eight-H20 one-shot
campaign.  It cannot execute until a separate fault freeze fills all null
bindings and an independently audited formal configuration is supplied.  It
requires explicit authorization, a new output root, an immutable one-shot
lock, eight distinct H20 UUIDs, a 15-minute timeout per fault, and exactly one
reference--clean--mutant attempt with no tuning or retry.

No file outside this directory is modified by the local build or validation
commands.

`ACCEPTANCE_GATES.md` defines the fail-closed local checks.
`H20_ONE_SHOT_PLAN.md` gives the conditional formal ordering and claim boundary.
`COMMANDS.md` records the two CPU-only reproduction commands.
