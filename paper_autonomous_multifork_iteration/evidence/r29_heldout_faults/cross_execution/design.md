# Round-29 held-out fault cross-execution design

Status: frozen before candidate execution.

## Separation and independence boundary

The held-out fault author owns the frozen suite, public fault module, author
tests, and launcher.  The cross-executor did not edit those files.  It owns
only the generic RR2 context adapter, strict aggregator, focused tests, and
this execution-input binding.

Candidate/live code produces the fresh resident cache objects, model calls,
kernel ledgers, and existing KV/GDN witnesses.  The executor independently
produces lane nonces, allocator/disposal receipts, exact action-sequence
replay, token/logit sidecars, exception provenance, and detector tri-state
records.  The aggregator independently reopens and hashes every FP32 sidecar,
recomputes token/full-logit comparisons, validates intervention/restoration,
replays the action-sequence predicate, authenticates rejection authorities,
checks fresh-case and allocator cleanup, and rejects pooled-rate fields.

The executor imports the already SHA-bound RR2 audit modules to execute the
uniform receipt battery.  It imports no outcome table.  Detector/scoring
functions contain no `H01`, `H02`, or `H03` branch and no expected-gate map.
The only fault-class dispatch is the frozen public intervention adapter:
state faults call `apply_state_fault`; the action fault consumes
`h02_action_sequence` exactly.

## Uniform execution

Each assigned rank loads one model and performs one discarded N=2 warmup.
It then constructs and disposes three fresh sequential cases in the frozen
order `clean`, `fault_conventional`, and `fault_forkaudit`.  Every model call
uses `frozen_query_bank[rank][0][31]`.  The conventional lane withholds
ForkAudit receipt verdicts.  The ForkAudit lane runs the same six ordered
receipt predicates for every fault and stops at the first authenticated
rejection.

Generic crashes are operationally invalid.  A conventional production
assertion counts only when its exact exception type, message, source function,
and traceback match one of two pre-frozen production-kernel assertions already
used by the fixed stack.  Missing token/logit output is `not_evaluated`.

## Strict aggregation and claim boundary

The aggregate contains three per-fault rows and no pooled detection rate.  A
valid negative or escaped fault remains visible.  `scientific_valid=true`
requires three distinct H20 UUIDs, all clean cases complete, all interventions
non-no-op, all state mutations restored, all fresh cases disposed, exact
post-warmup allocator restoration, finite referenced FP32 sidecars, valid
detector tri-state semantics, and zero operationally invalid cases.

The experiment evaluates historical-pattern-inspired consequence-level
mutations on one fixed Qwen3.5/H20 stack.  It does not establish a naturally
occurring bug, reproduce an upstream implementation, estimate population
sensitivity, or support a cross-runtime claim.

