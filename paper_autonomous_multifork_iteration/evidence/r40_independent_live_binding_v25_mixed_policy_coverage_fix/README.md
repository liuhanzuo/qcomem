# R40 v25 mixed-policy and abort-aware receipt successor

Status: **HOLD_PENDING_FRESH_AUDIT_AND_H20**. This directory is the minimal
producer-receipt successor to v24. Local validation and packaging are not H20
evidence and cannot authorize paper claims. No v25 GPU result marker has been
produced.

V23 repaired the selected setup descriptor but failed before science because
the global cache hook also saw the persistent document prefill. V24 added the
exact persistent-build scope and completed the frozen runner, including its
scientific shard, but then failed the producer receipt gate. The failure had
two accounting causes: a borrowed-only count was compared with the combined
borrowed/materialized request count, and expected fault-control exceptions had
pre-hook observations without a post-return observation. The v24 execution did
not close terminally and is not scientific evidence.

## Minimal v25 correction

The immutable Round-04 runner, resident builder, native Qwen3.5 cache adapter,
R39 entrypoint, R40 `ActualBindingVerifier`, passive clone-lineage verifier,
scientific protocol, and selected cell remain unchanged. V25 changes only the
producer runtime receipt and its fail-closed rank gate:

- `_prepare_request_gdn_base` still directly materializes 60 compact tensors
  for each materialized request and delegates the exact original helper for
  each borrowed request;
- the resident-group wrapper now records borrowed and materialized returned
  requests separately, requires delegated borrow calls to equal borrowed
  returns, requires canonicalized materialization calls to equal materialized
  returns, and requires their sum to equal all wrapped requests;
- the rank-lifetime post-hook is registered with `always_call=True`; an expected
  backbone exception records one aborted cached call and performs no recurrent
  post-rebind, while a successful call still post-rebinds exactly 30 recurrent
  states; and
- total cached calls must equal successful postprocessed plus aborted calls.
  The original fault exception remains authoritative and is not replaced.

The inherited exact persistent scope still binds `_convert_persistent` to one
backbone, document, pre-hook, post-hook, cache, and returned persistent object.
An unmarked cache outside that scope fails closed. The descriptor predicate is
not loosened.

## Current local validation

The targeted compact-rebind suite passes 9/9 with zero skips. In addition to
the inherited endpoint, lifecycle, lineage, helper-restoration, and exact
persistent-scope checks, it now covers:

- exact full-protocol mixed-policy cardinalities for rank 0
  (`34` groups, `498` requests, `238` borrowed, `260` materialized) and ranks
  1--7 (`32`, `494`, `234`, `260`); these include max-N priming, four-arm
  warmup, the formal `3 x 4 x {memory,witness}` matrix, and N=2 fault groups;
  and
- a wrapped cached call that intentionally raises: one aborted call, zero
  successful postprocesses, zero recurrent rebinds, complete call accounting,
  and the original exception preserved.

These are local mechanism results, not formal GPU evidence.

The non-overwriting release archive basename is
`r40-independent-live-binding-v25-mixed-policy-coverage-fix-20260901b.tar.gz`.
The earlier `20260901a` pre-final candidate is superseded and must not be used
as the operator-approved overlay.

## Remaining gates

- independently rebuild and audit the frozen v25 overlay and clean stage;
- explicitly approve the exact v25 source, archive, and canonical-v6 hashes;
- run one new non-overwriting eight-rank H20 execution; and
- accept nothing until scientific, cleanup, and terminal-closure gates all pass.

V21/V22 diagnostics, the v23 pre-science failure, the nonterminal v24 run, and
all immutable primary bytes remain preserved.
