# R40 v26 construction-step receipt successor

Status: **HOLD_PENDING_FRESH_AUDIT_AND_H20**. This directory is the minimal
producer-receipt successor to v25. Local validation and packaging are not H20
evidence and cannot authorize paper claims. No v26 GPU result marker has been
produced.

V24 repaired the exact persistent-build scope. V25 added correct abort-aware
cached-call accounting and completed all eight remote shards, but every rank
then failed the producer receipt. The failed predicate confused a request's
first borrowed construction step with its final GDN policy: a request that
finishes materialized still first delegates the immutable borrowed-base helper.
The v25 execution did not close terminally and is not scientific evidence.

## Minimal v26 correction

The immutable Round-04 runner, resident builder, native Qwen3.5 cache adapter,
R39 entrypoint, R40 `ActualBindingVerifier`, passive clone-lineage verifier,
scientific protocol, selected cell, and v25 abort-aware hook behavior remain
unchanged. V26 changes only the producer runtime receipt, its fail-closed rank
gate, and preregistered accounting labels:

- every returned request takes exactly one borrowed construction step through
  the exact original `_prepare_request_gdn_base` helper;
- a request whose final policy is materialized then takes one second step that
  directly compact-clones 60 tensors into the final request destination;
- therefore delegated borrowed construction calls equal all wrapped requests,
  canonicalized materialization calls equal materialized-final requests, and
  borrowed-final plus materialized-final requests equal all wrapped requests;
- the full mixed-policy regression freezes rank 0 at `498` wrapped / `498`
  delegated / `238` borrowed-final / `260` materialized-final, and ranks 1--7
  at `494` / `494` / `234` / `260`; and
- a failed producer gate now reports the sorted failed predicates together
  with the receipt's canonical counters, so a future failure is diagnosable
  without changing the scientific path or artifact tree.

The rank-lifetime post-hook remains registered with `always_call=True`.
Expected backbone exceptions record an aborted cached call and perform no
recurrent post-rebind; successful calls still post-rebind exactly 30 recurrent
states. Total observed cached calls must equal successful postprocessed plus
aborted calls, and the original fault exception remains authoritative.

The inherited exact persistent scope still binds `_convert_persistent` to one
backbone, document, pre-hook, post-hook, cache, and returned persistent object.
An unmarked cache outside that scope fails closed. No verifier predicate is
relaxed.

## Current local validation

The targeted compact-rebind suite passes 10/10 with zero skips. It includes an
exact two-request builder regression: borrow-final yields one delegated step;
materialized-final yields one delegated step followed by one materialization
step. It also covers the complete rank-dependent formal cardinalities and the
abort path's zero-rebind/original-exception behavior. These are local mechanism
results, not formal GPU evidence.

The non-overwriting release archive basename is
`r40-independent-live-binding-v26-construction-step-receipt-fix-20260901a.tar.gz`.

## Remaining gates

- independently rebuild and audit the frozen v26 overlay and clean stage;
- explicitly approve the exact v26 source, archive, and canonical-v6 hashes;
- run one new non-overwriting eight-rank H20 execution; and
- accept nothing until scientific, cleanup, and terminal-closure gates all pass.

V21/V22 diagnostics, the v23 pre-science failure, the nonterminal v24 and v25
runs, and all old allocator values remain preserved.
