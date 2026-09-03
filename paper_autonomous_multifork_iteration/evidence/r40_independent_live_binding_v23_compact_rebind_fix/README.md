# R40 v23 compact-rebind producer-fix successor

Status: **HOLD_PENDING_FRESH_AUDIT_AND_H20**. This directory is the producer-fix
successor to v22. Local validation and packaging are not H20 evidence and
cannot authorize paper claims. No v23 GPU result marker has been produced.

V21 passed formal preflight and entered eight-rank GPU science, but every rank
failed the first selected generation callback at the unchanged exact descriptor
gate. V22 preserved the predicate and showed coordinate `(0, conv, 0)` with
only a stride difference: expected `[33546240,1,8192]`, current
`[32768,4,1]`. Shape, offset, and interval were identical. V23 therefore
canonicalizes the materialized request endpoint before the verifier captures
its authorized descriptor; it does not loosen the descriptor predicate.

## Producer repair

The immutable Round-04 runner, resident builder, native Qwen3.5 cache adapter,
R39 entrypoint, and Transformers runtime remain unchanged. The v23 R40 overlay
adds `executed_source/r40_compact_rebind_fix.py` and installs it from the R40
rank wrapper. For the materialize policy, it temporarily replaces only the
builder global `_prepare_request_gdn_base`. Each request's 60 GDN tensors is the
direct destination of one contiguous clone under the unchanged passive
clone-lineage mode. Borrow delegates the exact original helper. Helper identity
is restored at rank exit and any drift fails closed.

The R39 runtime receipt requires every linear cache layer's
`update_conv_state.__func__` and `update_recurrent_state.__func__` to remain its
exact intercepted wrapper. V23 therefore does not replace either updater.
Instead, it installs one rank-lifetime backbone pre/post-hook pair after the
formal runtime is loaded:

1. before a cached single-token backbone call, the target request's 30
   convolution states are rebound to differentiable fresh contiguous compact
   clones;
2. the unchanged Transformers causal-convolution fallback mutates those new
   private buffers in place, preserving the frozen single-token route and call
   count; and
3. after every cached backbone call, the target request's 30 recurrent states
   are rebound to fresh contiguous compact clones before the immutable runner's
   callback and allocator endpoint.

The rank-wide resident-group wrapper only marks and validates every returned
request. It does not change updater identities. The hooks therefore apply to
warmup, every formal memory and witness cell, and cached full-model calls in the
fault controls, rather than only the selected R40 witness.

The compact helper performs metadata-only runtime validation: new object and
storage, unchanged shape/dtype/device, contiguous layout, zero offset, and
storage bytes exactly equal tensor bytes. It deliberately performs no GPU
hash/equality operation in the measured path. Unit tests and the inherited
formal semantic exactness gates cover value preservation.

## Current local validation

The new targeted suite passes 6/6 with zero skips:

- nonzero-offset, oversized-backing, and noncontiguous recurrent inputs become
  fresh compact endpoints;
- the single-token route leaves the prior convolution endpoint unchanged and
  mutates a fresh endpoint;
- one 32-token call followed by seven single-token calls produces globally
  fresh compact convolution/recurrent endpoints while retaining exact route
  counts and updater function identity; and
- rank-wide installation covers multiple groups and removes both hooks on
  restore;
- the exact v22 noncanonical setup stride becomes `[32768,4,1]` through 60
  direct persistent-rooted clone edges with no derived or extra edge; and
- borrow construction delegates the original helper and restores its identity.

Inherited real-binding/hook checks used during development also pass: 24
hook/real-binding tests and 7 generation-lifecycle tests. These are local
mechanism results, not formal GPU evidence.

## Remaining gates

- perform an independent rebuild/audit and exact clean-stage replay; and
- obtain explicit approval of the frozen source, overlay, and canonical-v6
  hashes; and
- only then authorize a new non-overwriting H20 run and rerun allocator and
  semantic measurements.

Old v21/v22 failures and all immutable primary bytes remain preserved.
