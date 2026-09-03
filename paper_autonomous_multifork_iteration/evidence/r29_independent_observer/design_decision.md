# Round 29 independent GDN observer design

Date: 2026-08-25  
Status: frozen before H20 candidate outputs

## Question

The existing ForkAudit GDN ownership evidence is replayable after capture, but
the same candidate module both observes live tensor/storage state and emits the
serialized rows.  This experiment asks a narrower question: on the frozen
Qwen3.5/H20 stack, does a separately implemented live observer derive the same
GDN ownership facts and lifecycle transitions as the candidate producer?

## Selected minimal experiment

One H20 runs two fresh N=2 cells on the same frozen rank-0 PG19-train window:

1. shared immutable GDN base at setup; and
2. materialized per-request GDN bases at setup.

The KV axis remains fixed to shared Q16 reuse.  In each cell, the independent
observer and candidate producer inspect the same live cache at three points:
setup, after request 0 has executed one 32-token forward, and after request 1
has executed one 32-token forward.  The independent observer is called before
and after the candidate capture; those two observations must be byte-identical.

Each phase contains 180 GDN tensor rows: 60 persistent tensors and 60 tensors
for each of two requests, spanning all 30 linear layers and both convolutional
and recurrent state families.  The comparison recomputes all 16,110 unordered
pair relations from opaque storage identity plus byte intervals.  Across two
policies and three phases, the frozen run therefore has 1,080 row observations
and 96,660 pair-relation comparisons.

## Independence boundary

Candidate-controlled components are model loading, Q16 cache construction,
request execution, and the existing candidate storage-witness JSON.  The new
observer does not import any QComem or ForkAudit module.  It independently:

- traverses the live owners through `layers`, `conv_states`, and
  `recurrent_states`;
- obtains shape, stride, dtype, storage extent, view interval, and content
  bytes through PyTorch tensor/storage APIs;
- maps raw object and storage identities to stable HMAC tokens under an
  observer-only random key that is never serialized;
- pins setup tensor objects to prevent allocator-address ABA reuse;
- recomputes exact-alias, partial-overlap, and disjoint relations;
- checks persistent immutability, incomplete-request stability, completed
  out-of-place rebinds, and policy-specific ownership; and
- compares those independently derived facts with the candidate rows without
  trusting candidate `passed` fields.

The runner imports both implementations only to place their results beside one
another.  The CPU replay imports only the independent observer and re-derives
the comparison from raw rows.

Shared trusted dependencies remain: both observers execute in one Python
process, inspect the same candidate-created objects, and rely on the same
PyTorch runtime and storage API.  The experiment is an independently
implemented live GDN observer, not an OS-level external monitor and not an
independent implementation of model execution.

## Alternatives rejected

1. Replaying only existing JSON was rejected because it adds no independent
   live capture.
2. Reusing the candidate capture helper under a new wrapper was rejected
   because the core facts would remain circular.
3. An out-of-process CUDA memory monitor was rejected for this round because it
   cannot bind allocator blocks to the 180 semantic owner coordinates without
   new invasive runtime instrumentation.
4. Rebuilding the entire ForkAudit producer was rejected as unnecessary for
   the smallest falsifiable test; KV pages, attention dispatch, logits, and
   compiled-kernel attestation remain outside this observer's scope.

## Claim boundary

A passing run supports exact agreement between two live GDN ownership
producers for two N=2 policy cells and three lifecycle phases on one frozen
Qwen3.5/H20 stack.  It also independently supports out-of-place GDN rebind
timing and persistent-base immutability for those cells.

It does not establish independent end-to-end recapture, KV ownership
recapture, kernel/dispatch identity, compiled-binary or autotuning attestation,
production-serving concurrency, generality to another model or runtime, or
resistance to a malicious candidate that deliberately deceives same-process
introspection.
