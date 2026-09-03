# R33 trusted-capture design decision

Date: 2026-08-25  
Status: bounded H20 scientific execution and independent acceptance complete

## Target objection

The reviewer-facing objection is not that R29 disagreed with ForkAudit.  It is
that both R29 producers inspected candidate-created objects in one process and
trusted the same live Python state.  A second replay of serialized JSON cannot
resolve that objection.

## Alternatives considered

1. **OS/driver allocation monitor.** Rejected as the immediate experiment
   because CUDA does not expose enough public cross-process metadata to bind
   arbitrary allocations back to all 180 semantic GDN owner coordinates.
   Implementing it would require new C++/CUDA allocator instrumentation and
   would still need a trusted semantic-binding channel.
2. **Independent process fed serialized descriptors.** Rejected because the
   critical descriptor facts would still be produced by the candidate.
3. **Independent process fed copied tensor bytes.** Rejected because copying
   each view destroys storage alias/overlap identity.
4. **Selected: PyTorch shared-memory/CUDA-IPC tensor reduction.** The receiver
   imports live storage-backed views, pins them against receiver-side ABA, and
   reconstructs all descriptors and relations itself.  Only opaque slot ids
   and tensors cross the live wire; judgment labels do not.

The selected design is the strongest minimally invasive experiment compatible
with the frozen Python/PyTorch runtime.  It changes the process and address
space that perform capture while preserving live storage aliasing.

## Producer/observer split

The frozen slot manifest maps opaque slot ids to the complete owner-coordinate
grid.  The producer traverses `layers`, `conv_states`, and
`recurrent_states` according to that manifest and sends exactly one tensor per
slot.  The worker source imports no QComem, ForkAudit, Transformers, vLLM, or
model-runtime module.  It has its own HMAC secret and never serializes raw
addresses.  One worker is used per three-phase policy cell so identities remain
stable and imported mappings can be pinned without retaining the first cell
through the second.

The offline replay—not the live producer or worker—binds capture ids to setup,
transition, and generation and evaluates shared/materialized ownership plus
persistent/incomplete/completed lifecycle rules.  A scientific replay must
also bind to the exact preregistration digest.

## Formal outcome and authorization

The prospectively frozen execution completed on one Qwen3.5-35B-A3B/H20-3e
stack.  The two policy cells used observer PIDs 2178 and 2384, both distinct
from producer PID 1816.  Across six captures, the observers reconstructed 1,080
rows and 96,660 pair relations.  The frozen replay passed all six phase and both
lifecycle verdicts.  A separate read-only acceptance audit reproduced the
archived replay exactly and independently checked the deterministic manifest,
slot-coordinate bindings, live-wire receipts, PID separation, row/relation
counts, model steps, and kernel-ledger receipts.

Authorization occurs in `formal_h20/independent_acceptance.json` and the
experiment registry after successful replay and acceptance.  The immutable raw
result correctly remains a pre-replay artifact with
`status=completed_pending_independent_replay` and `claim_authorized=false`; it
was not rewritten after the outcome was known.

## Falsification

The run fails closed if CUDA IPC cannot preserve a stable receiver mapping, if
any slot is absent/duplicated, if any live judgment field appears, if observer
and producer PIDs coincide, if descriptor/relation hashes drift, or if any
phase/lifecycle expectation fails.  An operational IPC failure is not a
scientific negative about ForkAudit; an observed valid descriptor, relation,
or lifecycle mismatch is preserved as scientific evidence.

## Remaining boundary after the pass

A pass removes the same-process and candidate-verdict dependencies.  It does
not remove trust in producer-side semantic slot enumeration, PyTorch/CUDA IPC,
or paused phase orchestration.  It is therefore appropriately described as
"out-of-process PyTorch/CUDA-IPC recapture," not "external ground truth" or
"trusted-computing-base-free capture."  It also does not establish malicious-
producer resistance, OS/driver allocation enumeration, independent model
execution or end-to-end correctness, KV ownership recapture, kernel/dispatcher/
compiled-binary/autotuning attestation, continuous batching, production
performance or capacity, the full 96-cell primary protocol, cross-model/
runtime/hardware generality, or detection of transient writes restored between
the paused captures.
