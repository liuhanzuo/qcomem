# Design and hard acceptance gates

## Selected existing cell

The hook is active only inside one already-existing ownership-witness cell per
rank:

- `resident_count = 8`;
- KV policy `vllm-q16-shared-document-reuse`;
- GDN policy `materialize-request-base-functional-rebind`.

The corresponding formal-memory cell and every other factorial cell remain on
the exact v6 path.  The immutable runner, runner argv, model/data paths,
generation schedule, and compiled-dispatch recorder remain unchanged.

## Registration path

The combined rank entrypoint chains the v3 hook after v6 installs its existing
scope wrappers.  When the selected witness calls `build_resident_request_group`,
the v3 wrapper starts a spawn-created registrar, calls the unchanged builder,
and then sends only raw state-container events.  It never constructs a
slot/tensor list for the registrar.  Each event contains:

`owner_kind, request_index, layer_index, conv_states, recurrent_states`.

The registrar rejects any slot manifest, slot ID, candidate row, phase verdict,
or expected detector outcome.  It independently selects the six preregistered
semantic coordinates, validates container/state geometry, derives opaque slot
IDs, and freezes challenge/descriptor/storage-relation observations.

The unchanged builder performs the real live-object construction/binding first.
At post-transition, before the immutable `_write_witness_phase` producer
serialization, the hook refreshes the selected request/layer raw container and waits
for registrar acknowledgement.  This produces a new registrar epoch while the
old live handle remains pinned for the stale-handle negative control.

## Candidate path

A separate candidate binder derives the same selected coordinates without
reading registrar rows.  Clean and mutant maps are sent to a source-distinct
observer process.  Faults modify live tensor references only:

- coherent same-geometry swap;
- one-way cross-layer substitution;
- materialized request role bound to persistent base;
- stale pre-transition handle after a registered post-transition rebind.

No graph mutation is required; therefore the immutable scientific witness
continues with the original clean graph after each off-path challenge.

## Hard gates inherited from the v1 audit

Formal eligibility requires all of the following:

1. registrar live messages contain no manifest, slot ID, slot/tensor list, or
   producer observation row;
2. the registrar independently derives its expected selected-slot set;
3. registration acknowledgement follows real object construction/binding and
   precedes every off-path candidate capture and immutable phase serialization;
4. registrar and candidate observer run in distinct spawn-created processes;
5. candidate binder source is not imported by the registrar process;
6. clean captures pass and all four exact live-handle faults fail closed with
   preregistered codes;
7. slot labels and wire schemas remain identical across clean/mutant captures;
8. primary memory cells receive zero registration or observation event;
9. v6 compiled-dispatch receipts and all immutable primary replay gates pass;
10. no source/input/model/argv/result-root substitution or post-outcome retry
    is allowed.

## Residual TCB and disturbance

The event hook still supplies owner/request/layer lifecycle metadata and is
therefore not malicious-producer-resistant.  PyTorch/CUDA IPC and hook install
integrity remain trusted.  The selected witness cell pauses for IPC import and
CPU challenge hashing, so execution timing is perturbed even though tensor
values, model calls, inputs, and primary memory endpoints are not.  Formal
results must say this explicitly.
