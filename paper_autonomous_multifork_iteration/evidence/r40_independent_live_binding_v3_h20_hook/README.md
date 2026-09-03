# R40 v3 Qwen/H20 pre-serialization registration hook

Status: **locally prepared and tested; formal GPU execution not run**.

This package is a non-overwriting integration draft for combining an
independent live-binding challenge with the existing R39 v6 compiled-dispatch
eight-rank run.  It does not modify or replace v1, v6, the immutable scientific
runner, its arguments, model revision, PG19 inputs, or primary output schema.

## What changes relative to v1

V1's oracle consumed a semantic manifest and its correct slot/tensor pairs
were assembled in the candidate producer.  V3 forbids both paths:

- the registrar process rejects `manifest`, `slot_manifest`, `slot_id`, and
  `slot_tensors` fields;
- the producer-side hook may send only a preregistered lifecycle coordinate
  `(owner kind, request index, layer index)` plus the raw `conv_states` and
  `recurrent_states` containers;
- the registrar process independently enumerates state families and indices,
  derives the selected semantic slots, imports the already-bound live tensor
  handles, and freezes their challenges before any off-path candidate capture;
- a different observer implementation later receives only opaque slot/tensor
  pairs and independently reconstructs challenge responses and storage
  relations.

The hook is enabled only for the existing N=8, shared-KV,
materialized-GDN ownership-witness cell.  Primary memory cells are untouched.
Setup covers coherent slot swap, cross-layer substitution, and request/base
misbinding.  The post-transition wrapper first refreshes the registrar from a
raw request-layer state container and waits for acknowledgement; it then tests
a retained stale handle before the immutable producer writes its normal phase
witness.

## Local gates

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=executed_source \
python3 -m unittest discover -s tests -p 'test_*.py' -v

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=executed_source \
python3 scripts/static_audit.py

bash -n formal/launch_combined_h20_draft.sh
```

## Formal launch status

`formal/launch_combined_h20_draft.sh` is deliberately gated by an explicit
`R40_H20_EXECUTION_AUTHORIZED=yes` environment variable and uses a new result root.  It
contains no QS command and has not been executed.  A fresh authorized 8-H20 run
must pass the complete existing v6 preflight/result gates plus the v3 rank
captures before any formal claim is eligible.

## Boundary

This mechanism removes the shared producer manifest and candidate-binder-built
oracle found in v1.  It still trusts the instrumentation hook's lifecycle
event `(owner, request, layer)`, hook installation integrity, PyTorch tensor
and CUDA-IPC semantics, and the pause until registrar/observer acknowledgement.
It is designed against accidental or independently injected binder faults,
not a malicious producer that forges construction events.  No GPU result is
claimed by this directory.
