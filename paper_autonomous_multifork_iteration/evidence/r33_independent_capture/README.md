# R33 out-of-process GDN capture

Status: the separate-process protocol, local CPU engineering gate, and
prospectively frozen Qwen3.5/H20 execution are complete.  The frozen CPU replay
and a separate read-only acceptance audit passed.  The acceptance and evidence
registry authorize only the bounded claim stated below; the immutable raw file
retains its pre-replay `claim_authorized=false` and
`status=completed_pending_independent_replay` fields by design.

## What changed from R29

R29 used a source-distinct observer in the candidate Python process.  R33
moves descriptor and relation reconstruction into a `spawn`-created process.
The producer sends each live tensor through PyTorch multiprocessing reduction
under an opaque slot id.  The worker receives no phase, policy, completion,
expected-relation, candidate-row, `passed`, or verdict field.  It independently
derives, in its own process:

- shape, stride, offset, dtype, device, storage extent, and byte interval;
- a content digest;
- HMAC-obscured receiver-local storage and view identities; and
- every exact-alias, partial-overlap, and disjoint pair relation.

The worker emits raw observations only.  The standalone replay binds opaque
capture ids to the prospectively frozen phase/policy plan and computes the
ownership and lifecycle verdicts.

## Local engineering gate

The local gate uses real multiprocessing shared-memory reduction on CPU and a
three-layer mock cache.  It is not scientific model evidence.  It completed
two N=2 policy cells and three captures per cell:

- two observer PIDs, both distinct from the producer PID;
- 108 receiver-derived row observations;
- 918 receiver-derived pair relations;
- zero live judgment/candidate-verdict fields received;
- both three-phase lifecycle replays passed; and
- six unit/fault/tamper/protocol-binding tests passed.

Artifacts:

- `local_gate/raw-result.json`, file SHA-256
  `0742d87d730937dd8b4193aa0938639d6ab1010e13ebe2d2e57a957005c440da`
- `local_gate/replay.json`, file SHA-256
  `1cee3cba42cf0814fe789d25c4279b1385502e833c137de7e5b75276ba7d2916`

Reproduce from the paper root:

```bash
python3 scripts/r33_test_independent_capture.py -v
python3 scripts/r33_run_local_capture_gate.py \
  --output /tmp/r33-local-result.json \
  --replay-output /tmp/r33-local-replay.json
```

## Completed frozen H20 execution

The formal protocol is `preregistration.json` (SHA-256
`67c001a04e61006967befe1ff1e26018b58bc02630b8e48dd40f16067a44ea65`).
It reuses the validated R29 Qwen3.5/H20 runtime setup but does not call either
the candidate storage-witness capture or the R29 same-process observer.  It
produced 1,080 receiver rows and 96,660 receiver-derived relations across two
policies and six phase points.  The frozen launcher is
`scripts/r33_launch_h20_independent_capture_1gpu.sh`.
The disclosed pre-execution amendment adds only explicit GPU scheduling:
`R33_GPU_INDEX` defaults to 7 and may be set to another index in `[0,7]`.

The formal run used Qwen/Qwen3.5-35B-A3B revision
`59d61f3ce65a6d9863b86d2e96597125219dc754` on one NVIDIA H20-3e.  Producer
PID 1816 and observer PIDs 2178 and 2384 were distinct.  All six captures
reported exactly the frozen live request and slot fields, zero judgment or
candidate-verdict fields, 180 rows, and 16,110 pair relations.  The independent
replay passed 2/2 policy cells, 6/6 phase verdicts, and 2/2 lifecycle verdicts.
Each cell preserved 60 persistent rows and observed the registered 60-row
request-0 and request-1 rebindings.  A fresh read-only CPU recomputation was
object-identical to the archived replay.

Formal artifacts and file SHA-256 values:

- preregistration: `67c001a04e61006967befe1ff1e26018b58bc02630b8e48dd40f16067a44ea65`;
- raw result: `50d39cfcea072fb770da539d90abeddcd8a40802b88f4f95315001333c09e974`;
- independent replay: `dfda58f7596643b6a7366f217123aba5f51a29b0e1e93408419f73176eda8180`;
- terminal ledger: `0261aa174435b6e1affdbd2fc116e5b30b38c8ca4d1bf6b57ba2f6a6a10961ae`;
- read-only acceptance: `formal_h20/independent_acceptance.json`,
  `d730793e9cf57fabeccdc5d0dba16ef7ba2da8b64c3e74c109bdb4da5134d1b0`.

The active evidence identifier is
`E-R33-OUT-OF-PROCESS-GDN-CAPTURE-A`.  R29's same-process source-distinct
observer remains an internally preserved corroboration cohort and is not the
active manuscript support for the capture-independence statement.

## Exact trust boundary

This is materially stronger than same-process agreement, but it is not an
OS/driver-level adversarial monitor.  The producer still enumerates the frozen
semantic owner slots and exports the corresponding live tensor handles; both
sides trust PyTorch tensor/storage semantics and PyTorch CUDA IPC reduction.
The producer is paused until observer acknowledgement.  The completed run
therefore supports out-of-process PyTorch/CUDA-IPC GDN recapture on this fixed
stack.  It does not support resistance to a malicious producer,
producer-independent semantic-slot enumeration, OS/driver allocation
monitoring, external ground truth, independent model execution or end-to-end
correctness, KV recapture, kernel/dispatcher/binary/autotuning attestation,
continuous batching or production performance/capacity, coverage of the full
96-cell primary protocol, cross-model/runtime/hardware generality, or exclusion
of transient writes restored between the paused captures.
