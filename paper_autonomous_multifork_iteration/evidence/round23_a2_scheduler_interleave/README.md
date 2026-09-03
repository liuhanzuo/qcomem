# Round-22 A2 scheduler/interleaving experiment

This directory contains a new affected-only experiment.  It does not rerun or
replace the stable factorial, vLLM, SGLang, GDN, lifecycle, Transformers, or
Marconi evidence.

## Scope and claim boundary

- Frozen implementation: Qwen3.5-35B-A3B revision
  `59d61f3ce65a6d9863b86d2e96597125219dc754` with the existing ForkAudit
  vLLM-Q16 adapter and pinned upstream code ledger.
- New operational case: a deterministic scheduler-managed request-step
  interleaving over four resident requests, followed by cancellation of suffix
  slot 3, zero-scrub, epoch increment, exact reservation reassignment, and
  completion of the replacement interleaved with the three survivors.
- New geometry cells: page size 64 with a 3,985-token prefix (17-token tail),
  and page size 128 with a 4,033-token prefix (65-token tail).
- New preregistered production-path faults: stale dispatch after reclamation,
  cross-request dispatch under a valid live lease, and reclaim before zero
  scrub.  The gate mapping was frozen before GPU output.
- This is scheduler-managed interleaving, not evidence of concurrent CUDA
  kernels, throughput, continuous batching, cross-model/runtime generality, or
  completeness over scheduler faults.

## Pre-GPU gates and preserved invalid attempt

The local and node CPU/mock suites both passed five unit tests and the three
frozen fault-to-gate checks.  The first static attempt is preserved as
`invalid_attempt_a.json`: N=3 was rejected by the frozen adapter before model
load or scientific GPU execution because the supported resident counts are
N in {1, 2, 4, 8, 16, 32}.  A new design preregistration changed only N to 4
and moved the cancelled suffix slot to 3; geometry, fault set, expected gates,
and reporting rules did not change.  The original traceback is retained as
`invalid-attempt-a.stderr.log`.

## Debug result

The debug-only rank-0 run executed on the persistent 8xH20 node, using one
H20-3e, from `2026-08-20T18:07:14Z` to `2026-08-20T18:09:21Z`.

For both geometry cells:

- the exact 18-event lifecycle schedule replayed;
- cancellation, zero-scrub, slot-epoch invalidation, and exact reservation
  reuse completed;
- document bytes remained immutable;
- generated tokens, full-vocabulary logits, final logical KV, and final GDN
  state were exactly equal to the uninterrupted control;
- each of the three held-out faults reached its preregistered gate.

`debug-attempt-b.json` remains explicitly `debug_only=true` and
`formal_evidence_eligible=false`.  No detection rate is reported from it.

## Formal result

The frozen formal launch completed on Job 246593 / Trial 1871681 from
`2026-08-20T18:42:35Z` to `2026-08-20T18:44:10Z`.  All eight independent rank
shards completed, using eight distinct PG19-train books and both frozen
geometry cells per rank.  The aggregate is a valid positive result and is
eligible as formal evidence:

- all 16 clean geometry/rank cells passed exact semantic and storage gates;
- all 48 preregistered held-out fault trials reached their frozen expected
  gates, with zero expected-gate misses;
- all lifecycle schedules, slot leases, cancellation/reclamation transitions,
  and exact reservation reuse checks passed;
- `sha256sum -c receipts/raw-artifacts.sha256` verified all eight raw shards;
- an independent aggregate replay produced a byte-identical summary; and
- an additional manual replay checked schedules, leases, semantic/storage
  equality, immutable documents, scrubbed physical IDs, fault mappings, and
  shard hashes without trusting producer-side `passed` fields.

The result remains bounded to the scheduler-managed interleaving and frozen
fault set described above.  It does not establish concurrent CUDA-kernel
safety, continuous-batching throughput or capacity, cross-model/runtime
generality, fault-set completeness, or production end-to-end correctness.

## Key bindings

- Valid design preregistration SHA-256:
  `c7e80ff62d68a2d942888d3f0a1c1027d69180ae6aa1726bb49aeccc38019847`
- Static input preregistration SHA-256:
  `2c7480e9301860fd24a87fa8aa05b25360456824181cc9b456b0ee0b855a85eb`
- Runner SHA-256:
  `8a53591e53d4b9ff1efafca60fd2c42f48986c9b9719d60a93dc5a49549f32f4`
- Scheduler contract SHA-256:
  `e9eb78d7981bf2c6a56032774a3bb64904e6e9892dcc16d07e2fb6b42205617f`
- Debug output SHA-256:
  `2a3036face3c1cb90a4f645b7d19dd8b69047ea8aa8e560e54c461516b305702`
- Frozen upstream code ledger SHA-256:
  `7620f05821fc5435a9aaa260ae82577988a5a20eff0f42901b66e6c6871fd2b9`
- Formal launcher SHA-256:
  `8014ed86cf2bcc2bedfafc9cdd5211c2a87273b12fc6f8b4970777df72ce6343`
- Formal summary SHA-256:
  `3fa46011ec65b921ffcf4f36b1294fc01d3c4d6b565e1645ee6da1c7d3600d21`
- Independent replay summary SHA-256:
  `3fa46011ec65b921ffcf4f36b1294fc01d3c4d6b565e1645ee6da1c7d3600d21`
- Raw-artifact ledger SHA-256:
  `5cc929a300d1c9f29bf8310c47bd7448106d27f4ad367647dfd9d952f755f9fd`

The remote debug working directory is
`/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_debug/qcomem-8h20-20260821a/a2_scheduler_interleave`.
The remote formal run directory is
`/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/forkaudit-scheduler-interleave-formal-20260821a`.
