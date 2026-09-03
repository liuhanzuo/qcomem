# Trial 1892234 HYPIC Store recovery acceptance

This directory freezes the audit-honest, cell-scoped recovery of the H20 Store
measurements from QS job `247699`, trial `1892234`, official HYPIC commit
`98147c01909004e66d98bcb18b886927d41b0ee5`.

The platform trial terminal status is `Failed`; the original NFS tree retains
both failure markers and has no `COMPLETED`, stage `30`, stage `99`, original
blind-replay output, terminal static rehash, terminal-idle snapshot, or final
all-artifacts ledger.  This package does not reinterpret the trial as a
successful whole run.  It accepts only the 16 GPU measurement cells that had
already completed before the post-cell replay failure.

## Accepted result

- Stable schema: `hypic-rwd5-trial1892234-external-store-acceptance-v1`
- Status: `passed_external_replay_16_of_16`
- Prefix Cache, ranks 0--7: `[145653760, 145653760, 133857280, 145653760, 146964480, 146964480, 146964480, 146964480]` bytes
- Prefix Cache exact median: `146309120` bytes = `139.53125` MiB
- HYPIC transition-rope-recompute, ranks 0--7: `[339415040, 339456000, 328171520, 339476480, 340193280, 340459520, 340500480, 340418560]` bytes
- HYPIC exact median: `339834880` bytes = `324.091796875` MiB
- HYPIC / Prefix Cache: `33187/14288` = approximately `2.3227183651`
- Prefix Cache / CoMem Q8 (`16664352` bytes): `4572160/520761` = approximately `8.7797665340`
- HYPIC / CoMem Q8: `10619840/520761` = approximately `20.3929249694`

The denominator is the exact target-entry-owned physical tensor-range union;
metadata is excluded.  All 16 pre-snapshot and all 16 terminal Mamba domains
were independently rederived as exact, unique, and duplicate-free.  No global
allocator-correctness, runtime-safety, capacity, NVML/process-allocation, or
preallocated-pool claim is made.

## What was revalidated

`accept_trial1892234_store_recovery.py` performs a fail-closed read of the
original NFS authority.  It validates the exact 16-file sets for raw shards and
pre-measurement Store receipts, the exact 16 terminal receipts, all 16 targets,
server/readiness receipts, scheduler-worker receipts, the preregistration,
static hashes, job/trial/mode/rank/workload identities, runtime manifest, and
the complete receipt hash graph.  It hashes 132 remote members before and after
validation; `validated-artifacts.sha256` records those members.

The verifier hash-pins `replay_trial1892234_external.py`, re-executes it over
the immutable cells, and requires byte identity with
`external-blind-replay.json`.  The sole replay correction accepts an empty
scheduler procfs environment only with the exact `sglang::scheduler` command
line and hash.  Recovery validation additionally requires:

- exact 15-key frontend environment equality and its canonical hash;
- worker `ppid == frontend.pid`;
- exact first ancestry row and a complete, unique parent chain to PID 0;
- full equality between `receipt.authority.scheduler_process` and
  `worker.process`.

`accept_trial1892234_store_cell_recovery.py` is a mechanically copied, narrowly
patched version of the original strict verifier.  Its two documented
producer-representation corrections are:

1. the raw shard holds the exact `/server_info` configuration, while the server
   receipt extends it only with equal `rwd5_expected` and `rwd5_observed` fields;
2. retained-state payloads and slot metadata must be `cuda:0`, while excluded
   HYPIC `token_ids` index metadata must be `cpu`.

The original strict verifier is retained byte-for-byte as
`accept_trial1892234_store_original_strict.py` for comparison.

## Reproduction authority

The recovery verifier was executed on the CPU development pod with the original
read-only NFS boot root:

```text
/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-rwd5-autobootstrap-job247699-trial1892234
```

The exact runtime overlay must be staged at
`/tmp/rwd5-hypic-store-runtime-1892234`; its 77-member manifest is included as
`RUNTIME-SHA256SUMS`.  The verifier writes only its new output files outside the
original boot root.  The authoritative machine-readable result is
`acceptance.json`; `SHA256SUMS` closes this local package.

## Explicitly unrecovered

- original whole-run `COMPLETED` and stage `99` closure;
- original frozen replay completion and stage `30`;
- post-cell whole-run terminal static/model-byte rehash;
- whole-run terminal GPU/process-idle snapshot;
- final original all-artifacts ledger.

Those absent whole-run artifacts remain absent and are not claimed by the
paper-facing Store result.
