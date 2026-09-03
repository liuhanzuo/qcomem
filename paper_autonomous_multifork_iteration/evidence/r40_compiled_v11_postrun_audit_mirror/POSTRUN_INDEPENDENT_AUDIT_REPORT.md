# R40 compiled-dispatch v11 post-run independent audit record

Audit date: 2026-08-28 (Asia/Shanghai)  
Decision: **PASS at the declared fixed-stack scope**

This file records the result of the independent post-run audit of the formal
8xH20 execution.  The audit parsed the remote terminal tree without sampling.
The compact local directory is an **index mirror**, not a replay-closed copy of
all remote products; the reproducibility boundary is stated explicitly below.

## Formal execution identity

- Job / trial: `253976 / 1911962`
- Remote immutable result root:
  `/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/r40-primary-compiled-dispatch-v11-20260827k`
- Frozen source archive:
  `evidence/r40_primary_compiled_dispatch_v11/packages/r40-primary-compiled-dispatch-v11-20260827k.tar.gz`
- Frozen source archive SHA-256:
  `0013e1e458711263342b37c1a274b6a36d227a602a885201f12892a8968b3641`
- Source-ledger-file SHA-256:
  `958e795ef473d87cd9addfc2924cb20c50df2434c0a66069d08ed2ad0d4c08a3`

## Terminal and result anchors

- `formal-binding/formal-aggregate.json` SHA-256:
  `04b5ae63dc2f2dbe7c116a7136c2cdda2d9cab2e433b72b31d57cd28125c7a1f`
- `primary/forkaudit-summary.json` SHA-256:
  `5221d9ae0eb12092e311929fed6269122c290baddc97b6014a69f0266e634353`
- Root 949-entry terminal ledger SHA-256:
  `909d47d38ba3e37f196ceca340b4a0d2e40bbe6b8c494f63bc78286ca217fa5d`
- Formal 169-entry terminal ledger SHA-256:
  `b01d76704b4155d826ebc21fdce8abe85a9ed8ce9aac64c9308feecf49b4e525`
- Raw 536-file ledger SHA-256:
  `cc8a39aedd87ee196dd6424db5403c3b5ac7cc2b86c68b089dfa730989b780de`
- Scientific ledger SHA-256:
  `50097b75ea925cc4ef7b6393113e10bcdf78d1508573dac218652d0270cc4758`
- Runtime preflight SHA-256:
  `e4467acfbf440fff5b9a4c4ca99b46a282edea79f31d9cd44ef5d90036991651`

Both root and formal `COMPLETE` markers were present.  The independent audit
replayed the complete root/formal ledgers and parsed all eight shards, per-rank
receipts/replays, source manifests, oracle records, fault records, and bound
negative controls at the remote result root.

## Closed counts

- 8 ranks, 96 factorial configurations, and 192 execution cells.
- 209,920 registered attention calls.
- 635,520 registered GDN calls: 5,760 document-prefill calls and 629,760
  request calls.
- 8 unique rank/process/GPU-UUID tuples.
- 8 distinct selected compiled-artifact IDs and one selected compile
  configuration.
- 536 raw files: 8 primary shards plus 528 evidence artifacts.
- 28 bound negative controls per rank, 224 total; all rejected.
- The primary aggregate reports all eight selected numerical-oracle ranks
  passing and the matched M1--M9 campaign passing with no escaped, wrong-gate,
  unexpected-crash, or clean-false-positive IDs.

The remote auditor additionally parsed 64 real oracle gates and 8 diagnostic
oracle records.  Those lower-level records are not all copied into the compact
local mirror, so that count is an audit-record statement rather than a locally
replayable computation from this directory alone.

## Authorized claim

On this frozen honest-process 8xH20 run, each registered attention call was
bound before invocation to the selected fully hashed Triton launcher artifact
and configuration, including the exact autotune-selection record or an exact
no-autotuner observation, and was sealed only after normal return on the same
assigned GPU and stream.

This does **not** establish a compiled GDN binary, identity of the eager GDN
path's underlying ATen/CUDA operators, device/driver binary attestation,
malicious-runtime resistance, or cross-model/runtime/hardware generality.

## Local mirror boundary

The compact mirror stores the two top-level result JSON files, the runtime
preflight, both terminal ledgers, raw/scientific ledgers, aggregate audit log,
and completion markers.  Of the root ledger's 949 products, only six
ledger-listed products are locally mirrored.  The eight raw shards, 528 raw
evidence files, per-rank receipts, rank launch identities, GPU-assignment
receipts, preflight logs, supervisor closure, and most other terminal products
remain at the remote result root.

Therefore this directory supports local verification of the mirrored anchors
and records the completed remote audit, but it must not be described as a
locally replay-closed mirror.  Repeating the full independent audit requires
the remote result root or a larger mirror excluding only the approximately
73 GB model-weight view.
