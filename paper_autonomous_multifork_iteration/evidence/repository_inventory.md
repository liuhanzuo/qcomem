# Repository evidence inventory

This inventory was rebuilt without reading the archived manuscript or its reviews.

## Primary experiment

- Anonymous evidence root: `supplement_anonymous/raw_primary/`
- Aggregate: `supplement_anonymous/raw_primary/multifork-resident-summary.json`
- Raw shards: `supplement_anonymous/raw_primary/resident-shards/multifork-resident-shard-{0..7}.json`
- Anonymous integrity ledger: `supplement_anonymous/ANONYMOUS_MANIFEST.sha256`
- Original-byte digest binding: `supplement_anonymous/provenance/original_evidence_digests.json`
- Formal method files: `supplement_anonymous/code/` (18-file executed dependency closure, launcher, and protocol).

## Supporting same-kernel experiment

- Aggregate: `supplement_anonymous/raw_supporting/fair-v2-summary.json`
- Purpose: single-request same-kernel fair control, full-logit parity, and timing/memory boundary. It does not authorize a speedup claim.

## Negative compatibility experiment

- Aggregate: `supplement_anonymous/raw_supporting/cross-backend-negative-summary.json`
- Purpose: documents the failure of a Transformers-eager cross-backend compatibility gate. It prevents the same-kernel result from being generalized to eager-backend bitwise equivalence.

## RR2 ownership-factorial response experiment

- Blind aggregate: `evidence/rr2_formal_w/forkaudit-summary.json`
- Detached receipt manifest: `evidence/rr2_formal_w/detached-receipt-manifest.json`
- Raw-artifact ledger: `evidence/rr2_formal_w/all-raw-artifacts.sha256`
- Live-mutant extract: `evidence/rr2_formal_w/mutant-outcomes.json`
- Model-load closure: `evidence/rr2_formal_w/model-load-closure.json`
- Prior-context tolerance record: `evidence/forkaudit_fp32_calibration_manifest.json`
- Purpose: eight pre-specified PG-19 windows, the clean $2\times2$ KV-by-GDN
  ownership factorial at $N\in\{1,8,32\}$, one pre-execution-selected IEEE-FP32
  full-attention row per rank, and nine live faults with separately rebuilt
  matched-clean controls.
- Validation boundary: the scientific aggregate is valid positive.  The primary scientific run
  completed all shards and blind aggregation, but the post-aggregate recursive
  package audit rejected generated Python bytecode.  A later bytecode-disabled
  attempt stopped before GPU execution on stale governance-test expectations;
  therefore a clean terminal-closure claim is withheld.

## R40 V29 independent live-binding formal evidence

- Evidence/claim IDs: `E-R40-INDEPENDENT-LIVE-BINDING-V29-A` and
  `C-R40-INDEPENDENT-LIVE-BINDING-V29-01`.
- Frozen source package:
  `evidence/r40_independent_live_binding_v29_result_pycache_whitelist_fix/`;
  archive SHA-256 `893202582f3cac7ef9f8b61fc2d5c574c7609c51aa811cf518c488a1f1efd297`
  and source-ledger-file SHA-256
  `4d0563a99997a6d2c0a76ee6694b195599fff0caaa1119d22b4e78ad3ad489b0`.
- Read-only post-run index:
  `evidence/r40_independent_live_binding_v29_postrun_audit_mirror/POSTRUN_INDEPENDENT_AUDIT.json`
  (SHA-256 `7af92c70ec48bd35ad44bf475ddbba556b65e6a7c9cff891674f9688a464450d`).
- Execution identity: QS Job `256220`, Trial `1936087`, Pod
  `qs-256220-1936087-ai-1482497-master-0`, run ID
  `71391b1a7ce85c4dfa8beb18f3c2189a`.
- Formal result: `valid_positive`, eight ranks/eight shards.  In the selected
  N=8 shared-document/materialized-GDN ownership-witness cell, six
  preregistered coordinates were checked at three phases per rank: 144
  selected rows, 12,960 storage rows, 3,840 direct clone-lineage edges, 24
  stable `primary/raw/rank-X/...` phase artifacts, and 96 primary calls with
  zero primary-memory hook events.  No temporary rank path remains.
- Terminal result: the real builder-to-serializer and direct-clone lineage
  checks passed; the exact result-cache authority contains 31 intentional
  `.cpython-311.pyc` files plus 13 parent directories outside the immutable
  source trees; primary `99_done`, formal `COMPLETE`, root `COMPLETE`, the
  1,367-node terminal tree, and the independent read-only post-run audit all
  passed.
- Result anchors: primary summary
  `d49f25ddef31d8a0afffeccba855b05123210b1b1ccdcdc364ebef56ae3e298c`,
  primary scientific ledger
  `ffdd40f02d114ce2a50ddd042701ae4282177de87c3e32875b90bc598e66fd13`,
  formal aggregate
  `feae2481a4cf9e6a45135896741b08a4529d9b264a63622e5e8004cfe766c1fb`,
  formal terminal ledger
  `d814ffa69d9bb1fcb502fa8704edb351606cf1ccba147bd1376caa1ee98f4a10`,
  R40 aggregate
  `40e1b45d715a20222fff6d85344d8fbbd06dbeae6a7d0056462e5d90af53d4fa`,
  CUDA smoke
  `2ac8aa3c6b8324d10ebd01e7bded1631e11d7e1aeb058899204969ffb666a4e9`,
  terminal closure
  `7ba11f6a71e8558eabd82af742e7f4c901ba8ceb9ce9ccd6a3d15e3f9c9610bf`,
  and terminal tree
  `6aadf2d4e066f0e78978c6e216be3ef1ad34f46959f74cba3be79dde91a1f72a`.
- Admissibility history: the immutable V17 HOLD remains diagnostic-only.
  V26, V27, and V28 each ended with `science_accepted=false` and no root
  `COMPLETE`; they remain inadmissible and must not be cited or pooled with
  V29.
- Claim boundary: one honest-process fixed
  Qwen3.5/PyTorch/Transformers/vLLM/Triton/8xH20 execution.  V29 narrows the
  trusted-capture gap for only the selected cell, six coordinates, and three
  phases.  It does not establish completeness elsewhere, malicious-runtime
  capture honesty, device program counters or driver binaries, underlying
  eager ATen/CUDA identity, IPC/framework independence, scheduler,
  concurrent/CUDA-graph behavior, performance, or cross-stack generality.
  Manuscript integration and fresh review are still pending.

## Source and method files

- Paged Q16 arena and partial-tail COW: `supplement_anonymous/code/qcomem_vllm_paged_kernel.py`
- Qwen3.5 cache integration: `supplement_anonymous/code/qcomem_qwen35_vllm_paged_integration.py`
- Functional recurrent-state binding: `supplement_anonymous/code/qcomem_qwen35_native_cache.py`
- Multi-resident construction and ownership audit: `supplement_anonymous/code/qcomem_vllm_paged_multifork_resident.py`
- Formal execution, invariant replay, and aggregation: `supplement_anonymous/code/run_qcomem_qwen35_vllm_paged_multifork_resident.py`
- Frozen protocol and release governance: `supplement_anonymous/code/MULTIFORK_RESIDENT_PROTOCOL_ZH.md` and the launcher beside it.

## Evidence limitations

- One model family and one hardware family.
- One 4095-token, page-size-128 partial-tail geometry; N in {1,2,4,8,16,32}; 32-token queries and 8 greedy generation steps.
- Request objects coexist, but model steps execute serially on one CUDA stream in round-major order.
- No production vLLM engine scheduler, continuous batching, ragged batching, cancellation, eviction, throughput, TTFT, TPOT, NVML, F1/EM, Q8/Q4, multi-document, or aligned-4096 evaluation.
- The fresh control retains the source arena and its N-scaled private reservations, so `80+90N` versus `80+5N MiB` is a controlled ownership contrast, not an optimized production full-copy baseline.
- Eight ranks provide eight distinct books/query banks, but deterministic page-geometry values are not independent statistical memory replicates.
- RR2 remains one model/checkpoint, one H20 hardware family, Q16, one
  4095-token partial-tail geometry, sequential single-stream execution, and
  $N\leq32$.
- The IEEE-FP32 oracle covers one selected full-attention row per rank; it is
  not an independent implementation of the 30 GDN layers or of the full
  end-to-end model.
- The nine live mutants are targeted positive controls and do not establish
  detector completeness over all possible runtime faults.
