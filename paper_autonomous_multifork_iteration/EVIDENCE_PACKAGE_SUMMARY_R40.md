# R40 evidence-package summary

Snapshot: 2026-09-01 (Asia/Shanghai)  
Repository root: `/Users/liuhanzuo/MacLLM-Bench`  
Evidence root: `paper_autonomous_multifork_iteration/evidence`  
Latest manuscript source at this snapshot: `paper_autonomous_multifork_iteration/main_r40_submission_candidate.tex`

This file is a decision-oriented index of the evidence that supports the current
paper and of the R40 packages still under audit. It does not replace each
package's manifest, preregistration, terminal ledger, or replay instructions.
The registry snapshot is `evidence/experiment_registry.json`; the synchronized
claim map is `evidence/claim_evidence_map.tsv`.  Their exact snapshot hashes are
recorded in the final-integrity block at the end of this file so that the hashes
bind the completed summary rather than an intermediate edit.

## 中文结论（先看这一段即可）

- 当前摘要和 Introduction 已以 ownership 问题为主线：一个历史 alias 在
  **8/8** defective cells 中保持 token、FP32 logits、request GDN 和 logical
  KV 精确，却破坏 persistent base；修复后 **8/8** storage-clean。普通
  persistent-base invariant 也能检出该缺陷，ForkAudit 已证明的增量是更早的
  owner/layer/family transition 定位，不是独占检测。
- primary ownership、numerical oracle、固定 fault、historical alias、独立 slot
  census、dual repeat 和 Falcon-H1 bounded transfer 都已有可用的、范围受限的
  正向证据。
- compiled-dispatch v11 已完成 8xH20 正式运行和一次分离的无抽样后审，可以把当前
  论文里的 target 5 从 partial 改成 declared fixed-stack scope 下的 pass；但
  不能扩写成 driver/device attestation、compiled GDN 或跨 runtime 泛化。包内
  没有单独的审计员身份收据，因此不能写成 machine-identified independent audit。
- trusted capture 仍没有被完全消除。live-binding v17 因两个 pre-GPU
  blocker 被 HOLD；v18 因确定性 terminal-closure 漏项被 HOLD。V19 的正式
  launcher 在科学执行前的 87 项测试中有 2 项失败（其余 85 项通过），
  failure ledger 明确 `science_accepted=false`，未进入 CUDA 或 rank science。
  科学 10-file payload 与 v19 **10/10 字节相同**的 v20 随后通过 detached
  focused 162/162 和 stages 00--04，但 package signal test 因 `nohup` 后台
  继承 `SigIgn HUP/INT/QUIT=0x7` 而永久等待。进程树已终止，failure ledger
  记录 exit 143 和 `science_accepted=false`；正式 static、CUDA、rank science
  和 `COMPLETE` 均未发生。V20 因此也是 pre-science HOLD。V21 已以新的 QS
  Job/Trial 正式启动并通过 formal 87/87、162/static/GPU preflight；8 个 rank
  均加载模型进入 science，但在首个 N=8 shared-document/materialized-GDN
  witness 的第一次 generation callback 一致触发 unauthorized rebind gate。
  Failure ledger 为 exit 2、`science_accepted=false`，无 `COMPLETE`、terminal
  closure 或 aggregate。这是 scientific execution 中的 deterministic formal
  protocol/audit-gate negative，不是 infra/preflight，也尚不能称为论文现有主
  结果的反证。随后 diagnostic-only 的 V22 在 QS Job `256220` / Trial
  `1920822` 上完成定位：162/162、87/87、static 132/132 和 formal GPU
  preflight 均通过，8 个 rank 均加载模型；7 份已刷新的 rank 日志一致报告
  `first_coord=(0,'conv',0)`，setup authority 的非紧凑 stride
  `[33546240,1,8192]` 在 runtime endpoint 变成紧凑 stride `[32768,4,1]`。
  rank 7 在 sibling-failure coordination 中退出，未刷新同一行。V22 的
  failure ledger 为 exit 2，且预先固定为 `science_accepted=false`；它只定位
  producer/authority 与 runtime endpoint 的 descriptor 分歧并推动 V23
  producer fix，不是正向证据，也不证明论文现有主结果错误。V23 随后在 QS
  Job `256220` / Trial `1929035` 上通过 162/162、93/93、static 132/132、
  private-model-view、GPU preflight 和 GPU assignment；8 个 rank 均完成
  1,026-weight load，但在 warmup `_build_document_cache` 的全局 post-hook
  触发 `CompactRebindError: cached post-hook used an unwrapped request`。它未
  到达首个科学 cell，也没有 allocator、real-binding aggregate 或
  `COMPLETE`。Failure ledger 为 exit 2、`science_accepted=false`；这是
  pre-science producer-instrumentation failure，不是科学负结果或正向证据。
  V24 随后在 QS Job `256220` / Trial `1936087` 上通过 outer 162/162、package
  94/94、static 132/132、private-model-view 和 GPU preflight；8 个 rank 均加载
  1,026 weights，并明确跨过旧 warmup 断点。正式 3x4 factorial 已推进到至少
  11/12 calls/rank，rank 0/1 完成 12/12 并各提交约 62 MB shard；但 rank 级
  producer receipt 未把预期 fault exception 计入已启动但中止的 cached
  calls，触发 `compact rebind producer coverage drift`。运行在
  `eight_rank_shards` 阶段 exit 2，`science_accepted=false`；没有 aggregate、
  terminal closure 或 `COMPLETE`。这是 post-generation producer-coverage
  终检缺陷，不是科学负结果，也不是正向证据。V25 修复异常调用闭包，但误把
  builder 的 mandatory borrowed construction step 当成最终 borrowed-policy
  request universe；其正式 run 完成并提交 8/8 shards 后，8 ranks 均在同一
  producer gate 失败。Immutable builder 明确每个 request 先执行 borrowed base
  construction，materialized policy 再执行第二步 materialization。因此 v25
  新增的 `borrowed delegated == final borrowed requests` 必然为假；exit 2、
  `science_accepted=false`，仍无 aggregate、terminal closure 或 `COMPLETE`。
  V26 恢复了正确的 construction-step 等式并越过该 gate：8/8 shards、raw
  receipts 和 blind primary aggregate 均完成，primary summary 的所有科学
  判据为 true。但 immutable primary launcher 的 model-load lease 使用
  `python -I -c`，绕过外层 bytecode 环境变量并在只读 code snapshot 下产生
  1 个可写 `__pycache__` 和 14 个 `.pyc`；终端 code-snapshot 审计因此在
  stage 06 后拒绝运行。exit 1、`science_accepted=false`，仍无 formal-binding
  aggregate、terminal closure 或 `COMPLETE`。V26 是 post-science terminal
  governance failure，primary aggregate 也不可引用；V27 只修复该 bytecode
  隔离路径。
  held-out v10 已冻结方法，仍待 fresh audit 和外部 binding。
- `main_r40_submission_candidate.tex` 已同步 compiled-dispatch v11、提升历史
  alias、降级 Store--F1，完成静态校验、29 页逐页视觉检查、三次独立
  PDF-only review（**6/6/6**）和独立 meta-review（**6**）。它是最新 retained
  candidate；按既有 lexicographic gate，Round 23 仍是 best checkpoint。

## Status vocabulary

- **ADMISSIBLE**: the frozen result and its declared boundary have passed the
  applicable integrity/replay checks and may support a bounded paper claim.
- **SUPPORTING**: valid context or motivation, but not primary evidence for the
  main ownership claim.
- **FINAL PASS, NOT YET IN TEX**: a new post-run-audited result that is
  eligible for integration, but the current manuscript still contains the old
  wording.
- **FRESH AUDIT GO; H20 PENDING**: the frozen formal path passed an independent
  pre-run audit and may be submitted, but it is not scientific evidence until
  formal execution and post-run audit pass.
- **FORMAL SCIENCE AUDIT-GATE NEGATIVE; NOT POSITIVE EVIDENCE**: preflight
  passed and the failure occurred deterministically during scientific
  execution, but the run did not reach terminal closure or aggregate acceptance.
- **PRE-SCIENCE PRODUCER-INSTRUMENTATION FAILURE; NOT EVIDENCE**: formal
  preflight and model loading may have passed, but instrumentation failed in
  warmup before the first scientific cell and produced no scientific endpoint.
- **POST-SCIENCE PRODUCER-COVERAGE TERMINAL-GATE FAILURE; NOT EVIDENCE**:
  scientific calls produced partial rank artifacts, but a producer-coverage
  receipt failed before all shards, aggregate acceptance, and terminal closure.
- **POST-SCIENCE TERMINAL-GOVERNANCE FAILURE; NOT EVIDENCE**: the scientific
  shards and a primary aggregate may exist, but a later immutable-code or
  terminal-closure gate failed; no intermediate product is admissible.
- **HOLD / PENDING**: not eligible for a paper claim. A package can remain useful
  as a diagnostic without becoming scientific evidence.

## Executive status matrix

| Evidence block | Status | What it currently supports | Primary anchor |
|---|---|---|---|
| Same-stack H20 Store--F1 deployment | **ADMISSIBLE** | CoMem Q8 and per-layer mixed retained-state reductions at measured mean F1 | `evidence/h20_deployment_benchmark/summary.json` |
| Primary RR2 ownership factorial | **ADMISSIBLE** | 96 fixed configurations, ownership/lifecycle predicates, exact semantic relations, allocator result, nine fixed faults | `evidence/round_04_rr2_package/MANIFEST.json` |
| Expanded captured-boundary oracle | **ADMISSIBLE, BOUNDED** | All attention layers and 12/30 GDN layers at frozen captured inputs; 44 clean rows and 44 seeded controls | `evidence/r30_expanded_oracle_sweep/validation_report.json` |
| Independent GDN capture, preproducer slot census, and dual repeat | **ADMISSIBLE, BOUNDED** | Process-separated observations; producer rows removed as expected-set authority for one fixed GDN cohort; exact repeatability | `evidence/r33_independent_capture/formal_h20/independent_acceptance.json`; `evidence/r39_independent_slot_census`; `evidence/r39_dual_producer_repeat` |
| PDF-only constructed faults and historical alias regression | **ADMISSIBLE, BOUNDED** | Per-fault sensitivity/localization and one known pre-fix defect/repair | `evidence/r33_fresh_faults/formal_h20/RESULT_VERIFICATION.json`; `evidence/r35_historical_alias_regression/formal_h20/RESULT_VERIFICATION.json` |
| Falcon-H1 bounded transfer | **ADMISSIBLE, BOUNDED** | Token/logit/state/ownership agreement on one separately frozen Transformers/H20 configuration | `evidence/r39_falcon_h1_transfer_v2/formal_h20` |
| Primary compiled dispatch v11 | **ADMISSIBLE, BOUNDED; IN R40 CANDIDATE** | Per-call selected Triton launcher artifact/configuration binding and post-return sealing for the frozen 8xH20 run; the no-sampling audit operated on the remote tree, while the local copy is an index mirror; no standalone auditor-identity receipt | Frozen source: `evidence/r40_primary_compiled_dispatch_v11`; post-run mirror: `evidence/r40_compiled_v11_postrun_audit_mirror` |
| Live-binding v17 | **HOLD; DIAGNOSTIC ONLY** | No scientific claim; fresh audit found two pre-GPU self-containment/stable-snapshot blockers | `evidence/r40_independent_live_binding_v17_linux_stage/FRESH_AUDIT_HOLD_REPORT.md` |
| Live-binding v18 | **HOLD; PRE-SCIENCE TERMINAL-CLOSURE DEFECT; NOT EVIDENCE** | The producer writes eight per-rank `invocation.json` files that v18 omitted from its terminal expected-path set; its generic-sleep Trial was eventually allocated and stopped, but no v18 payload stage, CUDA, result, or science was launched | Frozen package: `evidence/r40_independent_live_binding_v18_self_contained_stage`; submission record: `evidence/r40_independent_live_binding_v18_fresh_audit_20260828/qs_submission_record.json` |
| Live-binding v19 | **HOLD; FORMAL PREFLIGHT FAILURE; NOT EVIDENCE** | Its clean stage was built on Trial `1917289`, but the formal preflight passed 85/87 tests and failed two launcher self-tests because outer control variables leaked into child test environments and a 0.2-second signal test could race to normal completion; the failure ledger records `science_accepted=false`, with no CUDA or rank science | Frozen package: `evidence/r40_independent_live_binding_v19_terminal_closure_fix`; operational record: `evidence/r40_independent_live_binding_v18_fresh_audit_20260828/qs_submission_record.json` |
| Live-binding v20 | **HOLD; FORMAL PREFLIGHT SIGNAL FAILURE; NOT EVIDENCE** | The formal path passed detached focused 162/162 and stages 00--04, then hung in the package atomic-signal test because the `nohup` background inherited `SigIgn HUP/INT/QUIT=0x7`; its SIGINT child could not trap and the busy-hold did not exit. The full tree was terminated; failure-ledger exit 143 records `science_accepted=false`. Formal static, CUDA, rank science, and `COMPLETE` did not occur | Frozen package: `evidence/r40_independent_live_binding_v20_preflight_env_isolation`; operational record: `evidence/r40_independent_live_binding_v18_fresh_audit_20260828/qs_submission_record.json` |
| Live-binding v21 | **FORMAL SCIENCE AUDIT-GATE NEGATIVE; NOT POSITIVE EVIDENCE** | Formal 87/87 and 162/static/GPU preflight passed; all eight ranks loaded the model and entered science. At the first generation callback of the first N=8 shared-document/materialized-GDN ownership witness, every rank triggered `r40_real_binding.py:246` (`functional rebind descriptor/offset/interval unauthorized`). Root/ranks terminated; failure-ledger exit 2 records `science_accepted=false`, with no `COMPLETE`, terminal closure, or aggregate. This is not an infra/preflight failure and is not yet a refutation of the manuscript's accepted main result | Frozen package: `evidence/r40_independent_live_binding_v21_signal_disposition_isolation`; operational record: `evidence/r40_independent_live_binding_v18_fresh_audit_20260828/qs_submission_record.json` |
| Live-binding v22 | **COMPLETED DESCRIPTOR DIAGNOSTIC; NOT POSITIVE EVIDENCE** | Job `256220` / Trial `1920822` passed formal 162/162, 87/87, static 132/132, and GPU preflight; all eight ranks loaded the model. Seven flushed rank logs identically locate `first_coord=(0,'conv',0)` and `descriptor_diff=[stride:expected=[33546240,1,8192],current=[32768,4,1]]`; rank 7 exited through sibling-failure coordination before flushing that line. Failure-ledger exit 2 and the frozen diagnostic contract both record `science_accepted=false`. This shows a setup-authority noncompact stride versus compact runtime endpoint and motivates a producer fix; it does not refute the accepted manuscript result | Frozen diagnostic package: `evidence/r40_independent_live_binding_v22_descriptor_diagnostic` |
| Live-binding v23 | **HOLD; PRE-SCIENCE PRODUCER-INSTRUMENTATION FAILURE; NOT EVIDENCE** | Job `256220` / Trial `1929035` passed 162/162, 93/93, static 132/132, private-model-view, formal GPU preflight, and GPU-assignment gates; all eight ranks loaded 1,026 weights. During warmup `_build_document_cache`, before the first scientific cell, the global post-hook raised `CompactRebindError: cached post-hook used an unwrapped request`. Failure-ledger exit 2 records `science_accepted=false`; no allocator result, real-binding aggregate, terminal closure, or `COMPLETE` exists. V24 is a minimal-scope successor under repair | Frozen package: `evidence/r40_independent_live_binding_v23_compact_rebind_fix`; failure-ledger SHA-256 `cb925284e53eb5c3d561a55b1f598bb3f5c48cd636e1417281d4527a2a53d94d` |
| Live-binding v24 | **HOLD; POST-SCIENCE PRODUCER-COVERAGE TERMINAL-GATE FAILURE; NOT EVIDENCE** | Job `256220` / Trial `1936087` passed outer 162/162, package 94/94, static 132/132, private-model-view, formal GPU preflight, and 8-rank 1,026-weight load. All ranks crossed the v23 warmup defect and reached at least 11/12 factorial calls; ranks 0/1 completed 12/12 and committed shards before the producer receipt raised `compact rebind producer coverage drift`. Failure-ledger exit 2 records `science_accepted=false`; only shards 0/1 committed, with no stage 04, aggregate, terminal closure, or `COMPLETE`. The deterministic defect was that expected exceptional fault calls were counted at pre-hook start but omitted from success/abort closure | Frozen package: `evidence/r40_independent_live_binding_v24_persistent_scope_fix`; archive SHA-256 `5c970b56d8795c9b11d24b4a62c97d4d43f4052945e887a4706bf20aa2b89250`; failure-ledger SHA-256 `fb5cbb2057069e120e49e07a65f8806ad513d1cddc1d7c21239285f58d2f31ba` |
| Live-binding v25 | **HOLD; POST-SCIENCE PRODUCER-COVERAGE TERMINAL-GATE FAILURE; NOT EVIDENCE** | Reused Job `256220` / Trial `1936087`; outer 162/162, package 96/96, static 134/134, private-model-view, formal GPU preflight, all 8 weight loads, all 12 factorial calls/rank, and all 8 primary shard commits completed. Every rank then failed the same post-main producer receipt because v25 incorrectly equated mandatory borrowed construction steps with only final borrowed-policy requests. Failure-ledger exit 2 records `science_accepted=false`; no stage 04, aggregate, terminal closure, or `COMPLETE` exists. The eight shards are inadmissible partial products | Frozen package: `evidence/r40_independent_live_binding_v25_mixed_policy_coverage_fix`; approved archive `20260901b` SHA-256 `fc9d02d21bd33669c6706a8c498dbd10d978d3b183ca603216db8c569114d031`; failure-ledger SHA-256 `b3825e07d128bffc69370b353926a75463d86693602b7c4de0ee57723f4b84ba` |
| Live-binding v26 | **HOLD; POST-SCIENCE TERMINAL-GOVERNANCE FAILURE; NOT EVIDENCE** | Reused Job `256220` / Trial `1936087`; outer 162/162, package 97/97, static 134/134, private-model-view, CUDA/GPU gates, all 8 weight loads, all 12 factorial calls/rank, 8/8 shards, detached receipts, and blind primary aggregate completed. The corrected producer gate passed. Terminal code-snapshot audit then found a writable `__pycache__` created by the immutable launcher's isolated model-load lease (`-I` ignored the environment-only bytecode control). Failure-ledger exit 1 records `science_accepted=false`; formal-binding aggregate, terminal closure, and `COMPLETE` are absent. The primary summary and shards are inadmissible | Frozen package: `evidence/r40_independent_live_binding_v26_construction_step_receipt_fix`; archive SHA-256 `902344af0d8e9bc31e407e2740dbe665ed29bf98879c73f0d8dae6f6d2263ad3`; failure-ledger SHA-256 `773538aaeadcaad8ef5e1484803bffe0593a852889fd55b05eb5a368daad721b` |
| Live-binding v27 | **HOLD; POST-SCIENCE LIVE-BINDING FINALIZER PATH FAILURE; NOT EVIDENCE** | Reused Job `256220` / Trial `1936087`; package 98/98, static 135/135, detached 162/162, primary 13/13, private-model-view, CUDA/GPU gates, all 8 weight loads, 8/8 shards, stages 04--06, primary `99_done`, and formal-binding `COMPLETE` all passed. The command-line `-B` wrapper eliminated the v26 bytecode contamination. The later R40 finalizer nevertheless followed receipt paths containing the producer's ephemeral `.forkaudit-rank-*` staging directory after that directory had been atomically published to stable `primary/raw/rank-*` paths. All 24 stable artifacts exist and match the recorded bytes/SHA values, but the finalizer rejected the stale paths; no R40 aggregate, terminal closure/tree, or root `COMPLETE` exists, and the exit-1 ledger records `science_accepted=false` | Frozen package: `evidence/r40_independent_live_binding_v27_no_bytecode_python_fix`; archive SHA-256 `241c7c80cf24c7bdd5d40c774fec6cd56bb79e7dd3013cc6f8781c4371ad1c73`; failure-ledger SHA-256 `94cfd5a33e6b776e2446fd919567a469d2fbe40f99a64eb80d39191d6b1d0e79` |
| Live-binding v28 | **HOLD; POST-SCIENCE TERMINAL-CLOSURE MODEL FAILURE; NOT EVIDENCE** | Reused Job `256220` / Trial `1936087`; all eight shards and the primary aggregate completed with `valid_positive=true`, primary reached `99_done`, and formal-binding published its aggregate, terminal ledger, and empty `COMPLETE`. The R40 aggregate also published. The next `expected-paths` gate failed because its model allowed only the `primary/pycache` directory while the immutable launcher's intentional result-sink `-m py_compile` step had produced exactly 31 `.cpython-311.pyc` files and 13 parent directories there. Exit-2 records `science_accepted=false`; no root `COMPLETE` or terminal closure/tree exists | Frozen package: `evidence/r40_independent_live_binding_v28_published_phase_path_fix`; archive SHA-256 `23fcdfc329c16308b15be748c2eda754223c55b273c850cc417502e44ec74393`; failure-ledger SHA-256 `ea993d14a7f118a8f08bc0127ffed0e3c4c06824f0ae704ce5512f3d34510807` |
| Live-binding v29 | **FINAL PASS, NOT YET IN TEX; FRESH ADMISSIBLE RESULT** | Job `256220` / Trial `1936087`, run `71391b1a7ce85c4dfa8beb18f3c2189a`: 8/8 shards, `valid_positive`, primary/formal/R40 aggregates, exact 31-pyc/13-directory result-cache authority, 1,367-node terminal tree, empty root/formal `COMPLETE`, and no failure ledger. Independent post-run attempt 2 passed. Attempt 1 was only a checker defect that mistook the fixed `0.005` threshold for the observed maximum; the corrected check confirms observed max `0.0017432502481433169 <= 0.005` | Frozen package: `evidence/r40_independent_live_binding_v29_result_pycache_whitelist_fix`; post-run mirror: `evidence/r40_independent_live_binding_v29_postrun_audit_mirror`; archive SHA-256 `893202582f3cac7ef9f8b61fc2d5c574c7609c51aa811cf518c488a1f1efd297` |
| Held-out method v10 | **HOLD PENDING FRESH AUDIT AND EXTERNAL BINDING; NOT EXECUTED** | Non-overwriting successor that repairs v9's method surfaces; no operator binding, fault identities, configuration, GPU result, or paper claim | `evidence/r40_heldout_fault_v10_method_freeze` |
| Local CI replay/storage accounting | **SUPPORTING** | Warm-cache Mac CPU replay time and logical package size, not H20 capture or online overhead | `evidence/r40_ci_cost_accounting_v1/aggregate.json` |
| Mac M4 control | **SUPPORTING / MOTIVATION ONLY** | Demonstrates that agreement within the split family does not imply agreement with vanilla dense | `evidence/mac_m4_motivation/summary.json` |
| vLLM/SGLang/HYPIC same-slice controls | **SUPPORTING, UNPOOLED** | Within-framework cache-on/off timing/quality context and separate retained-state receipts | `evidence/related_work_same_protocol/serving_panel_summary.json` |

## 1. Headline memory--quality evidence

The strongest engineering result is the same-checkpoint H20 Store--F1 panel.
The source is `evidence/h20_deployment_benchmark/summary.json` (SHA-256
`b769f94c6677902b99cb2c23a484f098165940d5cb0706af72a3039eccd6c47e`).
It uses Qwen3.5-35B-A3B, PyTorch 2.11.0+cu129, Transformers 5.14.1,
eight H20-3e GPUs, eight validation items from Qasper/2WikiMQA, a 4,096-token
input cap, greedy decoding, at most 32 generated tokens, and three timing
repeats per configuration.

| Configuration | Median Store (MiB/document) | Median TTFT (s) | Median TPOT (ms) | Median tok/s | Mean F1 | Authorized reading |
|---|---:|---:|---:|---:|---:|---|
| Vanilla dense recompute | 0.000000 | 0.649197 | 648.747841 | 1.357901 | 39.423172 | No retained document entry; not the semantic reuse comparator |
| Full-prefix KV cache, Q16/BF16 | 140.341797 | 0.163424 | 54.107524 | 13.380442 | 39.137225 | Matched full-prefix semantic comparator |
| CoMem Q8 (`qcomem-d7-r8-a8-l8`) | 15.892365 | 0.672933 | 55.442583 | 7.150262 | 39.137225 | **88.68% less Store**, mean-F1 delta **0.000** |
| CoMem Q4/Q4/Q8 (`qcomem-d7-r4-a4-l8`) | 10.007355 | 0.674121 | 54.971044 | 7.189570 | 39.012581 | Intermediate Store--F1 point |
| CoMem per-layer mixed (`qcomem-d7-mixed`) | 9.741730 | 0.673439 | 55.070814 | 7.193113 | 39.114823 | **93.06% less Store**, mean-F1 delta **-0.022** points |
| CoMem Q4 (`qcomem-d7-r4-a4-l4`) | 8.413605 | 0.673606 | 55.015483 | 7.185664 | 36.090443 | Smallest Store row; not the headline quality-preserving point |

The paper-facing headline is therefore:

> Relative to full-prefix Q16 reuse, the frozen per-layer CoMem policy reduces
> median retained-document tensor payload from 140.34 to 9.74 MiB/document
> (93.06% less), while measured mean F1 changes by -0.022 points. CoMem Q8
> retains 15.89 MiB/document (88.68% less) at the same measured mean F1.

This is a **Store--F1** claim. Store is the union of retained document tensor
payload under the package's target-entry-owned physical byte-range denominator.
It excludes Python metadata, allocator/preallocated pools, NVML/process deltas,
admission capacity, and timing. Full-prefix Q16 is faster than the reported
CoMem rows, so this evidence does not authorize a latency-speedup claim.

The eight workload shards and their hashes are enumerated inside the summary.
The local raw directory is
`results/gpu-deployment-validation-20260812i/`; its aggregate is
`deployment-summary.json`.  Exact machine-readable fields are
`.rows[].persistent_store_mib_median`, `.ttft_seconds_median`,
`.tpot_ms_median`, `.throughput_tokens_per_second_median`, and
`.longbench_f1_points_mean`.  The upstream deployment summary is hash-bound as
`1574741fdebe9b378196ea70f4af0efc7da1ba7941b543bb67519e596d10c835`;
the GPU-inventory receipt is
`50eef376f4c8f4325924cceca941831f45efdb159930b61a954a30530d51b415`.
Each workload shard used one H20-3e; this was not 8-way tensor-parallel serving.

## 2. Primary ownership factorial and allocator evidence

The replayable RR2 package is
`evidence/round_04_rr2_package`. Important anchors are:

- `MANIFEST.json`: SHA-256
  `51346e18c2d2685ea57712d1823e6056ea6bea11a5718da6d24f2fe1d1b65338`
- `derived/derived_summary_v2.json`: SHA-256
  `865b6b3d3fb4cff8fa21149b9752b8fbb67340562beb6928146ea07e04d53c78`
- `upstream/forkaudit-summary.json`: SHA-256
  `8700901ad7423d215e9e9e81a709e976f43963752e1b9f3d64441412b390d2bc`
- Offline replay: `cd evidence/round_04_rr2_package && ./replay/run_replay.sh`

Admissible results at the declared fixed scope:

- 8 ranks x 3 fan-outs x 4 KV-by-GDN arms = **96 configurations** and
  **192 rebuilt allocator/witness cells**.
- Tokens, full logits, final logical KV, final GDN, both fixed-axis
  projections, and **288 adjacent-fan-out comparisons** match exactly.
- The selected attention oracle has maximum relative L2
  `0.0017432502481433169`, below the frozen `0.005` tolerance.
- All **9/9 matched clean controls pass**, and all **9/9 fixed mutants** reach
  their expected gate.
- In the materialized-GDN, N=32 comparison, shared-document KV reduces the
  post-priming allocated delta from **4.901 to 2.229 GiB (54.5%)** while the
  registered observables remain canonically equal.

The complete N=32 allocator panel is:

| KV ownership / GDN setup | Final allocated delta (GiB) | Peak delta (GiB) | Generation increment (GiB) |
|---|---:|---:|---:|
| Full-copy / materialized | 4.901 | 4.920 | 0.019 |
| Full-copy / borrowed | 4.890 | 4.907 | 1.951 |
| Shared-document / materialized | 2.229 | 2.843 | 0.019 |
| Shared-document / borrowed | 2.229 | 2.843 | 1.950 |

Boundary: this is a frozen Qwen3.5/H20 ownership case study and an offline replay
of archived artifacts. It is not runtime-independent evidence, a production
scheduler benchmark, a capacity measurement, or an estimate of unseen-fault
recall.

### Seven formal targets after compiled-dispatch v11

The targets are the paper's R1--R7 contract obligations, not seven independent
experiments.  With v11, their bounded status is:

`evidence/seven_target_status.json` is a preserved 2026-08-20 pre-v11 snapshot
whose R5/overall fields are still `partial`; it is **not** the authority for the
R40 post-v11 status below.  The v11 formal aggregate, registry entry
`E-R40-PRIMARY-COMPILED-DISPATCH-V11-A`, and synchronized claim-map row are the
newer authorities.  A later artifact-maintenance pass should publish a
separately named post-v11 status snapshot instead of silently rewriting the old
file.

| Target | R40 status | Decisive local anchor | Boundary |
|---|---|---|---|
| R1 Frozen identity | complete / pass | `evidence/rr2_terminal_closure_a1_20260821/formal_refreeze_20260821a/final-formal-trial1872962-terminal-validation.json` | One frozen primary factorial and its terminal closure |
| R2 Prefix immutability | complete / pass | `evidence/round_04_rr2_package` | Captured registered phases only |
| R3 Private ownership | complete / pass | `evidence/round_04_rr2_package` | Pointer-free conservative intervals at registered points |
| R4 Tail-safe append | complete / pass | `evidence/round_04_rr2_package` | One partial-tail geometry |
| R5 Dispatch provenance | complete / pass at declared scope | `evidence/r40_compiled_v11_postrun_audit_mirror/formal-binding/formal-aggregate.json` | Compiled attention launcher/config binding; eager-route GDN only |
| R6 Cross-arm equivalence | complete / pass | `evidence/round_04_rr2_package` | One primary sequential schedule |
| R7 Cross-N consistency | complete / pass | `evidence/round_04_rr2_package` | Relational consistency, not an independent oracle |

R1's compact terminal-closure package records Job/Trial `246643/1872962`, a
valid-positive eight-rank run, 36-file preregistration/terminal code ledgers,
14-entry model lease closure, and a byte-identical independent replay.  Key
bindings are runner
`9da619fc037e2c670b146d778fd9f4d5344212b7e525f3d3f26a077f79d67775`,
launcher
`dd824426cad57cf0a43a488b0b776fa3afa4e988c893bb27f98956b425d45c05`,
code ledger
`c18d63c72acaf00d20006278998052c12b3894d116d1724ab855f77d04af011a`,
formal/independent summary
`5c510e59e62181b8c31dc722a7fca1337a5972b8533cd7ff556d266cb29e34c0`,
and scientific ledger
`d5cb071c39296232327cf216112f789662c312536a75dfa95f82731745403654`.
The compact package's `SHA256SUMS` has SHA-256
`74ae8f4f562e49d754f3513b6be9ac2d4e6c33b4a9bac18db4a150cf65a1b884`.

“Seven pass” remains conditional on honest and complete capture and on each
row's fixed boundary.  It does not mean compiled GDN identity, device-binary
attestation, malicious-runtime resistance, or cross-stack generality.

## 3. Numerical falsification evidence

`evidence/r30_expanded_oracle_sweep/validation_report.json` (SHA-256
`39aa442b88d357f06f022e51f5336e14dfcb3c51e8a6f90283f4e0285d26850d`)
records a fully preregistered, candidate-import-free CPU/NumPy FP32 replay from
frozen captured inputs:

- 20 attention rows spanning all 10 full-attention layers and 160 query
  positions: all clean rows pass; maximum relative L2 is `0.0018974`.
- 24 GDN rows spanning 12 of 30 recurrent layers and 192 token transitions: all
  clean rows pass; maximum output/state relative L2 is `0.0020726` /
  `2.0849e-7`.
- All 20 attention and 24 GDN seeded wrong-operator controls are rejected.

Boundary: the replay begins at producer-captured post-RoPE attention and
post-native-q/k-normalization recurrent inputs. It does not independently
validate upstream activation construction, all GDN layers, downstream logits,
end-to-end semantics, capture honesty, or another stack.

## 4. Capture independence and repeatability

Three separate bounded packages progressively narrow, but do not remove, the
capture trusted-computing-base assumption.

1. `evidence/r33_independent_capture/formal_h20/independent_acceptance.json`
   (SHA-256
   `d730793e9cf57fabeccdc5d0dba16ef7ba2da8b64c3e74c109bdb4da5134d1b0`)
   records two fresh N=2 GDN-policy cells on one H20: 1,080 receiver-derived
   observations, 96,660 pair relations, 6/6 phase verdicts, and 2/2 lifecycle
   verdicts pass in spawned observer processes.
2. `evidence/r39_independent_slot_census/formal_run_trial1907355_20260826a.tar.gz`
   (SHA-256
   `09e484ff9c05bbaaacb84637694e856e554d742ed057a1eea269701da9f38496`)
   derives the exact 180-slot set before producer start. Six captures close all
   1,080 rows and 96,660 relations; omission, duplication, and semantic-relabel
   controls fail closed. Producer manifests and rows are not used as expected-set
   authority.
3. `evidence/r39_dual_producer_repeat/formal_h20/r39-dual-producer-repeat-20260826a-formal-complete.tar.gz`
   (SHA-256
   `85bf3cfd3e960c6d9d91d5907ba0a7e1fe62d2838711eb250883f1133d2ab5cd`)
   repeats the same frozen producer implementation in two fresh serial
   processes with four distinct observers and reproduces the semantic
   coordinates, byte digests, descriptors, and relation labels at zero
   tolerance.

The remaining trusted boundary is explicit: correct slot-ID-to-live-tensor
binding, honest/complete producer capture, PyTorch tensor/storage semantics,
CUDA-IPC semantics, and paused capture. These packages do not establish
malicious-producer resistance or an independently executed receiver model.

## 5. Fixed fault and historical-defect evidence

### Designer--executor-separated constructed faults

`evidence/r33_fresh_faults/formal_h20/RESULT_VERIFICATION.json` (SHA-256
`f92a872d8aece11b2f846b02f23632bf1b7ac6d8bee1982b99959d60c5bc8e89`)
binds the result archive
`r33-fresh-faults-20260825b-result.tar.gz` (SHA-256
`0b8261a1c47dcf861379d2bb1629e84c804afb4f6719fdaa7e674a0dcef18441`).
Five PDF-only faults were frozen before executor preparation; all 5/5 matched
clean cases pass and all 5/5 mutants fail first at their frozen primary
predicate. Four of five preserve the surfaced token sequence.

### Historical alias regression

`evidence/r35_historical_alias_regression/formal_h20/RESULT_VERIFICATION.json`
(SHA-256
`4f04f6fdd630042aac76b9d23877bb3849664a42ea8c6edac51944fab58bd765`)
binds the formal output archive (SHA-256
`93e177c7cb483aa3c1f02ec7e602f8af2fbc33fefe8deee45189af0963e4317d`).
Across eight frozen cells, output and terminal state remain exact while the
pre-fix path violates persistent-base ownership; the repaired path is
storage-clean in 8/8 and matches the materialized control in 8/8.

These packages establish sensitivity/localization for their fixed cases and one
known defect mechanism. They do not estimate prevalence, precision, recall,
false-negative rate, or a natural defect population.

## 6. Bounded second-configuration transfer

`evidence/r39_falcon_h1_transfer_v2/formal_h20/r39-falcon-h1-transfer-v2-formal-h20-20260827a.tar.gz`
has SHA-256
`6cbcf860120078e743eb759e2bead74a3bf980e07c4c16f588ee735d3662d6c3`.
For the frozen Falcon-H1-0.5B-Base / Transformers 5.14.1 / H20 naive path,
8/8 ranks pass 96/96 generated-token comparisons, 96/96 full-vocabulary FP32
logit comparisons, 13,824/13,824 semantic-state rows, 192/192 ownership checks,
16/16 cross-arm checks, 8/8 cross-N checks, and 40/40 registered controls.

This is evidence for one separately declared configuration only. It is not a
claim of runtime independence and carries no performance, memory-saving, broad
quality, or production-serving conclusion.

## 7. New compiled-dispatch evidence (R40 v11)

Frozen local package:

- Directory: `evidence/r40_primary_compiled_dispatch_v11`
- Archive: `evidence/r40_primary_compiled_dispatch_v11/packages/r40-primary-compiled-dispatch-v11-20260827k.tar.gz`
- Archive SHA-256:
  `0013e1e458711263342b37c1a274b6a36d227a602a885201f12892a8968b3641`
- Source-ledger-file SHA-256:
  `958e795ef473d87cd9addfc2924cb20c50df2434c0a66069d08ed2ad0d4c08a3`

The frozen package's own `README.md`, `acceptance.json`, and
`preregistration.json` intentionally retain their pre-run HOLD state. They were
not rewritten after execution. The separate local post-run anchor is
`evidence/r40_compiled_v11_postrun_audit_mirror`; its minimal mirror archive has
SHA-256
`c9ef02c21ce782bef65dde1ad76fd18e8fda233e7d97d6fd20ea22428c99929d`.
Its post-run audit record is `POSTRUN_INDEPENDENT_AUDIT_REPORT.md` (SHA-256
`08832775a6da0960f6593fc6bae587301875016cc94acd48bf8e0729847ae5cd`).
The two primary result anchors are:

- `formal-binding/formal-aggregate.json`: SHA-256
  `04b5ae63dc2f2dbe7c116a7136c2cdda2d9cab2e433b72b31d57cd28125c7a1f`;
- `primary/forkaudit-summary.json`: SHA-256
  `5221d9ae0eb12092e311929fed6269122c290baddc97b6014a69f0266e634353`.

The mirror contains the completed aggregates, both `COMPLETE` markers, runtime
preflight, raw/scientific ledgers, and both terminal ledgers.  It is an index
mirror, not a locally replay-closed result tree: only six of the root ledger's
949 listed products are copied locally.  The raw shards, per-rank receipts,
rank/GPU identities, lower-level oracle/control records, and supervisor closure
remain at the remote result root.  It deliberately excludes the approximately
73 GB model-weight view.

The formal run used 8 H20-3e GPUs on the pinned Qwen3.5/PyTorch/Transformers/
vLLM/Triton stack. At the remote immutable result root, a separate post-run
audit turn replayed the full terminal ledgers and parsed every shard, receipt,
oracle, mutant, and negative-control record without sampling. Final status:
**PASS**. The package does not contain a standalone cryptographic or
machine-identified auditor-identity receipt, so the evidence should be called a
recorded no-sampling post-run audit, not identity-attested independent audit.
Repeating the full audit requires the remote tree or a larger local mirror; the
compact local mirror alone cannot replay every lower-level record.

Remote result root:
`/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/r40-primary-compiled-dispatch-v11-20260827k`
(Job/Trial `253976/1911962`).

Full closure counts:

- 8 ranks, 96 configurations, 192 cells.
- 209,920 unique attention call IDs and receipt digests.
- 635,520 GDN calls: 5,760 prefill and 629,760 request calls.
- Zero dense fallback in the declared aggregate/frozen contract.
- 8 distinct rank/PID/GPU-UUID identities; 8 distinct compiled artifact IDs;
  one selected compiled configuration.
- 536 raw files = 8 shards + 528 evidence artifacts, all hash/size closed.
- The remote audit parsed 64 real oracle gates plus 8 diagnostics; M1--M9 hit
  the expected gates with restoration; all 224 bound negative controls were
  rejected.  The 64+8 lower-level count is recorded by the audit report but is
  not independently recomputable from the compact mirror alone.
- Root terminal ledger: 949 entries, SHA-256
  `909d47d38ba3e37f196ceca340b4a0d2e40bbe6b8c494f63bc78286ca217fa5d`.
- Formal terminal ledger: 169 entries, SHA-256
  `b01d76704b4155d826ebc21fdce8abe85a9ed8ce9aac64c9308feecf49b4e525`.
- Scientific/raw ledgers:
  `50097b75ea925cc4ef7b6393113e10bcdf78d1508573dac218652d0270cc4758` /
  `cc8a39aedd87ee196dd6424db5403c3b5ac7cc2b86c68b089dfa730989b780de`.

Admissible narrow claim:

> On this frozen honest-process 8xH20 run, each registered attention call was
> bound before invocation to the selected fully hashed Triton launcher artifact
> and configuration, including the exact autotune-selection record or an exact
> no-autotuner observation, and was sealed only after normal return on the same
> assigned GPU and stream.

Explicit exclusions: this does not attest device/driver binaries, resist a
malicious runtime, establish compiled GDN or underlying ATen-CUDA identity, or
support cross-runtime/model/hardware generality. GDN routes are closed at the
declared mutually exclusive eager-route level.

The current `main_r40_submission_candidate.tex` integrates this result at the
declared boundary across the abstract, contract, results, limitations,
reproducibility statement, conclusion, and appendix maps. Its validator and
blind panel explicitly reject device completion, driver/device binary identity,
underlying ATen/CUDA identity, compiled GDN, and cross-stack interpretations.

## 8. Packages on HOLD or still pending

### Live-binding v17: HOLD

Archive SHA-256:
`ea2067b60b38576ce050da2e5fae4fa42da288843ea94aa789830d8d0cd73a43`.
The reproducible fresh-audit record is
`evidence/r40_independent_live_binding_v17_linux_stage/FRESH_AUDIT_HOLD_REPORT.md`
(SHA-256
`fdd33a13a99dad8ca09735934524c3c2e0afda1f0cd490215f991b2ba9fd5673`).
The exact clean-stage construction itself closed 260 canonical v6 members,
130 retained logical members, 130 exact AppleDouble exclusions, 39 overlay
members, and a final 138-file/37-directory tree with zero AppleDouble paths.
The audit nevertheless found two blockers before authorizing any H20 run:

1. The exact clean stage is not self-contained: one formal preflight test reads
   an unstaged v16 sibling. With the externally supplied canonical v6/overlay
   archives, the exact stage discovers 86 tests, obtains 85 passes and one
   `FileNotFoundError`, with zero skips.
2. The formal-launcher builder approves `a.v6` through one `read_bytes()` call
   and later reopens the pathname with `read_text()` for transformation.  The
   approved and consumed bytes are not one stable snapshot.

Therefore v17 is diagnostic only. No CUDA initialization or formal scientific
result was accepted from it. A separate v18 package was frozen to verify the
10-file scientific manifest self-containedly and to consume one stable
`O_NOFOLLOW`/`fstat`-checked byte snapshot. Its archive is
`evidence/r40_independent_live_binding_v18_self_contained_stage/packages/r40-independent-live-binding-v18-self-contained-stage-20260828a.tar.gz`
(SHA-256
`a775458bd171bf365193800b886bc5140c5caf6fcb375959d1e2b5a119431475`),
with a 38-entry source ledger whose ledger-file SHA-256 is
`c28418d468aeb2b2269643584e28881d121c018ae2c11a45364a0a255e86e308`.
Local construction reports 86/86 clean-stage tests and 128/128 static checks.
A separate fresh read-only audit independently reconstructed the overlay, clean
stage, stable launcher snapshot, and frozen science, and initially returned
**GO_TO_FORMAL_H20**. The audit report SHA-256 is
`f178c6a211352a97744b172e84c3b84dd13b76e4d29d6561dc8d5b474b189562`;
its machine-readable approval SHA-256 is
`e6ff23bd0f1a8873f596ef5eadb88251d9de7578ac86b69e2c8637a53254e821`.
Before any payload staging or scientific launch, a producer-accurate closure check then
found that v18 omitted exactly
`compiled-dispatch-capture/rank-{0..7}/invocation.json` from the terminal
expected-path set. The frozen producer always emits those files, so v18 would
run science and then deterministically fail before top-level completion. Trial
`1916846` was later allocated only as a generic-sleep container and was stopped;
no v18 clean stage, launcher, CUDA initialization, result root, or scientific
execution occurred. V18 is a permanent pre-science HOLD and is not evidence.

V19 is the non-overwriting controlled successor at
`evidence/r40_independent_live_binding_v19_terminal_closure_fix`. Nine of ten
payload files remain byte-identical to v18; only
`executed_source/r40_tree_closure.py` changes. It admits exactly the eight fixed
invocation paths and binds each invocation's canonical schema/rank, runner
SHA-256, argv, primary-shard SHA-256, and canonical argv SHA-256 to the
corresponding formal receipt. Its 38-entry source-ledger file SHA-256 is
`c80db234d2d4cba1472d4f530009873e618cf7f2cb3dfc02dcffc034e708e012`;
the deterministic overlay archive SHA-256 is
`faa2bde71bcf50a7b5c5ca195ed08b7c55074feea61766f52b9f5cb49ae88384`.
Independent archive and closure audits returned GO after exact clean staging,
87/87 zero-skip tests, 131/131 static checks, deterministic rebuild, and
producer-accurate path reconstruction. Formal Trial `1917289` (Job `255481`)
subsequently acquired Pod
`qs-255481-1917289-ai-1462969-master-0`; the reserved backup Trial `1917331`
was stopped. The exact v19 stage contained 175 nodes (138 files plus 37
directories) with zero AppleDouble paths. Its formal launcher nevertheless
stopped in preflight: 85 of 87 tests passed, while two tests in
`tests/test_frozen_import_and_packaging.py` failed because launcher-control
variables from the outer formal process leaked into child self-tests and the
0.2-second signal test could race to normal exit. The machine-readable failure
ledger records exit code 1, `science_accepted=false`, and
`HOLD_PENDING_FRESH_AUDIT_AND_H20`. No CUDA initialization or rank science
occurred, so v19 is an infrastructure/preflight failure and **not scientific
evidence**.

V20 is the non-overwriting governance-only successor at
`evidence/r40_independent_live_binding_v20_preflight_env_isolation`. Its ten
scientific payload files are byte-identical to v19 (10/10); only the preflight
test-control surface and mechanical v20 stage/result/package identifiers
change. The child self-tests now start from an explicitly scrubbed
launcher-control environment, and the signal regression waits for the child to
be ready before sending the signal. The source-ledger file SHA-256 is
`c5535a51edfeefdc6bf9fbfad13271185c0f13a6ea51289d76d41267601b21be`;
the deterministic archive SHA-256 is
`9f0162de487931de9004f965c98c1d55455145d3795859cea82e4da8f4d86db7`.
Local and exact-stage suites passed 87/87 with zero skips, the static audit
passed 132/132, and a fresh independent audit returned GO. V20 was transferred
to the same Trial/Pod, staged as an exact 175-node tree (138 files plus 37
directories), and started under the newly owned one-shot marker
`/tmp/r40-v20-formal-launch-used` with launcher PID `3215`. The remote formal
path passed the detached focused suite 162/162 and stages 00--04, then hung in
the package atomic-signal test. Because the formal process was started under
`nohup`, the background shell inherited `SigIgn HUP/INT/QUIT=0x7`; the SIGINT
self-test child could not install the expected effective behavior, and its
busy-hold loop never exited. The operator terminated the entire process tree.
The exclusive failure ledger records exit code 143 and
`science_accepted=false`; its SHA-256 is
`14ffd7730b84ad7a9d019de8fb9a23ba17205ae92490a3a39ab7c74d0aef85d3`.
The formal 132-check static phase did not run, CUDA and rank science did not
start, and `COMPLETE` does not exist. V20 is therefore a **pre-science HOLD**
and not evidence.

V21 is the minimal non-overwriting signal-disposition successor at
`evidence/r40_independent_live_binding_v21_signal_disposition_isolation`. Its
deterministic archive SHA-256 is
`a2d643cbd6a2b33a2ceabd0c8e91892b36041c126e31393a57f8c254c4edd642`,
and its source-ledger file SHA-256 is
`50d1f3d62e3a9e0d69754f6487b0113b31e06a49040766efa2dd75cf6e30e18e`.
It was submitted as QS Job `256090` / Trial `1920306` to `RL_main` queue 408
with resource package 183, overuse enabled, and one 8xH20 worker. Pod
`qs-256090-1920306-ai-1466108-master-0` built the exact 175-node stage (138
regular files plus 37 directories, zero AppleDouble paths), and the remote
frozen verifier passed.

The formal launcher was detached by a foreground Python `fork` plus `setsid`
operation, without `nohup` or shell `&`; formal PID/SID `4552` runs under PPID
1. Formal 87/87 plus the 162/static/GPU preflight gates passed, and all eight
ranks loaded the model and entered formal science. At the first generation
callback of the first N=8 shared-document/materialized-GDN ownership witness,
rank 0 through rank 7 consistently stopped at `r40_real_binding.py:246` with
`functional rebind descriptor/offset/interval unauthorized`. The observed
secondary allocator nonrecovery followed abnormal strong-reference/traceback
retention and is not an independent primary cause.

The root and all ranks terminated and the GPUs returned idle. The formal
failure ledger records exit code 2 and `science_accepted=false`; `COMPLETE`,
terminal closure, and the aggregate are absent. This is a deterministic formal
protocol/audit-gate negative during scientific execution, not an
infrastructure/preflight failure and not positive evidence. It also cannot yet
be called a refutation of the manuscript's existing accepted main result.

### Live-binding v22 descriptor diagnostic: completed, diagnostic only

V22 is the non-overwriting coordinate/field-level diagnostic at
`evidence/r40_independent_live_binding_v22_descriptor_diagnostic`. Relative to
v21, its sole controlled scientific-code change adds the first failing
coordinate and per-field expected/current values to the existing functional-
rebind error; the equality predicate is unchanged. Its deterministic `b`
archive SHA-256 is
`cb175499d97a656ac52c353b61b146b5f282e3a092d6ca913254ba36dbdd881c`,
and its source-ledger file SHA-256 is
`8f1ad35ca55d46b918811e5879bee55a1752ee213aaa425580ec780935be12ee`.
The package is frozen as `diagnostic_only=true` and
`science_accepted=false` regardless of its runtime outcome.

It ran as QS Job `256220` / Trial `1920822` on Pod
`qs-256220-1920822-ai-1466672-master-0`. The formal launcher had PID/SID
`1099`. Formal 162/162 and 87/87 suites, static 132/132, and formal GPU
preflight passed; all eight ranks loaded the model and entered the diagnostic
science path. Seven rank logs flushed the identical line
`first_coord=(0,'conv',0); descriptor_diff=[stride:expected=[33546240,1,8192],current=[32768,4,1]]`.
Rank 7 exited in sibling-failure coordination before flushing the same
diagnostic line, so the record contains seven identical observations rather
than an unsupported eighth copy. The formal failure ledger records exit code 2
and `science_accepted=false`.

This diagnostic shows that setup materialization preserved a noncompact
stride in the recorded authority, while the corresponding runtime endpoint
was compact. It therefore motivates a v23 producer fix that makes the
authority-producing state transition match the actual endpoint contract. V22
does not supply positive evidence, terminal acceptance, or a basis for saying
that the manuscript's accepted main result is false.

### Live-binding v23 compact-rebind fix: pre-science instrumentation HOLD

V23 is the non-overwriting producer-fix successor at
`evidence/r40_independent_live_binding_v23_compact_rebind_fix`. Its frozen
archive SHA-256 is
`33d2762dabc933e0f5e63644015c9c95e71a837b10de8a6806ae49b1d69fd615`,
and its source-ledger file SHA-256 is
`6e7a95a4404ddadd2685efb1547f03ed73b43bc1fc4dcf13a659e02185e5562a`.
It keeps the v22 binding verifier unchanged and attempts to canonicalize the
materialized setup endpoint plus the cached convolution/recurrent producer
endpoints without changing the native updater identities or route counts.

It ran as QS Job `256220` / Trial `1929035` on Pod
`qs-256220-1929035-ai-1475187-master-0`. The formal 162/162 and 93/93 suites,
static 132/132 checks, private-model-view gate, formal GPU preflight, and GPU
assignment all passed. All eight ranks completed loading the 1,026 model
weights. The run then stopped during warmup `_build_document_cache`: the
global producer post-hook raised
`CompactRebindError: cached post-hook used an unwrapped request`.

This happened before the first scientific cell. Consequently v23 produced no
scientific allocator endpoint, no real-binding aggregate, no terminal closure,
and no `COMPLETE`. The exclusive failure ledger records exit code 2 and
`science_accepted=false`; its SHA-256 is
`cb925284e53eb5c3d561a55b1f598bb3f5c48cd636e1417281d4527a2a53d94d`.
V23 is therefore a **pre-science producer-instrumentation failure**, not an
infrastructure/GPU-allocation failure, scientific negative result, or positive
paper evidence. It does not change the accepted manuscript claims. A v24
successor is applying the minimum scope correction so the post-hook acts only
on requests registered by the compact-rebind producer instrumentation; v24 is
recorded below and is not evidence.

### Live-binding v24 persistent-scope fix: post-science coverage-gate HOLD

V24 is the non-overwriting persistent-scope successor at
`evidence/r40_independent_live_binding_v24_persistent_scope_fix`. Its frozen
archive
`r40-independent-live-binding-v24-persistent-scope-fix-20260831a.tar.gz`
has SHA-256
`5c970b56d8795c9b11d24b4a62c97d4d43f4052945e887a4706bf20aa2b89250`;
the source-ledger file SHA-256 is
`6e04f24d2dbaf70040f3312fd35f005b2f950045fb24efa1383c6ef6cb1aeda4`.
The exact v22 binding verifier and the intended compact producer operations
remained unchanged. The controlled correction kept a persistent-document build
scope so rank-lifetime hooks could bypass exactly one prefill cache without
treating it as a wrapped request.

It ran as QS Job `256220` / Trial `1936087` on Pod
`qs-256220-1936087-ai-1482497-master-0`. The clean stage contained 37
directories and 144 files with zero AppleDouble paths. Outer 162/162, package
94/94, static 132/132, private-model-view, formal GPU preflight, CUDA smoke, and
GPU-assignment gates passed. All eight ranks loaded all 1,026 weights and
generated real phase artifacts, thereby definitively crossing v23's warmup
failure. Each rank completed N=1 and N=8 across all four arms and reached at
least 11/12 primary calls. Ranks 0 and 1 completed all 12 calls and committed
shards of 62,243,133 and 62,235,716 bytes with SHA-256
`0d0d9acc7a2bc7238a368ebbf40dbe12a799f5ca2c277e9a0ae60d5f5bdd8302`
and `46b62d6632d25e59714afa4fe5b28de12959b001188dfa25375f49a85d93bcad`.
Rank 7 also reached 12/12 in temporary capture; ranks 2--6 were at 11/12 when
the coordinated failure stopped them.

After rank-shard commit, ranks 0 and 1 each emitted
`compact rebind producer coverage drift`. The deterministic incomplete
predicate concerned exceptional forwards: fixed fault mutants intentionally
raise inside the backbone after the pre-hook, but the ordinary post-hook was
not `always_call`. Those started-and-aborted cached calls were therefore counted
as starts but not as postprocessed completions. The correct closure partitions
cached starts into successful postprocesses versus expected exceptional aborts
while retaining the 30-state success-path rebind gate. The builder's borrowed
construction count was not a v24 defect: the immutable builder creates every
request from a borrowed base, then materialized-policy requests execute a
second materialization step. Thus borrowed construction is correctly counted
over all wrapped requests. This is an audit-receipt count-closure defect, not a
GPU/model failure or an observed scientific endpoint violation.

The launcher stopped in phase `eight_rank_shards` at
`2026-09-01T13:13:36Z`. Only shards 0 and 1 committed; no stage 04 success,
detached raw-receipt acceptance, blind aggregate, terminal closure, or
`COMPLETE` exists. The exclusive failure ledger records exit code 2,
`science_accepted=false`, and status `HOLD_PENDING_FRESH_AUDIT_AND_H20`; its
SHA-256 is
`fb5cbb2057069e120e49e07a65f8806ad513d1cddc1d7c21239285f58d2f31ba`.
The formal log SHA-256 is
`0bc97cbd963c6f9063481b2123da1fca851a759e7bafab1839cfb7b7a1c28e42`.
V24 is therefore a **post-science producer-coverage terminal-gate failure** and
must not be pooled or cited. V25's completed outcome is recorded next. The
reviewed manuscript source and PDF remain unchanged.

### Live-binding v25 mixed-policy coverage fix: post-science coverage-gate HOLD

V25 is the non-overwriting abort-aware coverage successor at
`evidence/r40_independent_live_binding_v25_mixed_policy_coverage_fix`. The
approved frozen archive is
`r40-independent-live-binding-v25-mixed-policy-coverage-fix-20260901b.tar.gz`
with SHA-256
`fc9d02d21bd33669c6706a8c498dbd10d978d3b183ca603216db8c569114d031`;
the 46-line source-ledger file SHA-256 is
`3ca35856e6c4b24982e9b430f921cd309592ff103f24bcc01261aa0de9eeb0cc`.
The earlier `20260901a` candidate remains preserved as superseded and
`approved=false`. V25 kept the exact v22 real-binding verifier, added an
`always_call` post-hook that accounts for expected exceptional forwards without
recurrent rebind, and retained the successful-call 30-state gate.

The formal run reused QS Job `256220` / Trial `1936087` and Pod
`qs-256220-1936087-ai-1482497-master-0`, with a new scratch, stage, result,
one-shot marker, log, and failure ledger. The clean stage had 37 directories,
146 regular files, 183 nodes, and zero AppleDouble paths. Outer 162/162,
package 96/96, static 134/134, inner 162/162, private-model-view, formal GPU
preflight, CUDA smoke, and GPU-assignment gates passed. All eight ranks loaded
1,026 weights, completed N=1, N=8, and N=32 across all four arms, and committed
all eight primary shards. The shards total 498,207,018 bytes. Rank 0 and rank 1
have SHA-256
`cc96ce3921f209cd19e59ea86a6aaad0809969d1e389ee2ec32899c24eac52a5`
and `1e3a0bd65687e9bd7a2df237e23a5ada2b145af86dbedd583a9701385b8f7808`;
the complete eight-shard hash list remains in the remote failure tree.

After each immutable shard had committed and `base.main()` returned, all eight
ranks emitted `compact rebind producer coverage drift`. V25 had introduced an
incorrect equality between `borrowed_setup_calls_delegated` and
`borrowed_requests_returned`, where the latter counted only final borrowed-policy
groups. The immutable builder shows the actual construction semantics:
`_fresh_request` and `_reuse_request` call `_prepare_request_gdn_base(...,
policy=borrow)` for every request, and `_request_with_gdn_policy` then invokes a
second materialization helper for final materialized-policy requests. Therefore
borrowed delegation correctly equals all wrapped requests, while final borrowed
plus final materialized requests equals all wrapped requests. V25's new equality
was guaranteed false even though the producer executed correctly.

The launcher stopped in phase `eight_rank_shards` at
`2026-09-01T14:15:23Z`; stage 04, detached raw-receipt acceptance, blind
aggregate, terminal closure, and `COMPLETE` are absent. The exclusive failure
ledger records exit code 2 and `science_accepted=false`; its SHA-256 is
`b3825e07d128bffc69370b353926a75463d86693602b7c4de0ee57723f4b84ba`.
The formal log SHA-256 is
`e084bedea353e97beab4d47962329a9932b1178f7856a1bead056ee0db48ab55`.
V25 is a post-science producer-coverage terminal-gate failure, not a scientific
negative or positive evidence. Its eight shards must not be pooled or cited.
V26 restored the construction-step equality while preserving the v25
abort-aware successful/exceptional call closure; its completed nonterminal
outcome is recorded next. The reviewed TeX and PDF remain unchanged.

### Live-binding v26 construction-step receipt fix: post-science terminal-governance HOLD

V26 is frozen at
`evidence/r40_independent_live_binding_v26_construction_step_receipt_fix`.
Its deterministic archive
`r40-independent-live-binding-v26-construction-step-receipt-fix-20260901a.tar.gz`
has SHA-256
`902344af0d8e9bc31e407e2740dbe665ed29bf98879c73f0d8dae6f6d2263ad3`,
and its 48-row source-ledger file has SHA-256
`205ac90fdaa4ea2107168861923ba14e4e94db290b9b9e672ce73bfedde3c333`.
Local gates passed targeted 10/10, non-stage 82/82, full frozen 97/97 with zero
skips, static 134/134, and an independent byte-identical archive rebuild.

The formal run reused QS Job `256220` / Trial `1936087` and Pod
`qs-256220-1936087-ai-1482497-master-0`, with fresh scratch, stage, result,
marker, log, and failure-ledger paths. The clean stage contained 37 directories
and 148 regular files, with zero AppleDouble paths. Outer 162/162, package
97/97, static 134/134, inner 162/162, private-model-view, formal GPU preflight,
CUDA smoke, and GPU-assignment gates passed. All eight ranks loaded 1,026
weights and completed every N=1/N=8/N=32 arm. All eight primary shards were
atomically committed; stage `04_eight_rank_shards_ok`, detached raw-receipt
stage 05, and blind aggregate stage 06 all completed. The corrected producer
receipt therefore crossed the v25 failure. The nonterminal
`primary/forkaudit-summary.json` reports `passed=true`,
`scientific_run_valid=true`, `hypothesis_passed=true`,
`factorial_four_cell_exact=true`, and `oracle_all_ranks_passed=true`, but it is
not admissible without the later terminal gates.

The terminal code-snapshot audit then failed with
`ForkAudit code snapshot rejected: writable code entry present: __pycache__`.
The immutable primary launcher invokes its model-load lease through
`$PYTHON -I -c`. The R39 transparent proxy forwarded those arguments to the
real interpreter; Python isolated mode ignored the environment-only bytecode
control. Running as root, that process created one mode-755 `__pycache__`
directory and 14 writable `.pyc` files under the otherwise mode-555 immutable
primary code root. Their timestamps precede rank science and match the imported
lease dependencies. This is a deterministic bytecode-isolation/terminal-
governance defect, not a scientific negative result.

The run reached `06_blind_aggregate_ok` at `2026-09-01T15:05:46Z` and stopped
at `2026-09-01T15:06:19Z`. No formal-binding aggregate, R40 live-binding
aggregate, terminal closure, terminal tree, or top-level `COMPLETE` exists.
The exclusive exit-1 ledger records `science_accepted=false` and has SHA-256
`773538aaeadcaad8ef5e1484803bffe0593a852889fd55b05eb5a368daad721b`;
the formal log SHA-256 is
`0c8cb953fc11c5520260a75c31aa000efc502b277e9721772adfad2066b0ecb2`.
The nonterminal primary summary SHA-256 is
`d9c28cba474d11ac376d373821cd069881862022f205d718b16d753317ce5980`.
V26's shards and summary must not be pooled or cited. V27 is limited to a
source-ledger-bound real-Python wrapper that supplies command-line `-B` even
for transparent isolated invocations, plus a focused no-bytecode regression;
it must rerun under fresh non-overwriting identities. The reviewed TeX and PDF
remain unchanged.

### Live-binding v27 no-bytecode Python fix: post-science finalizer-path HOLD

V27 is frozen at
`evidence/r40_independent_live_binding_v27_no_bytecode_python_fix`. Its
deterministic archive
`r40-independent-live-binding-v27-no-bytecode-python-fix-20260901a.tar.gz`
has SHA-256
`241c7c80cf24c7bdd5d40c774fec6cd56bb79e7dd3013cc6f8781c4371ad1c73`,
and its source-ledger file has SHA-256
`f204d49c5c238ff23b6dced9ca2fbe72d631ebea6dd368af2a8a3ba9f00ae534`.
Targeted 10/10, packaging 13/13, non-stage 83/83, full frozen 98/98 with
zero skips, static 135/135, payload verification, and a byte-identical archive
rebuild passed.

The formal run reused QS Job `256220` / Trial `1936087` and Pod
`qs-256220-1936087-ai-1482497-master-0` under fresh scratch, stage, result,
marker, log, and failure-ledger identities. The wrapper supplied command-line
`-B` to both transparent isolated lease execution (`-B -I -c`) and routed rank
execution (`-B -I -B ...`). The immutable primary source tree contained zero
`__pycache__`, `.pyc`, or `.pyo` nodes during the lease, after the lease, and
at the frozen terminal code-integrity gate. V27 therefore crossed the exact
v26 failure point.

Detached 162/162, primary 13/13, private-model-view, formal GPU preflight,
CUDA smoke, and GPU assignment passed. All eight ranks loaded 1,026 weights,
completed the frozen science, and atomically committed all eight shards.
Stages `04_eight_rank_shards_ok`, `05_detached_raw_receipts_ok`, and
`06_blind_aggregate_ok` passed; primary reached `99_done`. The R39
formal-binding aggregate, terminal ledger, and empty `formal-binding/COMPLETE`
were also published, and `sha256sum -c formal-binding/terminal-files.sha256`
passes. These nonterminal products remain inadmissible without the later R40
aggregate and top-level closure.

The R40 finalizer then failed with
`RuntimeError: phase artifact missing on finalizer reread`. Each real-binding
receipt retained the producer's temporary relative path
`primary/raw/.forkaudit-rank-<rank>-<nonce>/rank-<rank>/...`. The immutable
producer atomically publishes that temporary tree to the stable path
`primary/raw/rank-<rank>/...` and removes the temporary name before the later
finalizer runs. A read-only postmortem found all 24 stable phase artifacts;
all 24 have exactly the receipt-recorded byte count and SHA-256. The defect is
therefore a stale pre-publication pathname in the new live-binding receipt,
not missing scientific data or a negative scientific result.

V27 stopped at `2026-09-02T00:00:43+08:00`. The exit-1 failure ledger records
`science_accepted=false` and has SHA-256
`94cfd5a33e6b776e2446fd919567a469d2fbe40f99a64eb80d39191d6b1d0e79`;
the formal log SHA-256 is
`29fe27bd8bb7bf951970dd3482caf069fd9e6fee4524898dcbfce85a7cae02d6`.
The nonterminal primary summary SHA-256 is
`a39545417aea10567947bca1592092910e99698d5e3edadf890a0b44be6bbedb`,
and the nonterminal formal-binding aggregate SHA-256 is
`b8c4a7af959cc254438952f6ea2d0757a95847c6343d491e8902b0acc7166f98`.
No R40 aggregate, terminal closure, terminal tree, or root `COMPLETE` exists;
none of V27's results may be cited or pooled. V28 is limited to rebinding each
already-validated phase receipt from its ephemeral producer path to the exact
stable publication path, plus a lifecycle regression covering temporary-tree
publication followed by finalizer reread. The reviewed TeX and PDF remain
unchanged.

### Live-binding v28 published-phase path fix: post-science closure-model HOLD

V28 is frozen at
`evidence/r40_independent_live_binding_v28_published_phase_path_fix`. Its
deterministic archive
`r40-independent-live-binding-v28-published-phase-path-fix-20260902a.tar.gz`
has SHA-256
`23fcdfc329c16308b15be748c2eda754223c55b273c850cc417502e44ec74393`;
the source-ledger file SHA-256 is
`301427f74f583fa8f2c025482a524ab4ca2656ad060307508e8397559af93172`,
the preregistration SHA-256 is
`8176fe142f94200bbc5696d075bbd10d599b04fc89d1240eb57a8312c0361a5d`,
and the generated launcher SHA-256 is
`3ce98dcfcb237841235951b1106f23aeeb38bcd9e317af9c55fdb0cf36228837`.

The formal run reused QS Job `256220` / Trial `1936087` and Pod
`qs-256220-1936087-ai-1482497-master-0`, with result root
`/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/r40-v28-published-phase-path-fix-20260902a`.
V28 crossed the v27 publication-path failure. All eight primary shards and the
primary aggregate completed; `primary/forkaudit-summary.json` reports
`valid_positive=true`, and primary reached `99_done`. Formal binding published
`formal-aggregate.json`, its terminal ledger, and an empty
`formal-binding/COMPLETE`. The R40 CUDA smoke and live-binding aggregate also
published before terminal closure began.

The subsequent `r40_tree_closure.py expected-paths` command failed with
`formal result exact expected path whitelist drift`. The immutable primary
launcher deliberately runs `-m py_compile` over its 31 frozen Python sources
with `PYTHONPYCACHEPREFIX=primary/pycache`. Command-line `-B` prevents automatic
import-cache writes but does not suppress this explicit compilation. The
result therefore contains exactly 31 `.cpython-311.pyc` files under the
result-sink `primary/pycache` and 13 required parent directories. These nodes
are expected non-scientific run products, are outside the immutable source
tree, and are distinct from v26's forbidden adjacent source bytecode. V28's
closure model included only the `primary/pycache` directory and omitted those
44 descendants. This is a deterministic terminal expected-path-model defect,
not a scientific negative result and not contamination of the science.

Available nonterminal hashes are:

- `primary/forkaudit-summary.json`:
  `5e494a8aa50647a5d97dd906eb3d406b43130b3ff512cda4195d541c953d7ac6`;
- `primary/scientific-artifacts.sha256`:
  `7f82013a0cb0e73146124abf019a84e915c25429350a5b11e9244a3c304acc79`;
- `formal-binding/formal-aggregate.json`:
  `7b770460ca2467a7147a2f944a5ab3259755ac36490b57fd13e4460894465e4b`;
- `formal-binding/terminal-files.sha256`:
  `ee3a92c1fef80458a1d0c200881e8273d182280ea27eba4ce69a418b045f248a`;
- `r40-formal/cuda-smoke.json`:
  `2ac8aa3c6b8324d10ebd01e7bded1631e11d7e1aeb058899204969ffb666a4e9`;
- `r40-formal/aggregate.json`:
  `f10e937a9711fc9e25b6160574a52dead43697a57a6cdb341cf56f965517ce27`.

The 38,078-byte formal log has SHA-256
`1c88bd832033079475c161441c6de90094f2ecf4bf0d50dabc0044629beb511e`.
The exclusive 139-byte exit-2 failure ledger records
`science_accepted=false` and has SHA-256
`ea993d14a7f118a8f08bc0127ffed0e3c4c06824f0ae704ce5512f3d34510807`.
No terminal closure, terminal tree, or root `COMPLETE` exists. V28 and every
intermediate product above remain inadmissible and must not be cited or pooled.
A fresh non-overwriting successor must bind the exact 31-file/13-directory
result-sink cache topology and pass its own full formal run and post-run audit.
The reviewed TeX/PDF, paper state, experiment registry, and claim map remain
unchanged.

### Held-out method v10: HOLD pending fresh audit and binding

- Archive SHA-256:
  `df3a4800e7c50b31168cf7033dec02e039f98c46a0fb92b59c7c1ec120e79ac8`
- Source-ledger JSON SHA-256:
  `8c6fd03fde33f6eb7274af5bb75b98c0f436d943c7ff0b1d2a5c759925ca3b17`
- `METHOD_FROZEN.json` SHA-256:
  `d6a075269dd33b124b502b8c73614a2778bdbc1005f53cf10bb30b232a18b62a`

V10 is a **method freeze only**. It contains no operator binding, designer fault
identities, frozen formal configuration, GPU execution, or scientific outcome.
Its packaged discovery surface is 18 CPU-only tests. Relative to the diagnostic
v9 HOLD, v10 binds durable consumption authority to canonical root objects,
uses child-observed executable identity, re-derives typed actual specs, closes
the formal process-creation surface, and requires exact terminal names. These
are design claims pending a new independent audit, not accepted evidence.

No result from live-binding v17--v28, or held-out v9/v10, should be cited as
positive evidence. V21 is a scientific-execution audit-gate negative, v22 is
the descriptor diagnostic, v23 is a pre-science warmup instrumentation failure,
v24/v25 are post-science producer-coverage receipt failures, and v26--v28 are
successive post-science terminal-governance/closure failures despite their
increasingly complete nonterminal products. None establishes that the
manuscript's existing main result is false.

## 9. Supporting cost and related-work context

### Local replay/storage accounting

`evidence/r40_ci_cost_accounting_v1/aggregate.json` has SHA-256
`8add66006445473050d12a7feaa484c53f7a7d711cc5eef983b27c2ff5906db8`.
On one Apple Mac16,8 CPU host, three fixed-order warm-cache profiles report:

- core replay median 100.146 s;
- extended supporting replay median 105.717 s;
- paired increment median 5.598 s;
- manifest-only core distribution 630 files / 850.95 logical MiB;
- 536 raw trace files / 888,785,811 logical bytes.

These are local CPU replay and logical-storage measurements only. They are not
H20 capture cost, GPU perturbation, cold-start latency, online inference cost,
or an uninstrumented-baseline delta.

### Mac M4 motivation control

`evidence/mac_m4_motivation/summary.json` has SHA-256
`d84511c5aa38124c5816f3365c72281ab8abd348ccd11dfbbc826fef322a1f2e`.
Across five fresh MLX processes and six teacher-forced queries per process,
Q8/Q4 agree 100% with the BF16 split path, while the BF16 split path agrees only
71.89% with vanilla dense. This motivates an independent oracle; it is not
primary evidence and supports no cross-hardware or production-speed claim.

| Mac/MLX row | Corpus Store (MiB) | Median online (ms) | Speedup vs. dense | Top-1 vs. BF16 split | Top-1 vs. dense |
|---|---:|---:|---:|---:|---:|
| Vanilla dense prompt | n/a | 194.702583 | 1.000000 | n/a | 1.000000 |
| CoMem BF16 split | 1.998047 | 168.877104 | 1.152925 | 1.000000 | 0.718935 |
| CoMem Q8 split | 1.061462 | 169.171271 | 1.150920 | 1.000000 | 0.718935 |
| CoMem Q4 split | 0.561951 | 169.247104 | 1.150404 | 1.000000 | 0.718935 |

The five raw process outputs are
`results/formal-20260811-ac.LIgSzd/frozen-{1..5}.json`, with SHA-256 values in
order:

1. `8e5c273738b570adcbac9fd9e7e80a8c6e10b089e7f1ab56195cacc1eb8e9b26`;
2. `30483bec46e49b377ec17af9761621428b32c7dc29cbc1451a2251d3e06aa285`;
3. `24dc4303778652cd6837c434caca637873e027026f2da91a0dada5b8726248b7`;
4. `1a1be88e1a7e71184826b10b0c6dfac670170a15d0e8a8af9cb7474fa6c98fa9`;
5. `1f331f755a22ee3a508e594f16008f222e3ee94f5bca29c74f664c93df3b3fd7`.

### Same-slice serving and HYPIC context

`evidence/related_work_same_protocol/serving_panel_summary.json` has SHA-256
`a9cc7c78d6e2222f463b0ca86b626bac0a323f02b00ad031105d7832a0a6b245`.
It reports within-framework cache-off/on controls for vLLM 0.26 and SGLang
0.5.17 plus an official-code HYPIC/SGLang 0.5.14 TP=1 adaptation. HYPIC retained
state is separately bound by
`hypic-retained-state-r34-trial1892234/acceptance.json` (SHA-256
`15dbee59e8f422a944cdcc2bd67c276b359b34327230569bd14f9afdb787cbec`):
16/16 externally replayed receipt cells pass, with median Prefix Cache Store
139.53125 MiB and transition-rope-recompute Store 324.0918 MiB.

Timing is interpreted only within each runtime block. These rows are not pooled
with in-process CoMem timings and do not form a cross-framework leaderboard.

Canonical same-slice summary anchors are:

- HYPIC timing/quality:
  `evidence/related_work_same_protocol/hypic-same-protocol-20260821c/summary.json`,
  SHA-256
  `0543b491e70ddfaf6d40651b1f1babec652bd9c8f2a5f9d0cca7305cc2cb1b3d`;
- vLLM prefix-cache control:
  `evidence/related_work_same_protocol/related-vllm-prefix-bootstrap-f-20260820a/summary.json`,
  SHA-256
  `625282cff4a7a371c2c5f4c55f4a4173b9a304cbcb0717365bbc254e660ed137`;
- SGLang/RadixAttention control:
  `evidence/related_work_same_protocol/related-sglang-radix-node-20260821c/summary.json`,
  SHA-256
  `509b0c6a148313eac1ab7f5d6011bff80bd0e7e8c91fc054b34a20f54a17070d`.

### Hydragen, Palu, and Marconi transfers

These packages make narrower same-checkpoint or same-trace comparisons; none is
an end-to-end CoMem-versus-related-work serving leaderboard.

- **Hydragen operator transfer.**  The formal summary is
  `evidence/round24_related_work_transfer/hydragen_formal/formal-hydragen-20260821g/hydragen-transfer-summary.json`
  (SHA-256
  `0d3c3d4f8d3bceb4e40886d9b98f7e7f1af69f96ed75bcae68363724a4ed4651`),
  with CPU replay report SHA-256
  `a2ba263e552bed004f4b354e9a7dd25f73ce9f0e28f2be6503c4b3a30484fc73`.
  On the captured Qwen3.5 layer-3 post-RoPE operator, N=8 and N=32 pass the
  preregistered numerical gates.  Replicated dense KV is 7.59x and 25.80x the
  Hydragen shared-prefix KV footprint, respectively.  This is not an
  end-to-end Hydragen integration, LongBench result, or serving-speed claim.
- **Palu projection transfer.**  The formal summary is
  `evidence/round24_related_work_transfer/palu_whiten_formal/palu-whiten-transfer-summary.json`
  (SHA-256
  `2056a2a8aed0d3f51b1d0256953f864edfa413550437c988361505ba2397cc6e`),
  with CPU replay report SHA-256
  `c6a8f95f1fff76c3b6a32a0f06cadd20cf34f352193f96db3f9d6f9e0460120e`.
  On one Qwen3.5 layer-3 K/V projection, activation whitening improves held-out
  projection error over plain SVD at ranks 64/128/192; the corresponding
  logical dense-over-latent KV ratios are 4.0x/2.0x/1.33x.  This is not an
  implemented Palu attention kernel, all-layer calibration, or end-to-end
  quality/latency result.
- **Marconi policy trace.**  The formal summary is
  `evidence/related_work_same_protocol/qcomem-marconi-formal-a/summary.json`
  (SHA-256
  `afb9c6d41973ab3259fc7e50441a6822cba5290a046ed7c237490d67dfe643ec`),
  binding trace SHA-256
  `536c676dec6cad496ec2825ef0f38e6da9e0dc64f543f0e4615db78e3eb9157f`.
  On the frozen 128-request trace and Marconi's native Attention--Mamba2
  geometry, the 5/10/20 GB policy rows report token-hit rates for
  vLLM+/SGLang+/Marconi.  Only those trace-level hit rates are portable;
  simulator wall time and predicted FLOPs are not model-serving measurements.

Prompt Cache, PagedAttention, ChunkAttention, and Preble remain
native-protocol published context rather than same-condition reproductions.

## 10. Version chronology and non-admissible history

This chronology retains failed versions for audit history and debugging while
distinguishing the accepted R39 and v11 anchors.  Failed artifacts must not be
silently pooled with accepted evidence:

- compiled-dispatch v7: pre-run HOLD; bytecode contamination in the archive;
  no formal result;
- compiled-dispatch v8: pre-run HOLD; packaging repair only;
- compiled-dispatch v9: H20 execution reached an observer that rejected a
  legitimate default-stream handle `0`; no admissible receipt;
- compiled-dispatch v10: all 8 ranks/192 cells ran, but terminal code audit
  found `__pycache__`; `COMPLETE` was not written;
- compiled-dispatch v11: the only accepted formal result in the v7--v11
  primary-factorial repair sequence; an earlier bounded R39 H20 package remains
  a separate accepted 120-attention-call result for the exact R29 execution;
- live-binding v16: pre-GPU Linux/AppleDouble staging failure;
- live-binding v17: fresh-audit HOLD documented above; no GPU result;
- live-binding v18: deterministic terminal expected-path omission found before
  payload staging; its later generic-sleep Pod was stopped with no v18 stage,
  CUDA, or scientific result;
- live-binding v19: controlled closure repair and fresh-audit GO, followed by a
  formal preflight failure (85/87 tests passed); its failure ledger records
  `science_accepted=false`, and no CUDA or rank science occurred;
- live-binding v20: governance-only preflight-environment repair with all ten
  scientific payload files byte-identical to v19; the formal path passed
  detached focused 162/162 and stages 00--04, then hung in its atomic-signal
  self-test because the `nohup` background inherited ignored signal
  dispositions. The full process tree was terminated with failure-ledger exit
  143 and `science_accepted=false`; formal static, CUDA, rank science, and
  `COMPLETE` never occurred;
- live-binding v21: minimal non-overwriting signal-governance repair formally
  launched as Job `256090` / Trial `1920306`; formal 87/87 and 162/static/GPU
  preflight passed and all eight ranks entered science, then every rank rejected
  the first generation callback of the first N=8 ownership witness at
  `r40_real_binding.py:246`. Failure-ledger exit 2 records
  `science_accepted=false`; no `COMPLETE`, terminal closure, or aggregate;
- live-binding v22: diagnostic-only successor formally launched as Job
  `256220` / Trial `1920822`; formal 162/162, 87/87, static 132/132, and GPU
  preflight passed and all eight ranks loaded the model. Seven flushed rank
  logs identically located `(0,'conv',0)` and a stride change from
  `[33546240,1,8192]` to `[32768,4,1]`; rank 7 exited in sibling-failure
  coordination before flushing that line. Failure-ledger exit 2 records
  `science_accepted=false`. This is diagnostic localization for a v23 producer
  fix, not positive evidence or a refutation of the accepted paper result;
- live-binding v23: compact-rebind producer-fix successor formally launched as
  Job `256220` / Trial `1929035` on Pod
  `qs-256220-1929035-ai-1475187-master-0`; 162/162, 93/93, static 132/132,
  private-model-view, GPU-preflight, and GPU-assignment gates passed, and all
  eight ranks loaded 1,026 weights. Warmup `_build_document_cache` then reached
  the global post-hook with a request outside the compact-rebind wrapper and
  raised `CompactRebindError: cached post-hook used an unwrapped request`.
  Failure-ledger exit 2 records `science_accepted=false`; no first scientific
  cell, allocator endpoint, real-binding aggregate, terminal closure, or
  `COMPLETE` occurred. This is a pre-science producer-instrumentation HOLD, not
  scientific evidence;
- live-binding v24: persistent-scope successor formally launched as Job
  `256220` / Trial `1936087` on Pod
  `qs-256220-1936087-ai-1482497-master-0`; outer 162/162, package 94/94,
  static 132/132, private-model-view, CUDA/GPU preflight, and all eight
  1,026-weight loads passed. All ranks crossed the old warmup fault and reached
  at least 11/12 factorial calls; ranks 0/1 completed 12/12 and committed
  shards. Their post-main producer receipt then failed because expected
  exceptional fault calls were counted at pre-hook entry but omitted from the
  completion/abort closure.
  Failure-ledger exit 2 records
  `science_accepted=false`; only two shards committed and no stage 04,
  aggregate, terminal closure, or `COMPLETE` exists. This is a post-science
  producer-coverage terminal-gate HOLD, not a scientific negative or positive
  evidence;
- live-binding v25: abort-aware successor formally launched on the same Job
  `256220` / Trial `1936087` with new non-overwriting paths; outer 162/162,
  package 96/96, static 134/134, inner 162/162, private-model-view, CUDA/GPU
  preflight, and all eight 1,026-weight loads passed. All ranks completed 12/12
  factorial calls and committed all eight shards, then uniformly failed the
  post-main producer receipt. V25 had incorrectly equated mandatory borrowed
  construction calls with only final borrowed-policy requests. The immutable
  builder instead runs borrowed construction for every request and adds a
  second materialization step for materialized policy. Failure-ledger exit 2
  records `science_accepted=false`; no stage 04, aggregate, terminal closure,
  or `COMPLETE` exists. This is another post-science coverage-gate HOLD, not
  scientific evidence;
- live-binding v26: construction-step successor formally launched on the same
  Job `256220` / Trial `1936087` with fresh non-overwriting paths; outer
  162/162, package 97/97, static 134/134, inner 162/162, private-model-view,
  CUDA/GPU preflight, all eight 1,026-weight loads, 12/12 factorial calls per
  rank, all eight shards, detached receipts, and blind primary aggregate
  completed. The corrected producer gate passed. Terminal code-snapshot audit
  then rejected one writable `__pycache__` and 14 `.pyc` files created by the
  immutable model-load lease's `python -I -c` path. Exit-1 ledger records
  `science_accepted=false`; formal-binding/R40 aggregate, terminal closure, and
  `COMPLETE` are absent. This is a post-science terminal-governance HOLD, and
  neither its shards nor its primary summary is evidence. V27 adds only a
  source-bound command-line `-B` wrapper for the transparent interpreter path;
- live-binding v27: no-bytecode successor completed all eight shards, primary
  `99_done`, and formal-binding `COMPLETE`, but its R40 finalizer retained
  cleaned temporary publication paths in durable phase receipts. Exit-1 records
  `science_accepted=false`; no R40 aggregate, terminal closure, or root
  `COMPLETE` exists;
- live-binding v28: published-path successor completed primary with
  `valid_positive=true`, formal-binding `COMPLETE`, and the R40 aggregate, then
  failed exact expected-path closure because its model omitted the immutable
  launcher's 31 intentional result-sink `.pyc` files and 13 parent directories.
  Exit-2 records `science_accepted=false`; no terminal closure/tree or root
  `COMPLETE` exists. V28 is not evidence;
- held-out v1--v8: superseded method-only/HOLD artifacts with no admissible
  outcome;
- held-out v9: method freeze only and independently HOLD; no operator-bound
  fault/configuration or GPU result;
- old HYPIC debug, recovery, and invalid-formal directories: diagnostic only;
  only the canonical HYPIC timing/quality and retained-State acceptance packages
  listed above are evidence anchors.

Failure history is kept here so that a later revision cannot accidentally cite
an obsolete package merely because its files exist.  These infrastructure
failures need not be foregrounded in the manuscript; scientific limitations
must still be stated accurately.

## 11. Claim discipline for the next manuscript revision

The next paper copy should obey all of the following:

1. Lead with the retained-Store result: **140.34 to 9.74 MiB/document,
   93.06% less, mean-F1 delta -0.022**; retain Q8 as the same-F1 point.
2. Do not call Store process memory, allocator capacity, or serving capacity.
3. Do not claim runtime independence. State that feasibility and all formal
   predicates were demonstrated on declared fixed configurations.
4. Integrate compiled-dispatch v11 only at its audited per-call Triton
   artifact/configuration boundary; do not claim driver/device attestation or
   compiled GDN identity.
5. Keep trusted capture explicit. The census and repeats narrow expected-set and
   single-run risks, but correct live binding and capture honesty remain trusted
   unless a later live-binding package independently passes. V24/v25 failed
   producer gates; v26 failed terminal code-snapshot governance; v27 failed R40
   finalization; and v28 reached both formal-binding `COMPLETE` and an R40
   aggregate but failed exact terminal path closure. None closes this boundary.
6. Describe constructed faults as fixed-case sensitivity/localization evidence,
   never as a detection rate or population estimate.
7. Keep Mac, HYPIC, vLLM, and SGLang context unpooled and scope-separated.
8. Do not mention live-binding v17--v28 or held-out v9/v10 as positive results.
   Record v21 only as a scoped deterministic audit-gate negative
   during formal science, v22 only as its descriptor diagnostic, and v23 only
   as a pre-science warmup instrumentation failure. None by itself refutes the
   current main result. Record v24/v25 only as post-science producer-coverage
   receipt failures and v26/v27/v28 only as post-science terminal-governance,
   finalizer-path, and closure-model failures, respectively. Their shards and
   nonterminal products are inadmissible, and a successor remains non-evidence
   until its own formal terminal PASS and post-run audit.
9. When the manuscript is edited, update abstract, introduction, contract,
   results, limitations, conclusion, target table, cohort table, captions, and
   appendix claim maps together; then reread and render the entire PDF.

## 12. Immediate open work

- Preserve the terminated v21 tree and exit-2 ledger; v22 Job `256220` / Trial
  `1920822`, its seven identical flushed descriptor lines, rank 7
  sibling-failure exit, and exit-2 diagnostic ledger; and v23 Job `256220` /
  Trial `1929035`, its warmup post-hook failure, and exit-2 ledger with SHA-256
  `cb925284e53eb5c3d561a55b1f598bb3f5c48cd636e1417281d4527a2a53d94d`.
  Preserve v24 Job `256220` / Trial `1936087`, its two committed shards,
  `eight_rank_shards` failure, and exit-2 ledger SHA-256
  `fb5cbb2057069e120e49e07a65f8806ad513d1cddc1d7c21239285f58d2f31ba`.
  Preserve v25's eight committed shards, `eight_rank_shards` failure, and
  exit-2 ledger SHA-256
  `b3825e07d128bffc69370b353926a75463d86693602b7c4de0ee57723f4b84ba`.
  Preserve v26's eight committed shards, stages 04--06, nonterminal primary
  summary, and exit-1 ledger SHA-256
  `773538aaeadcaad8ef5e1484803bffe0593a852889fd55b05eb5a368daad721b`.
  Preserve v27's stable artifacts and exit-1 ledger SHA-256
  `94cfd5a33e6b776e2446fd919567a469d2fbe40f99a64eb80d39191d6b1d0e79`.
  Preserve v28's primary/formal/R40 nonterminal products and exit-2 ledger
  SHA-256
  `ea993d14a7f118a8f08bc0127ffed0e3c4c06824f0ae704ce5512f3d34510807`.
  A successor must preserve the successful v28 scientific path while binding
  the exact intentional result-sink pycache topology in terminal closure.
  Retain the exact v22 verifier and successful scientific producer path. Treat v21's secondary allocator nonrecovery as
  traceback/strong-reference retention unless independently shown causal.
  Preserve v20's exit-143 ledger as separate pre-science history.
- Run a fresh independent audit of held-out method v10. Only after an audit GO
  may an independent operator bind the snapshot, fault set, configuration,
  durable authority root, and formal execution; no v9 identity or status may be
  inherited.
- Preserve `main_r40_submission_candidate.tex` and its reviewed PDF identity.
  Apply review-driven writing changes in a successor only, and integrate a
  later live-binding successor or v10 only after its formal evidence and
  post-run audits pass. Live-binding v17--v28 cannot be promoted to positive
  evidence.
- The registry and claim map include accepted v11 and historical HOLD entries.
  Do not add live-binding v17--v28 as positive evidence. Add a later successor
  or v10 only after its own formal PASS and post-run audit; the v21 negative,
  v22 localization, v23 pre-science failure, v24/v25 producer-gate failures,
  and v26--v28 terminal failures do not authorize a manuscript claim change.
- If local replay closure for v11 is required, mirror the whole remote result
  tree except the model-weight files.  The current compact mirror is sufficient
  as a top-level index but not for no-network replay of all 949 terminal-ledger
  products.

## 13. Final-integrity snapshot

Unless an absolute path is shown, paths in this summary are relative to
`paper_autonomous_multifork_iteration/`.  At the completion of this summary:

- `evidence/experiment_registry.json` SHA-256:
  `b6d86797f26ff8e08c6a69ff2b622ec4a21212deddba0ab383cd993499d9345a`;
- `evidence/claim_evidence_map.tsv` SHA-256:
  `f4dba44f8c664d7a46dbdfef0af2236c0f049f32fe93a4fe6f9fd18aa8fb0482`.

The registry parses as JSON, every claim-map row has seven tab-separated
fields, and `scripts/validate_r40_compiled_dispatch_integration.py` passes on
the compact v11 mirror while reporting the intended nonclaims:
`full_remote_raw_audit_reexecuted_locally=false` and
`standalone_postrun_auditor_identity_receipt_present=false`.

## 2026-09-02 live-binding V29 fresh admissible PASS

V29 is frozen at
`evidence/r40_independent_live_binding_v29_result_pycache_whitelist_fix`; its
archive SHA-256 is
`893202582f3cac7ef9f8b61fc2d5c574c7609c51aa811cf518c488a1f1efd297`
and source-ledger-file SHA-256 is
`4d0563a99997a6d2c0a76ee6694b195599fff0caaa1119d22b4e78ad3ad489b0`.
The run used QS Job `256220` / Trial `1936087`, Pod
`qs-256220-1936087-ai-1482497-master-0`, and run ID
`71391b1a7ce85c4dfa8beb18f3c2189a`.

All eight shards completed and the scientific outcome is `valid_positive`.
The N=32 materialized full-copy-to-shared final-memory reduction is
`54.531038401%` (display `54.5%`). The oracle observed maximum relative L2 is
`0.0017432502481433169`, within the fixed `0.005` threshold. Live-binding
closure contains 144 selected rows, 12,960 storage rows, 3,840 clone edges,
24 stable phase artifacts, 96 primary calls, and zero global primary
memory-hook events. Terminal closure accepted exactly 31 authorized pyc files
and 13 descendant directories; the final tree has 1,367 nodes. Primary reached
`99_done`; root and formal `COMPLETE` are empty; the failure ledger is absent.

The formal launcher terminal path succeeded. Post-run audit attempt 1 failed
only because the checker treated `0.005` as an observed maximum rather than the
threshold. Attempt 2 enforced `threshold == 0.005` and observed vector maximum
`<= threshold` and passed. Its exact local mirror is
`evidence/r40_independent_live_binding_v29_postrun_audit_mirror`.

Anchors are primary summary `d49f25ddef31d8a0afffeccba855b05123210b1b1ccdcdc364ebef56ae3e298c`,
scientific ledger `ffdd40f02d114ce2a50ddd042701ae4282177de87c3e32875b90bc598e66fd13`,
formal aggregate `feae2481a4cf9e6a45135896741b08a4529d9b264a63622e5e8004cfe766c1fb`,
formal ledger `d814ffa69d9bb1fcb502fa8704edb351606cf1ccba147bd1376caa1ee98f4a10`,
R40 aggregate `40e1b45d715a20222fff6d85344d8fbbd06dbeae6a7d0056462e5d90af53d4fa`,
CUDA smoke `2ac8aa3c6b8324d10ebd01e7bded1631e11d7e1aeb058899204969ffb666a4e9`,
terminal closure `7ba11f6a71e8558eabd82af742e7f4c901ba8ceb9ce9ccd6a3d15e3f9c9610bf`,
and terminal tree `6aadf2d4e066f0e78978c6e216be3ef1ad34f46959f74cba3be79dde91a1f72a`.
V26--V28 remain inadmissible; V29 alone is the fresh admissible result.
