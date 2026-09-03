# R39 second-model / second-runtime ForkAudit transfer — official ModelScope C

Status: **non-overwriting ModelScope-C source and protocol frozen; no C GPU output has been produced**.

This is a bounded transfer experiment for the text path of
`Qwen/Qwen3.5-0.8B` at immutable Hugging Face revision
`2fc06364715b967f1860aea9cf38778875588b17`.  It uses Transformers 5.14.1
`DynamicCache`, not the primary vLLM paged-Q16 stack.  The preregistered cell
is eight independent H20 ranks (only if eight suitable devices are present),
`N={1,2}`, a 64-token document, two 8-token queries, split depth 7, and two
greedy semantic steps.

The two arms are independent lower-state materialization per request and one
persistent Q16 lower state followed by request-local forks.  Q16 is the
lossless 16-bit representation from the audited A4 `qcomem_torch` source:
the immutable BF16 boundary may be shared, while every mutable cache tensor is
cloned.  Full-vocabulary CPU-FP32 sidecars authorize exact cross-arm and
cross-N replay.  Prefix immutability and normalized storage-range disjointness
authorize the ownership claims.

The standard full-model reference is conditional.  It is authoritative only
when both preregistered validation controls pass: (1) manual one-shot layer
splitting agrees with the official wrapper, and (2) official full-model
`DynamicCache` document/query chunking agrees with official one-shot
recomputation.  A failed control is a valid negative, never an excuse to tune a
threshold or swap the reference after output.

`DynamicCache` has no paged partial tail, so that target is N/A.  Dispatch is
partial: Python adapter/mask/layer classes and source hashes are recorded, but
compiled CUDA/Triton binaries, autotune choices, and instruction traces are
not claimed.  This experiment does not authorize performance, capacity,
concurrency, scheduler, vision, paged-tail, compiled-dispatch, or production
portability claims.

## Acquisition-only C change

The scientific cell, inputs, arms, thresholds, controls, rank count, and claim
boundary are exactly equal to A and B after stripping the acquisition object.
A stopped before model execution because the official Hugging Face Xet/CAS
data path exhausted its connection retries.  B also stopped before model
execution because the mirror redirected mandatory LFS objects to a CDN whose
TLS handshake timed out.  Both failed runs and partial directories remain
untouched.  C changes only the preregistered acquisition authority:

- official Qwen ModelScope endpoint `https://modelscope.cn` with public,
  token-free access;
- immutable ModelScope Git commit
  `4d58a7b524cd33ed843d5125be8cd8f0a452d9bf`;
- canonical Hugging Face identity remains full commit
  `2fc06364715b967f1860aea9cf38778875588b17`;
- the 1,746,942,600-byte weight and 12,807,982-byte tokenizer have exact
  cross-source SHA-256 equality;
- the pinned 14-file tree is checked against the live official API before
  revision-pinned per-file HTTPS downloads begin;
- each file has an independent temporary, resumable Range transfer and exact
  size/SHA-256 gate; and
- a fresh C model root is mandatory.  No A or B partial is read or reused.
  The complete tree, source marker, and pre/terminal authorities are hashed,
  and every model file and directory is closed read-only.

## Local, non-GPU verification

```bash
cd /Users/liuhanzuo/MacLLM-Bench
python3 -m unittest \
  paper_autonomous_multifork_iteration/evidence/r39_second_model_transfer_modelscope_c/executed_source/test_r39_second_model_transfer.py
bash -n \
  paper_autonomous_multifork_iteration/evidence/r39_second_model_transfer_modelscope_c/executed_source/launch_r39_second_model_transfer_8gpu.sh \
  paper_autonomous_multifork_iteration/evidence/r39_second_model_transfer_modelscope_c/executed_source/launch_trial_1907355_modelscope_c.sh
```

The detached replay entry point is:

```bash
python3 paper_autonomous_multifork_iteration/evidence/r39_second_model_transfer_modelscope_c/executed_source/replay_r39_second_model_transfer.py \
  --package-root paper_autonomous_multifork_iteration/evidence/r39_second_model_transfer_modelscope_c \
  --run-root /path/to/completed/run
```

## Frozen remote stage and exact existing-Trial command

Stage the reviewed repository at this new path before execution:

`/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r39_second_model_transfer_20260826c`

Then the single authorized command inside the already-running Trial pod is:

```bash
bash /mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r39_second_model_transfer_20260826c/paper_autonomous_multifork_iteration/evidence/r39_second_model_transfer_modelscope_c/executed_source/launch_trial_1907355_modelscope_c.sh
```

The wrapper neither creates, stops, nor evicts QS resources.  If the pinned
snapshot is absent, it may resolve and download the public repository once
through the frozen official ModelScope policy above, hash every local file,
make the dedicated C snapshot tree read-only, and close it with a terminal
full-file reread.  It never prints or reads a credential into experiment
evidence.
