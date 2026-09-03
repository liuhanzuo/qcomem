# R39 second-model / second-runtime ForkAudit transfer — official ModelScope D

Status: **non-overwriting ModelScope-D source and protocol frozen; no D GPU output has been produced**.

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

## Acquisition-only D change

The scientific cell, inputs, arms, thresholds, controls, rank count, and claim
boundary are exactly equal to A, B, and C after stripping the acquisition object.
A stopped before model execution because the official Hugging Face Xet/CAS
data path exhausted its connection retries.  B also stopped before model
execution because the mirror redirected mandatory LFS objects to a CDN whose
TLS handshake timed out.  C stopped before model or GPU execution after an
independent audit established that the official ModelScope content endpoint
answers a real nonzero `Range` request with HTTP 200 rather than a resumable
206 response.  All predecessor packages and partial directories remain
untouched.  D changes only the preregistered acquisition authority:

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
- every attempt for every file starts at byte zero in a newly created
  temporary, sends no `Range` header, and accepts only HTTP 200 with an exact
  full-file `Content-Length`;
- transport, short-body, size, or hash failure deletes that attempt and starts
  a distinct zero-origin attempt, up to the fixed 12-attempt limit; no append
  or partial resumption exists; and
- a fresh D model root is mandatory.  No A, B, or C partial is read or reused.
  The complete tree, source marker, and pre/terminal authorities are hashed,
  and every model file and directory is closed read-only.

## Local, non-GPU verification

```bash
cd /Users/liuhanzuo/MacLLM-Bench
python3 -m unittest \
  paper_autonomous_multifork_iteration/evidence/r39_second_model_transfer_modelscope_d/executed_source/test_r39_second_model_transfer.py
bash -n \
  paper_autonomous_multifork_iteration/evidence/r39_second_model_transfer_modelscope_d/executed_source/launch_r39_second_model_transfer_8gpu.sh \
  paper_autonomous_multifork_iteration/evidence/r39_second_model_transfer_modelscope_d/executed_source/launch_trial_1907355_modelscope_d.sh
```

The detached replay entry point is:

```bash
python3 paper_autonomous_multifork_iteration/evidence/r39_second_model_transfer_modelscope_d/executed_source/replay_r39_second_model_transfer.py \
  --package-root paper_autonomous_multifork_iteration/evidence/r39_second_model_transfer_modelscope_d \
  --run-root /path/to/completed/run
```

## Frozen remote stage and exact existing-Trial command

Stage the reviewed repository at this new path before execution:

`/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r39_second_model_transfer_20260826d`

Then the single authorized command inside the already-running Trial pod is:

```bash
bash /mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r39_second_model_transfer_20260826d/paper_autonomous_multifork_iteration/evidence/r39_second_model_transfer_modelscope_d/executed_source/launch_trial_1907355_modelscope_d.sh
```

The wrapper neither creates, stops, nor evicts QS resources.  If the pinned
snapshot is absent, it may resolve and download the public repository once
through the frozen official ModelScope policy above, hash every local file,
make the dedicated D snapshot tree read-only, and close it with a terminal
full-file reread.  It never prints or reads a credential into experiment
evidence.
