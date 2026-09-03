# R39 second-model / second-runtime ForkAudit transfer

Status: **source and protocol frozen; no GPU output has been produced**.

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

## Local, non-GPU verification

```bash
cd /Users/liuhanzuo/MacLLM-Bench
python3 -m unittest \
  paper_autonomous_multifork_iteration/evidence/r39_second_model_transfer/executed_source/test_r39_second_model_transfer.py
bash -n \
  paper_autonomous_multifork_iteration/evidence/r39_second_model_transfer/executed_source/launch_r39_second_model_transfer_8gpu.sh \
  paper_autonomous_multifork_iteration/evidence/r39_second_model_transfer/executed_source/launch_trial_1907358.sh
```

The detached replay entry point is:

```bash
python3 paper_autonomous_multifork_iteration/evidence/r39_second_model_transfer/executed_source/replay_r39_second_model_transfer.py \
  --package-root paper_autonomous_multifork_iteration/evidence/r39_second_model_transfer \
  --run-root /path/to/completed/run
```

## Frozen remote stage and exact existing-Trial command

Stage the reviewed repository at this new path before execution:

`/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r39_second_model_transfer_20260826a`

Then the single authorized QS command is:

```bash
qs exec 1907358 -- bash /mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r39_second_model_transfer_20260826a/paper_autonomous_multifork_iteration/evidence/r39_second_model_transfer/executed_source/launch_trial_1907358.sh
```

The wrapper neither creates, stops, nor evicts QS resources.  If the pinned
snapshot is absent, it may download the public 1.77-GB repository once with
`huggingface_hub.snapshot_download(token=False)`, hash every local file, make
the dedicated snapshot tree read-only, and close it with a terminal full-file
reread.  It never prints or reads a credential into experiment evidence.

