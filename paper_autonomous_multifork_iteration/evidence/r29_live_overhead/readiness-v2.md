# Round 29 live ForkAudit overhead readiness v2

Status: the second-attempt implementation is locally dependency-closed,
hash-consistent, and frozen, but it has not been uploaded or executed. An
independent remote preflight is still required before any GPU command.

The immutable scientific preregistration remains byte-identical at
`2114d2cd85bedc1eafa5d1398fd0afd0d57819c0360c3be3f9ec20f1b2878939`.
Its scientific question, Qwen3.5 revision, PG19/document/query hashes,
warmup/measured schedule, request slots, timed boundary, peak definition,
estimands, oracles, negative-result policy, resource request, and claim
boundary did not change.

## Retained first attempt

The first formal attempt used QS Job 249885 / Trial 1898483, staged package
`qcomem_r29_live_overhead_20260825a`, run directory
`r29-live-overhead-20260825a`, and GPU UUID
`GPU-2645e3a2-b265-9923-19c5-c3a1fc537ec0`. It reached stages `00-started`
and `01-preflight-passed`, then failed in the first warmup pair's common
4,033-token document prefill. It did not reach Q16 conversion, resident
request construction, either arm timer, or artifact creation. No formal
result, independent replay, terminal ledger, or audit artifact exists.

The formal stderr SHA is
`73d5e96c85829a7b9c13e5f1cb3c775577f89a2e0ff631b9f56824bd11f3d5cd`;
stdout is empty with SHA
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
The failure is retained as a pre-arm implementation failure, not a scientific
negative result or overhead observation. Full disclosure is frozen in
`pre-second-execution-amendment-v2.json`.

## Root-cause correction

The reused `_build_document_cache` helper expects its caller to establish
inference mode. The v1 pair builder did not do so; `model.eval()` alone left
autograd enabled and the 4,033-token prefill retained a training graph until
the H20 exhausted memory. V2 places the entire warmup and measured pair loop
under one `torch.inference_mode()` scope and fails closed at `run_pair` and
both arm entries if that scope is absent. Therefore common prefill, Q16
conversion, two-request construction, and both timed arms now share the same
inference semantics. The input length and allocator environment remain
unchanged; v2 does not add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

The v1 arms also used different ledger implementations and different kernel
injection paths. V2 routes both through one `build_arm_ledger` factory, which
always constructs `MultiForkHitLedger` with the same explicitly resolved
frozen vLLM `unified_attention` callable. The sole factory argument difference
is `call_observer=None` for baseline versus the live callback for the
instrumented arm. Append observers, KV/GDN ownership witnesses, content
hashing, and audit-artifact persistence remain instrumented-only. The formal
pair validity gate and independent replay both reject ledger-class or kernel
identity asymmetry.

## Timing and allocator boundary

The timing definition is unchanged. Each arm calls GC, releases only free
cached CUDA blocks, synchronizes, resets peak-memory statistics, and records
allocated/reserved bytes immediately before its timer. The timer includes the
common adapter registration, one 16-token model step, operator verification,
backend teardown, and final CUDA synchronization. The instrumented timer also
includes the preregistered observer, ownership, hashing, binary capture, and
atomic receipt work. Both timers stop before `finalize_common_output`, so the
common float32 full-vocabulary D2H copy and semantic hashing remain excluded.
Incremental peak is still `max_memory_allocated - allocated_before`; raw
absolute before/after/peak counters, arm order, and physical slot remain
recorded.

## Frozen local checks

- all eight v2 ledger entries pass `sha256sum -c`;
- all three Python sources compile;
- all eight focused tests pass, including the inference-mode fail-closed
  regression, same-ledger/explicit-kernel static gate, full fake formal replay,
  negative-delta preservation, and capture/semantic tamper gates;
- the CPU/mock preflight passes with `gpu_executed=false` and the frozen
  five-pair summary shape;
- the launcher passes `bash -n`;
- the v2 executable-ledger raw SHA is
  `10ad91ece5bfc08e4608a0d059fc0804081ef6d34307ffe1be8194b09817f1d0`;
- only `source-code-v2.sha256` is executable for a second attempt; the original
  `source-code.sha256` and `readiness.md` are retained as v1 history.

The remotely inspected environment path is
`/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/envs/vllm-cu129-v1`.
Its `bin/python -> python3 -> /root/miniconda3/bin/python3` symlink chain
resolves to `/root/miniconda3/bin/python3.11`, but invocation through the venv
path preserves `sys.prefix` at the venv. The observed packages are Python
3.11, PyTorch `2.11.0+cu129`, Transformers `5.14.1`, and vLLM `0.26.0`. This
matches the environment used by attempt a; the symlink identity is not a
blocker.

## Second-attempt resource and command contract

Required resource isolation is unchanged: one exclusive H20 (logical GPU7),
one process, no service port, no collective, read-only model/PG19/upstream
assets, and experiment-private run/Triton/TorchInductor paths. The second
attempt must use fresh staged package
`/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r29_live_overhead_20260825b`
and fresh run directory
`/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/r29-live-overhead-20260825b`.
Neither directory may alias or reuse attempt a.

After independent upload/hash/path verification and explicit GPU assignment,
the frozen launcher command is:

```bash
R29_PACKAGE_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r29_live_overhead_20260825b \
UPSTREAM_CODE_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_gpu_forkaudit_lifecycle_transfer_20260819c/gpu \
MODEL_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/models/Qwen3.5-35B-A3B-59d61f3 \
MODEL_ARTIFACT_LEDGER=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/models/Qwen3.5-35B-A3B-59d61f3/model-artifacts.sha256 \
MODEL_WEIGHT_LEDGER=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/models/Qwen3.5-35B-A3B-59d61f3/model-weights.sha256 \
PG19_DATA=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/data/pg19/qcomem_pg19_train_smoke64.jsonl \
PG19_MANIFEST=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/data/pg19/qcomem_pg19_train_smoke64.manifest.json \
ENV_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/envs/vllm-cu129-v1 \
RUN_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/r29-live-overhead-20260825b \
R29_GPU_UUID=GPU-2645e3a2-b265-9923-19c5-c3a1fc537ec0 \
bash gpu/launch_r29_live_overhead_1gpu.sh
```

This readiness record does not authorize execution. Any second attempt must
retain a pass, failure, positive, negative, or counterintuitive valid result.
