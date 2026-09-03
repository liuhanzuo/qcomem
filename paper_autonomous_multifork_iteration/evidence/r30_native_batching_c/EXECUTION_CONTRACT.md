# R30C execution contract

## Authorization state

This package is frozen for review.  **Do not execute the formal GPU command
until the root agent explicitly approves the frozen hashes.**  Never stop or
delete the QS node from this lane.

## Fixed target and assets

- job 249885 / trial 1899487
- pod `qs-249885-1899487-ai-1443683-master-0`
- physical GPU1 only, after a fresh `nvidia-smi` idle-device audit
- image `artifactory.devops.xiaohongshu.com/media/redaccel:0.9.1-gpu`
- model: `/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/models/Qwen3.5-35B-A3B-59d61f3`
- data: `/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/data/pg19/qcomem_pg19_train_smoke64.jsonl`
- venv: `/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/envs/vllm-cu129-v1`

The package bundle and every internal SHA must verify before extraction and
execution.  The preregistration raw SHA is
`994856185cb7bff0b12508d8bc1796aa570789d6d9b4122cb84ac0035b56a0d3`.

## Fresh paths

```bash
R30_STAGE=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r30_native_batching_20260825c
R30_RUN=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/r30-native-batching-20260825c
R30_ENV=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/envs/vllm-cu129-v1
R30_MODEL=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/models/Qwen3.5-35B-A3B-59d61f3
R30_DATA=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/data/pg19/qcomem_pg19_train_smoke64.jsonl
R30_PREREG_SHA=994856185cb7bff0b12508d8bc1796aa570789d6d9b4122cb84ac0035b56a0d3
```

Both fresh paths must be absent.  R30B is immutable and must not be reused or
overwritten.

## Static phase (CPU only)

```bash
test ! -e "$R30_RUN"
mkdir -p "$R30_RUN"
"$R30_ENV/bin/python" "$R30_STAGE/executed_source/r30_run_native_batching.py" \
  --mode build-static \
  --model "$R30_MODEL" \
  --data "$R30_DATA" \
  --model-artifact-ledger "$R30_STAGE/model-artifacts.sha256" \
  --model-weight-ledger "$R30_STAGE/model-weights.sha256" \
  --preregistration "$R30_STAGE/preregistration.json" \
  --expected-prereg-sha256 "$R30_PREREG_SHA" \
  --output-dir "$R30_RUN" \
  >"$R30_STAGE/build-static.stdout.log" \
  2>"$R30_STAGE/build-static.stderr.log"
```

This phase must hard-gate tokenizer width 248,077, model/logits width 248,320,
243 padded slots, exact installed vLLM source SHAs, and both synthetic mapping
fixtures before writing `MODEL_LEDGER_VERIFIED`.

## Formal phase (only after explicit approval)

```bash
R30_INPUT_SHA=$(sha256sum "$R30_RUN/static/input_manifest.json" | awk '{print $1}')
nvidia-smi -i 1 --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits

CUDA_VISIBLE_DEVICES=1 R30_PHYSICAL_GPU_INDEX=1 \
"$R30_ENV/bin/python" "$R30_STAGE/executed_source/r30_run_native_batching.py" \
  --mode execute \
  --model "$R30_MODEL" \
  --data "$R30_DATA" \
  --model-artifact-ledger "$R30_STAGE/model-artifacts.sha256" \
  --model-weight-ledger "$R30_STAGE/model-weights.sha256" \
  --preregistration "$R30_STAGE/preregistration.json" \
  --expected-prereg-sha256 "$R30_PREREG_SHA" \
  --expected-input-sha256 "$R30_INPUT_SHA" \
  --output-dir "$R30_RUN" \
  >"$R30_STAGE/formal.stdout.log" \
  2>"$R30_STAGE/formal.stderr.log"
```

After process exit, preserve all terminal logs, run the detached replay once
more into a distinct file, generate a terminal SHA ledger, and retain the full
run.  A failed gate remains internal and is never reframed or automatically
inserted into the manuscript.
