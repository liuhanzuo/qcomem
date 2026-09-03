# R30 replacement-node execution contract

The original job 249885 / trial 1898483 terminated before this package was
uploaded or any scientific execution began.  Do not target that pod again.

## Replacement node

Minimum for this experiment alone: one exclusive NVIDIA H20-3e/H20-141G with
at least 140 GiB device memory, 32 CPU cores, 256 GiB host RAM, and the shared
`diandian` mount.  For the authorized multi-agent campaign, the known-working
replacement is an exact clone of job 249885:

- image: `artifactory.devops.xiaohongshu.com/media/redaccel:0.9.1-gpu`
- queue: `Verifier` (`queue_id=471`)
- cloud/cluster: `cloud_id=6`, `cluster_id=53`
- resource package: `183`
- one worker with 170 CPU, 1800 GiB RAM, and 8 H20-141G GPUs
- writable mount: `/mnt/tidal-alsh-hilab/dataset/diandian`
- no service port, no collective, no elastic restart
- persistent-node command:

```bash
bash -lc 'umask 077; mkdir -p /mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_debug/qcomem-r30-native-batching-20260825a; cd /mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_debug/qcomem-r30-native-batching-20260825a; exec sleep infinity'
```

Reserve exactly one idle physical GPU for this lane.  The frozen command below
uses GPU1; change both `CUDA_VISIBLE_DEVICES` and `R30_PHYSICAL_GPU_INDEX`
together before execution if the replacement assignment differs.  Verify the
chosen device with `nvidia-smi` first.

## Required shared assets

```text
/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/envs/vllm-cu129-v1
/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/models/Qwen3.5-35B-A3B-59d61f3
/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/data/pg19/qcomem_pg19_train_smoke64.jsonl
```

The venv must report Python 3.11.13, torch 2.11.0+cu129, Transformers 5.14.1,
vLLM 0.26.0, and Triton 3.6.0.  The frozen preregistration raw SHA is
`5c6b27304bd19ef9564cc703511133fcf8d903e080b344650874290a42185626`.

## In-pod commands after package upload

Stage the package at the fresh path shown below and retain all stdout/stderr.
The static phase hashes all 67 GiB of model weights and builds exact token IDs
before the model is loaded.

```bash
R30_STAGE=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r30_native_batching_20260825a
R30_RUN=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/r30-native-batching-20260825a
R30_ENV=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/envs/vllm-cu129-v1
R30_MODEL=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/models/Qwen3.5-35B-A3B-59d61f3
R30_DATA=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/data/pg19/qcomem_pg19_train_smoke64.jsonl
R30_PREREG_SHA=5c6b27304bd19ef9564cc703511133fcf8d903e080b344650874290a42185626

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

R30_INPUT_SHA=$(sha256sum "$R30_RUN/static/input_manifest.json" | awk '{print $1}')
nvidia-smi -i 1 --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

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

After process exit, copy the four terminal logs into `R30_RUN`, append their
SHA-256 values to `SHA256SUMS`, run the detached replay once more, and download
the entire run directory.  Do not stop the replacement node without explicit
confirmation.
