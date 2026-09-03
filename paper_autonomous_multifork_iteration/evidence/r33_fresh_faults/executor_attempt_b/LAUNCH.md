# Exact R33 Attempt-B H20 launch

Formal protocol SHA-256:

`7a5172e212e8cfb1541f7c8b901c72099a141ea9da55d2f7f14e4296cabb5ad4`

After extracting the Attempt-B package at the path below on Trial `1900821`,
run exactly:

```bash
PACKAGE_ROOT=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r33_fresh_faults_20260825b
OUTPUT_ROOT=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/r33-fresh-faults-20260825b
PROTOCOL="$PACKAGE_ROOT/paper_autonomous_multifork_iteration/evidence/r33_fresh_faults/executor_attempt_b/formal-protocol.json"
EXECUTION_INPUT="$PACKAGE_ROOT/paper_autonomous_multifork_iteration/evidence/r29_heldout_faults/cross_execution/execution-input-v3.json"
FAULTS_JSON="$PACKAGE_ROOT/paper_autonomous_multifork_iteration/evidence/r33_fresh_faults/author_freeze/FAULTS.json"
ENV_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/envs/vllm-cu129-v1
RR2_CODE_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_gpu_forkaudit_rr2_formal_20260818w/gpu

sha256sum "$PROTOCOL" "$EXECUTION_INPUT" "$FAULTS_JSON"
bash "$PACKAGE_ROOT/paper_autonomous_multifork_iteration/scripts/r33_launch_fresh_faults.sh" \
  "$PACKAGE_ROOT" \
  "$OUTPUT_ROOT" \
  "$PROTOCOL" \
  "$EXECUTION_INPUT" \
  "$FAULTS_JSON" \
  "$ENV_DIR" \
  "$RR2_CODE_DIR" \
  1900821
```

The three pre-launch digests must be, in order:

```text
7a5172e212e8cfb1541f7c8b901c72099a141ea9da55d2f7f14e4296cabb5ad4
5a522e48650e3010621e6e06c7c8bbab67c074bca1b6a6a0c70aa50133b4e98d
b1f4d6c544c30fccc32370a03e170aee38596a370d02c0db4a6748c83cc34dff
```

The launcher refuses an existing output root, starts one full clean/mutant pair
on each of the first five physical GPU UUIDs, and emits the aggregate only if
all five rank processes complete successfully.
