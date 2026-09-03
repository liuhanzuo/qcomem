# Exact R33 H20 launch

The frozen formal protocol SHA-256 is:

`7d0cc087b6b529aa41a8003cf748121b3adedc7433cb6c75dbde50da5ba62fb7`

After extracting the launch package at the path below on Trial `1900821`, run exactly:

```bash
PACKAGE_ROOT=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r33_fresh_faults_20260825a
OUTPUT_ROOT=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/r33-fresh-faults-20260825a
PROTOCOL="$PACKAGE_ROOT/paper_autonomous_multifork_iteration/evidence/r33_fresh_faults/executor_preparation/formal-protocol.json"
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
7d0cc087b6b529aa41a8003cf748121b3adedc7433cb6c75dbde50da5ba62fb7
5a522e48650e3010621e6e06c7c8bbab67c074bca1b6a6a0c70aa50133b4e98d
b1f4d6c544c30fccc32370a03e170aee38596a370d02c0db4a6748c83cc34dff
```

The launcher uses the first five physical GPU UUIDs, runs one matched pair per GPU, waits for every rank, and only then invokes the detached five-pair aggregate. It refuses an existing output root. A failed rank leaves its raw clean/mutant/invalid artifact and logs in place but prevents `summary.json` from being emitted.
