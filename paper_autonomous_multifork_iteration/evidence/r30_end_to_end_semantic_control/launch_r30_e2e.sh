#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_debug/qcomem-r30-8h20-multiagent-20260825a/r30-e2e-semantic-control-gpu4-20260825a
PYTHON=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/envs/vllm-cu129-v1/bin/python
MODEL=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/models/Qwen3.5-35B-A3B-59d61f3
REPAIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r30_postdiscovery_d_clean_20260825c/qcomem_single_token_gdn_ownership.py
CLEAN_ROOT=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/r30-postdiscovery-d-clean-20260825c
CLEAN_RESULT="$CLEAN_ROOT/raw/clean-result.json"
DETACHED_REPLAY="$CLEAN_ROOT/receipts/detached-replay.json"
INPUT_SHA=ecd2208219e183f2f1e5ad057ff536fe02c2b1a46b6f1ae9bf8feb042a52aa42
REPAIR_SHA=4a2938cc99503f54abf91f780034e08ae64e4105a51c0736433b84ff363bad7a
CLEAN_RESULT_SHA=a7758f1b28dad20276660710d31dbb0a63a33a8d9b0ffd34f76282c81f3492ef
DETACHED_REPLAY_SHA=fd5d8562cb15d15e49246f9df8330cf3c138f3f209f4ebacc31ec89a9ca9d60e
MODEL_ARTIFACT_LEDGER_SHA=c0a23e9d3f9d220257af97b78fd97661f315f0c82a3a010b57a771e3eeefbbfb
MODEL_WEIGHT_LEDGER_SHA=8314a82c9188b9b817193e039b0b0eb0636b328512f19b0c12455853b7e20014
GPU_UUID=GPU-d917fce5-80f1-78ac-3965-0476bf8bd441
ATTEMPT="$ROOT/results/frozen-attempt-001"

[[ "$PWD" == "$ROOT" ]]
[[ -f code/SOURCE.sha256 ]]
sha256sum -c code/SOURCE.sha256
printf '%s  %s\n' "$INPUT_SHA" input-manifest.json | sha256sum -c -
printf '%s  %s\n' "$REPAIR_SHA" "$REPAIR" | sha256sum -c -
printf '%s  %s\n' "$CLEAN_RESULT_SHA" "$CLEAN_RESULT" | sha256sum -c -
printf '%s  %s\n' "$DETACHED_REPLAY_SHA" "$DETACHED_REPLAY" | sha256sum -c -
printf '%s  %s\n' "$MODEL_ARTIFACT_LEDGER_SHA" "$MODEL/model-artifacts.sha256" | sha256sum -c -
printf '%s  %s\n' "$MODEL_WEIGHT_LEDGER_SHA" "$MODEL/model-weights.sha256" | sha256sum -c -
[[ "$(jq -r .status "$CLEAN_RESULT")" == valid_clean_positive ]]
[[ "$(jq -r .status "$DETACHED_REPLAY")" == detached_clean_replay_passed ]]
[[ ! -e "$ATTEMPT" ]]

gpu_line="$(nvidia-smi --id=4 --query-gpu=index,uuid,name,memory.used --format=csv,noheader,nounits)"
[[ "$gpu_line" == "4, $GPU_UUID, NVIDIA H20-3e, 0" ]]
if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits | grep -q "^$GPU_UUID,"; then
  echo "physical GPU4 is occupied" >&2
  exit 1
fi

mkdir -p "$ATTEMPT/logs" "$ATTEMPT/artifacts"
cp input-manifest.json "$ATTEMPT/input-manifest.json"
cp code/SOURCE.sha256 "$ATTEMPT/SOURCE.sha256"
cp code/EXECUTION_CONTRACT.md "$ATTEMPT/EXECUTION_CONTRACT.md"
sha256sum "$ATTEMPT/input-manifest.json" "$ATTEMPT/SOURCE.sha256" "$ATTEMPT/EXECUTION_CONTRACT.md" > "$ATTEMPT/prerun-artifacts.sha256"

export CUDA_VISIBLE_DEVICES=4
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/code/upstream"
set +e
timeout 7200 "$PYTHON" "$ROOT/code/r30_e2e_reference.py" \
  --input-manifest "$ATTEMPT/input-manifest.json" \
  --expected-input-sha256 "$INPUT_SHA" \
  --model "$MODEL" \
  --artifact-root "$ATTEMPT/artifacts" \
  --candidate-sentinel "$ATTEMPT/artifacts/candidate-result.json" \
  --output "$ATTEMPT/artifacts/reference-result.json" \
  > "$ATTEMPT/logs/reference.stdout.jsonl" \
  2> "$ATTEMPT/logs/reference.stderr.log"
reference_rc=$?
set -e
printf '%s\n' "$reference_rc" > "$ATTEMPT/logs/reference.exit-code.txt"
[[ "$reference_rc" == 0 ]]
REFERENCE_SHA="$(sha256sum "$ATTEMPT/artifacts/reference-result.json" | awk '{print $1}')"
printf '%s\n' "$REFERENCE_SHA" > "$ATTEMPT/reference-result.sha256"

set +e
timeout 14400 "$PYTHON" "$ROOT/code/r30_e2e_candidate.py" \
  --input-manifest "$ATTEMPT/input-manifest.json" \
  --expected-input-sha256 "$INPUT_SHA" \
  --reference "$ATTEMPT/artifacts/reference-result.json" \
  --expected-reference-sha256 "$REFERENCE_SHA" \
  --model "$MODEL" \
  --repair-source "$REPAIR" \
  --clean-result-sha256 "$CLEAN_RESULT_SHA" \
  --detached-replay-sha256 "$DETACHED_REPLAY_SHA" \
  --artifact-root "$ATTEMPT/artifacts" \
  --output "$ATTEMPT/artifacts/candidate-result.json" \
  > "$ATTEMPT/logs/candidate.stdout.jsonl" \
  2> "$ATTEMPT/logs/candidate.stderr.log"
candidate_rc=$?
set -e
printf '%s\n' "$candidate_rc" > "$ATTEMPT/logs/candidate.exit-code.txt"
[[ "$candidate_rc" == 0 ]]
CANDIDATE_SHA="$(sha256sum "$ATTEMPT/artifacts/candidate-result.json" | awk '{print $1}')"
printf '%s\n' "$CANDIDATE_SHA" > "$ATTEMPT/candidate-result.sha256"

set +e
CUDA_VISIBLE_DEVICES='' PYTHONPATH='' timeout 1200 "$PYTHON" -I "$ROOT/code/r30_e2e_replay.py" \
  --input-manifest "$ATTEMPT/input-manifest.json" \
  --expected-input-sha256 "$INPUT_SHA" \
  --reference "$ATTEMPT/artifacts/reference-result.json" \
  --expected-reference-sha256 "$REFERENCE_SHA" \
  --candidate "$ATTEMPT/artifacts/candidate-result.json" \
  --expected-candidate-sha256 "$CANDIDATE_SHA" \
  --artifact-root "$ATTEMPT/artifacts" \
  --expected-repair-sha256 "$REPAIR_SHA" \
  --expected-clean-result-sha256 "$CLEAN_RESULT_SHA" \
  --expected-detached-replay-sha256 "$DETACHED_REPLAY_SHA" \
  --output "$ATTEMPT/artifacts/independent-replay.json" \
  > "$ATTEMPT/logs/replay.stdout.jsonl" \
  2> "$ATTEMPT/logs/replay.stderr.log"
replay_rc=$?
set -e
printf '%s\n' "$replay_rc" > "$ATTEMPT/logs/replay.exit-code.txt"
[[ "$replay_rc" == 0 ]]

find "$ATTEMPT" -type f ! -name final-artifacts.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$ATTEMPT/final-artifacts.sha256"
sha256sum "$ATTEMPT/artifacts/reference-result.json" \
  "$ATTEMPT/artifacts/candidate-result.json" \
  "$ATTEMPT/artifacts/independent-replay.json" \
  "$ATTEMPT/final-artifacts.sha256"
jq '{status,infrastructure_valid,ownership_gate_passed,primary_gate_passed,scientific_outcome,exact_generated_token_gate,full_vocabulary_secondary:{comparisons:.full_vocabulary_secondary.comparisons,summary:.full_vocabulary_secondary.summary}}' \
  "$ATTEMPT/artifacts/independent-replay.json"
