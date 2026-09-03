#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 CODE_ROOT EXECUTION_PACKAGE RESOURCE_AMENDMENT EXPECTED_AMENDMENT_SHA256" >&2
  exit 64
fi

code_root="$1"
execution_package="$2"
resource_amendment="$3"
expected_amendment_sha256="$4"

expected_code_root="/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r35_historical_alias_20260826a"
environment_python="/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/envs/vllm-cu129-v1/bin/python"
rr2_root="/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_gpu_forkaudit_rr2_formal_20260818w/gpu"
paper_root="$code_root/paper_autonomous_multifork_iteration"
evidence_root="$paper_root/evidence/r35_historical_alias_regression"
runner="$paper_root/scripts/r35_run_historical_alias_regression.py"
replay="$paper_root/scripts/r35_replay_historical_alias_regression.py"
aggregator="$paper_root/scripts/r35_aggregate_historical_alias_regression.py"
preregistration="$evidence_root/preregistration.json"
execution_input="$evidence_root/static-execution-input.json"
source_ledger="$evidence_root/source.sha256"

if [[ "$code_root" != "$expected_code_root" ]]; then
  echo "R35 launcher refused unexpected code root: $code_root" >&2
  exit 65
fi
for required in "$environment_python" "$execution_package" "$resource_amendment" "$runner" "$replay" "$aggregator" "$preregistration" "$execution_input" "$source_ledger"; do
  if [[ ! -f "$required" ]]; then
    echo "R35 launcher missing required file: $required" >&2
    exit 66
  fi
done

hash_file() {
  "$environment_python" -I -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "$1"
}

amendment_sha256="$(hash_file "$resource_amendment")"
if [[ "$amendment_sha256" != "$expected_amendment_sha256" ]]; then
  echo "R35 launcher resource-amendment hash drift" >&2
  exit 67
fi

preregistration_sha256="$(hash_file "$preregistration")"
execution_input_sha256="$(hash_file "$execution_input")"
source_ledger_sha256="$(hash_file "$source_ledger")"
execution_package_sha256="$(hash_file "$execution_package")"

"$environment_python" -I -c '
import json, pathlib, re, sys
amendment = json.loads(pathlib.Path(sys.argv[1]).read_bytes())
expected = {
    "preregistration_raw_sha256": sys.argv[2],
    "execution_input_raw_sha256": sys.argv[3],
    "source_ledger_raw_sha256": sys.argv[4],
    "execution_package_sha256": sys.argv[5],
}
if amendment.get("status") != "frozen_after_resource_creation_before_candidate_outputs":
    raise SystemExit("resource amendment is not preexecution-frozen")
if amendment.get("candidate_output_seen_when_frozen") is not False:
    raise SystemExit("resource amendment is not outcome-blind")
if amendment.get("science_design_changed") is not False:
    raise SystemExit("resource amendment changed the science design")
for key, digest in expected.items():
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or amendment.get(key) != digest:
        raise SystemExit(f"resource amendment binding drift: {key}")
assignments = amendment.get("gpu_assignments")
if not isinstance(assignments, dict) or set(assignments) != {str(rank) for rank in range(8)}:
    raise SystemExit("resource amendment GPU rank coverage drift")
uuids = []
for rank in range(8):
    row = assignments[str(rank)]
    if set(row) != {"physical_index", "uuid"} or row["physical_index"] != rank:
        raise SystemExit(f"resource amendment GPU assignment drift: rank {rank}")
    if not isinstance(row["uuid"], str) or not row["uuid"].startswith("GPU-"):
        raise SystemExit(f"resource amendment GPU UUID drift: rank {rank}")
    uuids.append(row["uuid"])
if len(set(uuids)) != 8:
    raise SystemExit("resource amendment GPU UUIDs are not unique")
' "$resource_amendment" "$preregistration_sha256" "$execution_input_sha256" "$source_ledger_sha256" "$execution_package_sha256"

"$environment_python" -I -c '
import hashlib, pathlib, re, sys
root = pathlib.Path(sys.argv[1]).resolve()
ledger = pathlib.Path(sys.argv[2])
seen = set()
for line_number, raw in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
    if not raw.strip():
        continue
    pieces = raw.split(None, 1)
    if len(pieces) != 2 or re.fullmatch(r"[0-9a-f]{64}", pieces[0]) is None:
        raise SystemExit(f"invalid source ledger line {line_number}")
    relative = pathlib.Path(pieces[1].lstrip("*"))
    if relative.is_absolute() or ".." in relative.parts or str(relative) in seen:
        raise SystemExit(f"unsafe or duplicate source ledger path: {relative}")
    seen.add(str(relative))
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise SystemExit(f"missing source-ledger target: {relative}")
    observed = hashlib.sha256(target.read_bytes()).hexdigest()
    if observed != pieces[0]:
        raise SystemExit(f"source-ledger hash drift: {relative}")
if not seen:
    raise SystemExit("empty source ledger")
' "$code_root" "$source_ledger"

run_root="$($environment_python -I -c 'import json, pathlib, sys; print(json.loads(pathlib.Path(sys.argv[1]).read_bytes())["output"]["run_root"])' "$execution_input")"
expected_run_root="/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/r35-historical-alias-20260826a"
if [[ "$run_root" != "$expected_run_root" || -e "$run_root" ]]; then
  echo "R35 launcher output root drift or pre-exists: $run_root" >&2
  exit 68
fi
mkdir -p "$run_root/logs" "$run_root/replay"

pids=()
terminate_children() {
  local pid
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap terminate_children INT TERM EXIT

for rank in 0 1 2 3 4 5 6 7; do
  gpu_uuid="$($environment_python -I -c 'import json, pathlib, sys; print(json.loads(pathlib.Path(sys.argv[1]).read_bytes())["gpu_assignments"][sys.argv[2]]["uuid"])' "$resource_amendment" "$rank")"
  CUDA_VISIBLE_DEVICES="$gpu_uuid" \
  PYTHONPATH="$code_root/gpu:$rr2_root" \
  TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=8 \
  "$environment_python" "$runner" \
    --execution-input "$execution_input" \
    --expected-execution-input-sha256 "$execution_input_sha256" \
    --resource-amendment "$resource_amendment" \
    --expected-resource-amendment-sha256 "$amendment_sha256" \
    --rank "$rank" \
    --physical-gpu-index "$rank" \
    --expected-gpu-uuid "$gpu_uuid" \
    --run-dir "$run_root/rank-$rank" \
    >"$run_root/logs/rank-$rank.log" 2>&1 &
  pids+=("$!")
done

runner_failure=0
for index in 0 1 2 3 4 5 6 7; do
  if ! wait "${pids[$index]}"; then
    echo "R35 rank $index failed; see $run_root/logs/rank-$index.log" >&2
    runner_failure=1
  fi
done
pids=()
trap - INT TERM EXIT
if [[ "$runner_failure" -ne 0 ]]; then
  exit 69
fi

for rank in 0 1 2 3 4 5 6 7; do
  "$environment_python" -I "$replay" \
    --rank-result "$run_root/rank-$rank/raw/rank-result.json" \
    --raw-root "$run_root/rank-$rank/raw" \
    --preregistration "$preregistration" \
    --expected-preregistration-sha256 "$preregistration_sha256" \
    --amendment "$resource_amendment" \
    --expected-amendment-sha256 "$amendment_sha256" \
    --output "$run_root/replay/rank-$rank-replay.json" \
    >"$run_root/logs/replay-rank-$rank.log" 2>&1
done

"$environment_python" -I "$aggregator" \
  --replay-root "$run_root/replay" \
  --expected-preregistration-sha256 "$preregistration_sha256" \
  --expected-amendment-sha256 "$amendment_sha256" \
  --expected-execution-input-sha256 "$execution_input_sha256" \
  --expected-source-ledger-sha256 "$source_ledger_sha256" \
  --output "$run_root/aggregate.json" \
  >"$run_root/logs/aggregate.log" 2>&1

"$environment_python" -I -c '
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
paths = [root / "aggregate.json"]
paths.extend(root / "replay" / f"rank-{rank}-replay.json" for rank in range(8))
receipt = {
    "schema_version": "forkaudit-r35-launch-completion-v1",
    "status": "completed",
    "artifacts": [
        {
            "path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    ],
}
target = root / "launch-completion.json"
with target.open("xb") as handle:
    handle.write(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode() + b"\n")
' "$run_root"

echo "R35 completed: $run_root/aggregate.json"
