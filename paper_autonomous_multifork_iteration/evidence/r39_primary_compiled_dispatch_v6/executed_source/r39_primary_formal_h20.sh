#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

# Fresh 8-H20 rerun of the byte-identical RR2 primary runner.  This script
# creates, stops, evicts, or deletes no QS resource.

STAGE_ROOT=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r39_primary_compiled_dispatch_20260827f
ASSET_ROOT=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets
RESULT_ROOT="$ASSET_ROOT/runs/qcomem/r39-primary-compiled-dispatch-20260827f"
EVIDENCE_ROOT="$STAGE_ROOT/paper_autonomous_multifork_iteration/evidence"
PRIMARY_CODE="$EVIDENCE_ROOT/round_04_rr2_package/executed_source/gpu"
PRIMARY_INPUT_PARENT="$EVIDENCE_ROOT/round_04_rr2_package/executed_inputs"
PRIMARY_INPUTS="$PRIMARY_INPUT_PARENT/qcomem_gpu_forkaudit_rr2_formal_20260818w-inputs"
PRIMARY_PROTOCOL_MANIFEST="$PRIMARY_INPUT_PARENT/qcomem_qwen35_vllm_paged_multifork_resident_protocol_manifest.json"
R39_PRIMARY_ROOT="$EVIDENCE_ROOT/r39_primary_compiled_dispatch_v6"
R39_SOURCE="$R39_PRIMARY_ROOT/executed_source"
R39_BASE_ROOT="$R39_SOURCE"
UPSTREAM_PREREGISTRATION="$EVIDENCE_ROOT/round_04_rr2_package/upstream/preregistration"
UPSTREAM_CODE_LEDGER="$UPSTREAM_PREREGISTRATION/code.sha256"
UPSTREAM_FROZEN_IDENTITY="$UPSTREAM_PREREGISTRATION/frozen-identity.json"
UPSTREAM_RELEASE_MANIFEST="$UPSTREAM_PREREGISTRATION/release-manifest.json"
DETACHED_REPOSITORY_ROOT="$EVIDENCE_ROOT/round_04_rr2_package/executed_source"
PRIOR_CONTEXT_FIXTURE="$DETACHED_REPOSITORY_ROOT/paper_autonomous_multifork_iteration/evidence/forkaudit_fp32_calibration_manifest.json"
RESPONSE_PLAN_FIXTURE="$DETACHED_REPOSITORY_ROOT/paper_autonomous_multifork_iteration/review/experiment_response_plan.json"
CALIBRATION_ARCHIVE="$DETACHED_REPOSITORY_ROOT/results/gpu-qwen35-vllm-paged-fair-v2-20260814c"

REAL_ENV="$ASSET_ROOT/envs/vllm-cu129-v1"
REAL_PYTHON="$REAL_ENV/bin/python"
RUNTIME_ROOT="$REAL_ENV/lib/python3.11/site-packages"
MODEL_DIR="$ASSET_ROOT/models/Qwen3.5-35B-A3B-59d61f3"
PG19_DATA="$ASSET_ROOT/data/pg19/qcomem_pg19_train_smoke64.jsonl"
PG19_MANIFEST="$ASSET_ROOT/data/pg19/qcomem_pg19_train_smoke64.manifest.json"
PRIMARY_RUN="$RESULT_ROOT/primary"
CAPTURE_ROOT="$RESULT_ROOT/compiled-dispatch-capture"
FORMAL_BINDING="$RESULT_ROOT/formal-binding"
PREFLIGHT_ROOT="$RESULT_ROOT/preflight"

[[ ! -e "$RESULT_ROOT" ]] || { echo "RESULT_ROOT already exists: $RESULT_ROOT" >&2; exit 2; }
for item in \
  "$REAL_PYTHON" "$RUNTIME_ROOT" "$MODEL_DIR" "$PG19_DATA" "$PG19_MANIFEST" \
  "$PRIMARY_CODE/run_qcomem_qwen35_forkaudit_review_revision.py" \
  "$PRIMARY_CODE/launch_qcomem_qwen35_forkaudit_review_revision_8gpu.sh" \
  "$PRIMARY_INPUTS/code.sha256" \
  "$PRIMARY_INPUTS/model-artifacts.formal.sha256" \
  "$PRIMARY_INPUTS/model-weights.canonical.sha256" \
  "$PRIMARY_INPUTS/rr2-pg19-input-manifest.json" \
  "$PRIMARY_INPUTS/rr2-frozen-query-banks.json" \
  "$PRIMARY_INPUTS/rr2-oracle-selection-plan.json" \
  "$PRIMARY_INPUTS/prior-fp32-context-manifest.json" \
  "$PRIMARY_INPUTS/review-experiment-response-plan.json" \
  "$PRIMARY_PROTOCOL_MANIFEST" \
  "$UPSTREAM_CODE_LEDGER" \
  "$UPSTREAM_FROZEN_IDENTITY" \
  "$UPSTREAM_RELEASE_MANIFEST" \
  "$PRIOR_CONTEXT_FIXTURE" \
  "$RESPONSE_PLAN_FIXTURE" \
  "$CALIBRATION_ARCHIVE/scientific-artifacts.sha256" \
  "$CALIBRATION_ARCHIVE/pg19-gate-shards/pg19-fair-v2-shard-0.json" \
  "$CALIBRATION_ARCHIVE/pg19-gate-shards/pg19-fair-v2-shard-1.json" \
  "$CALIBRATION_ARCHIVE/pg19-gate-shards/pg19-fair-v2-shard-2.json" \
  "$CALIBRATION_ARCHIVE/pg19-gate-shards/pg19-fair-v2-shard-3.json" \
  "$CALIBRATION_ARCHIVE/pg19-gate-shards/pg19-fair-v2-shard-4.json" \
  "$CALIBRATION_ARCHIVE/pg19-gate-shards/pg19-fair-v2-shard-5.json" \
  "$CALIBRATION_ARCHIVE/pg19-gate-shards/pg19-fair-v2-shard-6.json" \
  "$CALIBRATION_ARCHIVE/pg19-gate-shards/pg19-fair-v2-shard-7.json" \
  "$R39_BASE_ROOT/r39_compiled_dispatch_receipts.py" \
  "$R39_SOURCE/r39_primary_compact_dispatch.py" \
  "$R39_SOURCE/r39_primary_rank_entrypoint.py" \
  "$R39_SOURCE/r39_primary_finalize.py" \
  "$R39_SOURCE/r39_verify_tf514_gdn_routes.py" \
  "$R39_SOURCE/python_proxy_env/bin/python"; do
  [[ -e "$item" ]] || { echo "required path absent: $item" >&2; exit 2; }
done

(cd "$R39_PRIMARY_ROOT" && sha256sum -c source-code.sha256)
(cd "$R39_PRIMARY_ROOT" && sha256sum -c dependency-files.sha256)
(cd "$R39_PRIMARY_ROOT" && sha256sum -c focused-test-fixtures.source.sha256)
(cd "$R39_PRIMARY_ROOT" && sha256sum -c focused-test-fixtures.archive.sha256)
(cd "$PRIMARY_CODE" && sha256sum -c "$PRIMARY_INPUTS/code.sha256")
[[ "$(sha256sum "$PG19_DATA" | awk '{print $1}')" == "ef18893b6bfb0f4b8cb29eab85ccf2a0ef1fdb44606e9742a70405cca564e18c" ]]
[[ "$(sha256sum "$PG19_MANIFEST" | awk '{print $1}')" == "5d789d67aa239f089e92de8a4267b86d2f1d2723d5f1370970883738f5f89a9c" ]]

# The old executed-input preregister-probe predates the current immutable
# runner ledger and is not an authority for this fresh run.  Fail closed on
# the already-existing RR2 upstream preregistration, which binds the current
# 34-file code ledger and was frozen before this rerun or any candidate output.
[[ "$(sha256sum "$UPSTREAM_CODE_LEDGER" | awk '{print $1}')" == \
  "837f7a488d75cbedbc01e35a236a97f00b85259746e6a368b7aeec873045e94a" ]]
cmp -s "$UPSTREAM_CODE_LEDGER" "$PRIMARY_INPUTS/code.sha256" || {
  echo "upstream preregistration code ledger differs from executed input" >&2
  exit 2
}
[[ "$(sha256sum "$UPSTREAM_FROZEN_IDENTITY" | awk '{print $1}')" == \
  "150bad1abe7e2db320d7b6557dc42b119201f0128b9a02e862b377299664737e" ]]
[[ "$(sha256sum "$UPSTREAM_RELEASE_MANIFEST" | awk '{print $1}')" == \
  "05465256c451b14a65ede6329d56cedb56e70388c4ff7ec064bdfd6d4c7f3fcb" ]]
UPSTREAM_FROZEN_IDENTITY_SEMANTIC_SHA256=$(
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PRIMARY_CODE" \
    "$REAL_PYTHON" -B \
    "$PRIMARY_CODE/build_qcomem_qwen35_forkaudit_review_manifest.py" \
    digest-json --input "$UPSTREAM_FROZEN_IDENTITY"
)
[[ "$UPSTREAM_FROZEN_IDENTITY_SEMANTIC_SHA256" == \
  "4fa076706e6a90729d5392c18fd79d99efcec971e54b0d01d13245f3ae816882" ]]
UPSTREAM_RELEASE_SEMANTIC_SHA256=$(
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PRIMARY_CODE" \
    "$REAL_PYTHON" -B \
    "$PRIMARY_CODE/build_qcomem_qwen35_forkaudit_review_manifest.py" \
    digest-json --input "$UPSTREAM_RELEASE_MANIFEST"
)
[[ "$UPSTREAM_RELEASE_SEMANTIC_SHA256" == \
  "201b15c945676db1924bcf2e197ee93a10078feda81ec0f4a8da113c56fac456" ]]
env PYTHONDONTWRITEBYTECODE=1 "$REAL_PYTHON" -I -B -c '
import json, pathlib, sys
release = json.loads(pathlib.Path(sys.argv[1]).read_text())
identity = json.loads(pathlib.Path(sys.argv[2]).read_text())
expected = sys.argv[3]
assert release["frozen_identity"] == identity
assert release["frozen_identity"]["code_ledger_sha256"] == expected
assert release["frozen_identity_sha256"] == sys.argv[4]
assert release["source_artifacts"]["code_ledger"] == {
    "bytes": 3655,
    "sha256": expected,
}
assert pathlib.Path(sys.argv[5]).read_bytes() == pathlib.Path(sys.argv[6]).read_bytes()
' "$UPSTREAM_RELEASE_MANIFEST" "$UPSTREAM_FROZEN_IDENTITY" \
  "837f7a488d75cbedbc01e35a236a97f00b85259746e6a368b7aeec873045e94a" \
  "4fa076706e6a90729d5392c18fd79d99efcec971e54b0d01d13245f3ae816882" \
  "$UPSTREAM_CODE_LEDGER" "$PRIMARY_INPUTS/code.sha256"

FIXTURE_COPY_COUNT=0
while read -r _digest source_relative; do
  projection_prefix=focused_test_fixtures/repository_projection/
  [[ "$source_relative" == "$projection_prefix"* ]] || {
    echo "fixture source ledger path escaped frozen projection: $source_relative" >&2
    exit 2
  }
  projected_relative=${source_relative#"$projection_prefix"}
  cmp -s \
    "$R39_PRIMARY_ROOT/$source_relative" \
    "$DETACHED_REPOSITORY_ROOT/$projected_relative" || {
      echo "archive fixture projection differs from its frozen source: $projected_relative" >&2
      exit 2
    }
  FIXTURE_COPY_COUNT=$((FIXTURE_COPY_COUNT + 1))
done < "$R39_PRIMARY_ROOT/focused-test-fixtures.source.sha256"
[[ "$FIXTURE_COPY_COUNT" -eq 11 ]] || {
  echo "expected exactly 11 frozen focused-test fixture files" >&2
  exit 2
}
cmp -s "$PRIOR_CONTEXT_FIXTURE" "$PRIMARY_INPUTS/prior-fp32-context-manifest.json" || {
  echo "projected prior-context fixture differs from the frozen RR2 input" >&2
  exit 2
}
cmp -s "$RESPONSE_PLAN_FIXTURE" "$PRIMARY_INPUTS/review-experiment-response-plan.json" || {
  echo "projected response-plan fixture differs from the frozen RR2 input" >&2
  exit 2
}

# Gate the complete detached layout before the immutable launcher creates any
# primary shard.  The old launcher repeats the same 162-test suite inside its
# own frozen preflight; this earlier gate prevents another missing-fixture
# attempt from reaching that launcher's run lifecycle.
mkdir -p "$PREFLIGHT_ROOT/logs" "$PREFLIGHT_ROOT/stages" "$PREFLIGHT_ROOT/pycache"
date -u +%FT%TZ > "$PREFLIGHT_ROOT/stages/00_upstream_preregistration_authority_ok"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$PREFLIGHT_ROOT/pycache"
timeout --signal=TERM --kill-after=60s 1800s \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  PYTHONPATH="$PRIMARY_CODE" "$REAL_PYTHON" -B -m unittest -v \
  test_qcomem_forkaudit_storage_witness \
  test_qcomem_forkaudit_oracle \
  test_qcomem_forkaudit_mutants \
  test_qcomem_vllm_paged_multifork_resident \
  test_qcomem_qwen35_vllm_paged_integration \
  test_build_qcomem_forkaudit_rr2_input_manifest \
  test_build_qcomem_forkaudit_fp32_calibration_manifest \
  test_run_qcomem_qwen35_forkaudit_review_revision \
  test_launch_qcomem_qwen35_forkaudit_review_revision \
  > "$PREFLIGHT_ROOT/logs/detached-focused-tests-162.log" 2>&1
if grep -Eq 'skipped=|\.\.\. skipped|SKIP' \
  "$PREFLIGHT_ROOT/logs/detached-focused-tests-162.log"; then
  echo "detached focused test suite contained a skip" >&2
  exit 2
fi
grep -Eq '^Ran 162 tests in ' "$PREFLIGHT_ROOT/logs/detached-focused-tests-162.log"
grep -Eq '^test_real_tf514_qwen_call_consumes_and_advances_position_ids .* \.\.\. ok$' \
  "$PREFLIGHT_ROOT/logs/detached-focused-tests-162.log"
grep -Eq '^OK$' "$PREFLIGHT_ROOT/logs/detached-focused-tests-162.log"
date -u +%FT%TZ > "$PREFLIGHT_ROOT/stages/01_detached_focused_tests_162_ok"

env PYTHONDONTWRITEBYTECODE=1 "$REAL_PYTHON" -I -B \
  "$R39_SOURCE/r39_verify_tf514_gdn_routes.py" \
  --runtime-root "$RUNTIME_ROOT" \
  --output "$PREFLIGHT_ROOT/tf514-gdn-route-static.json"
grep -Eq '"status":"pass"' "$PREFLIGHT_ROOT/tf514-gdn-route-static.json"
date -u +%FT%TZ > "$PREFLIGHT_ROOT/stages/02_tf514_gdn_routes_static_ok"

timeout --signal=TERM --kill-after=30s 600s \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  PYTHONPATH="$R39_SOURCE:$R39_BASE_ROOT" "$REAL_PYTHON" -B \
  -m unittest discover -s "$R39_PRIMARY_ROOT/tests" -p 'test_r39_*.py' -v \
  > "$PREFLIGHT_ROOT/logs/r39-primary-tests-13.log" 2>&1
if grep -Eq 'skipped=|\.\.\. skipped|SKIP' \
  "$PREFLIGHT_ROOT/logs/r39-primary-tests-13.log"; then
  echo "Round-39 primary test suite contained a skip" >&2
  exit 2
fi
grep -Eq '^Ran 13 tests in ' "$PREFLIGHT_ROOT/logs/r39-primary-tests-13.log"
grep -Eq '^OK$' "$PREFLIGHT_ROOT/logs/r39-primary-tests-13.log"
date -u +%FT%TZ > "$PREFLIGHT_ROOT/stages/03_r39_primary_tests_13_ok"

if find "$PRIMARY_CODE" \
  \( -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \) \) \
  -print -quit | grep -q .; then
  echo "detached preflight contaminated the immutable primary code with bytecode" >&2
  exit 2
fi
date -u +%FT%TZ > "$PREFLIGHT_ROOT/stages/04_primary_code_bytecode_absent"

# The original launcher independently rejects writable code.  Make only its
# exact, hash-checked copied snapshot read-only; do not change external assets.
chmod -R a-w "$PRIMARY_CODE"
chmod +x "$R39_SOURCE/python_proxy_env/bin/python"

export R39_PRIMARY_REAL_PYTHON="$REAL_PYTHON"
export R39_PRIMARY_EXPECTED_RUNNER="$PRIMARY_CODE/run_qcomem_qwen35_forkaudit_review_revision.py"
export R39_PRIMARY_RANK_WRAPPER="$R39_SOURCE/r39_primary_rank_entrypoint.py"
export R39_PRIMARY_CODE_ROOT="$PRIMARY_CODE"
export R39_PRIMARY_RUNTIME_ROOT="$RUNTIME_ROOT"
export R39_PRIMARY_BASE_ROOT="$R39_BASE_ROOT"
export R39_PRIMARY_SOURCE_ROOT="$R39_SOURCE"
export R39_PRIMARY_CAPTURE_ROOT="$CAPTURE_ROOT"

CODE_DIR="$PRIMARY_CODE" \
MODEL_DIR="$MODEL_DIR" \
MODEL_ARTIFACT_LEDGER_FILE="$PRIMARY_INPUTS/model-artifacts.formal.sha256" \
MODEL_WEIGHT_LEDGER_FILE="$PRIMARY_INPUTS/model-weights.canonical.sha256" \
PG19_DATA="$PG19_DATA" \
PG19_MANIFEST="$PG19_MANIFEST" \
PG19_INPUT_MANIFEST="$PRIMARY_INPUTS/rr2-pg19-input-manifest.json" \
PRIOR_CAPACITY_MANIFEST="$PRIMARY_PROTOCOL_MANIFEST" \
FROZEN_QUERY_BANKS_INPUT="$PRIMARY_INPUTS/rr2-frozen-query-banks.json" \
PROTOCOL_SOURCE_MANIFEST="$PRIMARY_PROTOCOL_MANIFEST" \
ORACLE_SELECTION_INPUT="$PRIMARY_INPUTS/rr2-oracle-selection-plan.json" \
PRIOR_FP32_CONTEXT_MANIFEST="$PRIMARY_INPUTS/prior-fp32-context-manifest.json" \
REVIEW_RESPONSE_PLAN="$PRIMARY_INPUTS/review-experiment-response-plan.json" \
RUN_DIR="$PRIMARY_RUN" \
ENV_DIR="$R39_SOURCE/python_proxy_env" \
MODEL_ID=Qwen/Qwen3.5-35B-A3B \
MODEL_REVISION=59d61f3ce65a6d9863b86d2e96597125219dc754 \
EXPECTED_CODE_LEDGER_SHA256=837f7a488d75cbedbc01e35a236a97f00b85259746e6a368b7aeec873045e94a \
EXPECTED_MODEL_MANIFEST_SHA256=72c9e06109702dbca958a6a528d6686b68d6b8e3376d116c0261b4c319e3da29 \
EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256=c0a23e9d3f9d220257af97b78fd97661f315f0c82a3a010b57a771e3eeefbbfb \
EXPECTED_MODEL_WEIGHT_LEDGER_SHA256=8314a82c9188b9b817193e039b0b0eb0636b328512f19b0c12455853b7e20014 \
EXPECTED_PG19_SHA256=ef18893b6bfb0f4b8cb29eab85ccf2a0ef1fdb44606e9742a70405cca564e18c \
EXPECTED_PG19_MANIFEST_SHA256=5d789d67aa239f089e92de8a4267b86d2f1d2723d5f1370970883738f5f89a9c \
EXPECTED_PG19_WINDOWS_SHA256=39bc36bb2eb04d51122e66caaebfa72367c02b43b073f072a2da240ed068c166 \
EXPECTED_PG19_INPUT_MANIFEST_SHA256=9ab6f7b2c2fc91c457f61d6b869660a97364cc76383d28b28071627d7141c16c \
EXPECTED_PRIOR_CAPACITY_MANIFEST_SHA256=975bc6a12f43447024b889889d4156ca71c2f89b68de6157ac609b4a9687e9c0 \
EXPECTED_FROZEN_QUERY_BANKS_INPUT_SHA256=400921d147bc840e9802950dc542b002080f3f274661efbd5b4354ec364da7db \
EXPECTED_ORACLE_SELECTION_INPUT_SHA256=fcfa8a61f231c7faa7284ff81d89952e8b311ca7f70f3a552f628b3268bad59f \
EXPECTED_PRIOR_FP32_CONTEXT_MANIFEST_SHA256=fa64f663bb74a190a0a5c0898fda2a55528171c77a91af2b1321c24a5f310a1d \
EXPECTED_REVIEW_RESPONSE_PLAN_SHA256=e2be05e198d6c86276f229d4e862c3579c65e311d07c219e9e9c792a390cbfbb \
EXPECTED_PROTOCOL_SOURCE_MANIFEST_SHA256=975bc6a12f43447024b889889d4156ca71c2f89b68de6157ac609b4a9687e9c0 \
EXPECTED_RELEASE_MANIFEST_SHA256=201b15c945676db1924bcf2e197ee93a10078feda81ec0f4a8da113c56fac456 \
EXPECTED_ORACLE_SELECTION_PLAN_SHA256=26d472140170f12dd53897456b09d57f247808576dd1e576023c6f710d90003b \
EXPECTED_FROZEN_QUERY_BANKS_SHA256=abb2037cbc4a2d364432423f88e79d0ad36a78f4b4678a3b670ddc05231b88e9 \
EXPECTED_RUNNER_SHA256=9da619fc037e2c670b146d778fd9f4d5344212b7e525f3d3f26a077f79d67775 \
bash "$PRIMARY_CODE/launch_qcomem_qwen35_forkaudit_review_revision_8gpu.sh"

PYTHONPATH="$R39_SOURCE:$R39_BASE_ROOT" "$REAL_PYTHON" -I -B -c \
  'import runpy,sys; roots=sys.argv[1:3]; script=sys.argv[3]; sys.path[:0]=roots; sys.argv=[script,*sys.argv[4:]]; runpy.run_path(script,run_name="__main__")' \
  "$R39_SOURCE" "$R39_BASE_ROOT" "$R39_SOURCE/r39_primary_finalize.py" \
  --primary-run-root "$PRIMARY_RUN" \
  --capture-root "$CAPTURE_ROOT" \
  --code-root "$PRIMARY_CODE" \
  --runtime-root "$RUNTIME_ROOT" \
  --output "$FORMAL_BINDING"

(
  cd "$RESULT_ROOT"
  find preflight primary compiled-dispatch-capture formal-binding -type f \
    ! -path '*/terminal-files.sha256' ! -name COMPLETE -print0 | \
    sort -z | xargs -0 sha256sum > terminal-files.sha256
)
touch "$RESULT_ROOT/COMPLETE"
