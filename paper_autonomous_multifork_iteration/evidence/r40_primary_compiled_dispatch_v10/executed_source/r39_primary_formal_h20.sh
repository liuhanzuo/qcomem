#!/usr/bin/env bash
set -Eeuo pipefail
export LC_ALL=C

# Fresh one-shot 8-H20 rerun of the byte-identical RR2 primary runner.  This
# script creates, stops, evicts, or deletes no QS resource.  It is intentionally
# HOLD until an independent archive audit and explicit authorization.

STAGE_ROOT=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r40_primary_compiled_dispatch_v10_20260827j
ASSET_ROOT=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets
RESULT_ROOT="$ASSET_ROOT/runs/qcomem/r40-primary-compiled-dispatch-v10-20260827j"
EVIDENCE_ROOT="$STAGE_ROOT/paper_autonomous_multifork_iteration/evidence"
PRIMARY_CODE="$EVIDENCE_ROOT/round_04_rr2_package/executed_source/gpu"
PRIMARY_INPUT_PARENT="$EVIDENCE_ROOT/round_04_rr2_package/executed_inputs"
PRIMARY_INPUTS="$PRIMARY_INPUT_PARENT/qcomem_gpu_forkaudit_rr2_formal_20260818w-inputs"
PRIMARY_PROTOCOL_MANIFEST="$PRIMARY_INPUT_PARENT/qcomem_qwen35_vllm_paged_multifork_resident_protocol_manifest.json"
R40_PRIMARY_ROOT="$EVIDENCE_ROOT/r40_primary_compiled_dispatch_v10"
R40_SOURCE="$R40_PRIMARY_ROOT/executed_source"
R40_BASE_ROOT="$R40_SOURCE"
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
LAUNCH_IDENTITY_ROOT="$RESULT_ROOT/rank-launch-identities"
RUNTIME_PREFLIGHT="$PREFLIGHT_ROOT/runtime-preflight.json"
PRIMARY_LAUNCHER="$PRIMARY_CODE/launch_qcomem_qwen35_forkaudit_review_revision_8gpu.sh"

[[ "${R40_H20_EXECUTION_AUTHORIZED:-}" == "yes" ]] || {
  echo "formal H20 execution requires R40_H20_EXECUTION_AUTHORIZED=yes" >&2
  exit 2
}
[[ -d "$PRIMARY_CODE" ]] || {
  echo "detached primary code directory absent: $PRIMARY_CODE" >&2
  exit 2
}

# Reject a contaminated detached snapshot before consuming the fixed one-shot
# result root.  The same check is repeated after all Python preflight work so
# pre-existing and newly-created bytecode are both fail-closed.
if find "$PRIMARY_CODE" \
  \( -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \) \) \
  -print -quit | grep -q .; then
  echo "detached primary code contains forbidden Python bytecode" >&2
  exit 2
fi
[[ ! -e "$RESULT_ROOT" ]] || {
  echo "one-shot RESULT_ROOT already exists: $RESULT_ROOT" >&2
  exit 2
}

umask 077
mkdir -p "$RESULT_ROOT/supervisor"
OUTER_STDOUT="$RESULT_ROOT/supervisor/outer.stdout.log"
OUTER_STDERR="$RESULT_ROOT/supervisor/outer.stderr.log"
exec 3>&1 4>&2
exec >"$OUTER_STDOUT" 2>"$OUTER_STDERR"
CURRENT_PHASE=outer_bootstrap
OUTER_FINALIZED=false

outer_exit() {
  status=$?
  if [[ "$status" -eq 0 || "$OUTER_FINALIZED" == true ]]; then
    return
  fi
  set +e
  trap - EXIT ERR INT TERM
  exec 1>&3 2>&4
  failure="$RESULT_ROOT/supervisor/FAILURE.json"
  failure_tmp="$failure.tmp-$$"
  if [[ ! -e "$failure" && ! -e "$failure_tmp" ]]; then
    printf '{"complete_written":false,"exit_status":%d,"failed_phase":"%s","one_shot_retry_allowed":false,"schema_version":"forkaudit-r40-outer-failure-v1"}\n' \
      "$status" "$CURRENT_PHASE" > "$failure_tmp"
    ln "$failure_tmp" "$failure"
    rm -f "$failure_tmp"
  fi
  partial="$RESULT_ROOT/supervisor/failure-terminal-files.sha256"
  partial_tmp="$partial.tmp-$$"
  if [[ ! -e "$partial" && ! -e "$partial_tmp" ]]; then
    (
      cd "$RESULT_ROOT" || exit
      find . -type f \
        ! -path './supervisor/failure-terminal-files.sha256' \
        ! -path './supervisor/failure-terminal-files.sha256.tmp-*' \
        ! -name COMPLETE -print0 | sort -z | xargs -0 sha256sum
    ) > "$partial_tmp"
    ln "$partial_tmp" "$partial"
    rm -f "$partial_tmp"
  fi
  exit "$status"
}
trap outer_exit EXIT
trap 'status=$?; exit "$status"' ERR
trap 'exit 130' INT
trap 'exit 143' TERM

printf '%s\n' \
  '{"authorization_gate":"R40_H20_EXECUTION_AUTHORIZED=yes","authorization_value_persisted":false,"authorized":true,"schema_version":"forkaudit-r40-explicit-execution-authorization-v1"}' \
  > "$RESULT_ROOT/supervisor/execution-authorization.json"
printf '{"bash_version":"%s","expected_gpu_count":8,"execution_mode":"one-shot-no-retry","lc_all":"C","package_version_capture":"runtime-preflight.json","schema_version":"forkaudit-r40-outer-environment-allowlist-v1"}\n' \
  "$BASH_VERSION" \
  > "$RESULT_ROOT/supervisor/environment-allowlist.json"

CURRENT_PHASE=source_and_dependency_integrity
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
  "$R40_BASE_ROOT/r39_compiled_dispatch_receipts.py" \
  "$R40_SOURCE/r39_primary_compact_dispatch.py" \
  "$R40_SOURCE/r39_primary_rank_entrypoint.py" \
  "$R40_SOURCE/r39_primary_finalize.py" \
  "$R40_SOURCE/r39_verify_tf514_gdn_routes.py" \
  "$R40_SOURCE/r40_runtime_smoke.py" \
  "$R40_SOURCE/python_proxy_env/bin/python"; do
  [[ -e "$item" ]] || { echo "required path absent: $item" >&2; exit 2; }
done
"$REAL_PYTHON" --version > "$RESULT_ROOT/supervisor/python-version.stdout.txt" 2> "$RESULT_ROOT/supervisor/python-version.stderr.txt"

(cd "$R40_PRIMARY_ROOT" && sha256sum -c source-code.sha256)
(cd "$R40_PRIMARY_ROOT" && sha256sum -c dependency-files.sha256)
(cd "$R40_PRIMARY_ROOT" && sha256sum -c focused-test-fixtures.source.sha256)
(cd "$R40_PRIMARY_ROOT" && sha256sum -c focused-test-fixtures.archive.sha256)
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
    "$R40_PRIMARY_ROOT/$source_relative" \
    "$DETACHED_REPOSITORY_ROOT/$projected_relative" || {
      echo "archive fixture projection differs from its frozen source: $projected_relative" >&2
      exit 2
    }
  FIXTURE_COPY_COUNT=$((FIXTURE_COPY_COUNT + 1))
done < "$R40_PRIMARY_ROOT/focused-test-fixtures.source.sha256"
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
CURRENT_PHASE=detached_focused_tests
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
  "$R40_SOURCE/r39_verify_tf514_gdn_routes.py" \
  --runtime-root "$RUNTIME_ROOT" \
  --output "$PREFLIGHT_ROOT/tf514-gdn-route-static.json"
grep -Eq '"status":"pass"' "$PREFLIGHT_ROOT/tf514-gdn-route-static.json"
date -u +%FT%TZ > "$PREFLIGHT_ROOT/stages/02_tf514_gdn_routes_static_ok"

CURRENT_PHASE=frozen_runtime_no_cuda_smoke
env PYTHONDONTWRITEBYTECODE=1 "$REAL_PYTHON" -I -B -c \
  'import runpy,sys; sys.path[:0]=sys.argv[1:3]; script=sys.argv[3]; sys.argv=[script,*sys.argv[4:]]; runpy.run_path(script,run_name="__main__")' \
  "$R40_SOURCE" "$PRIMARY_CODE" "$R40_SOURCE/r40_runtime_smoke.py" \
  --code-root "$PRIMARY_CODE" \
  --runtime-root "$RUNTIME_ROOT" \
  --cache-root "$PREFLIGHT_ROOT/runtime-smoke-triton-cache" \
  --output "$RUNTIME_PREFLIGHT"
RUNTIME_PREFLIGHT_SHA256=$(sha256sum "$RUNTIME_PREFLIGHT" | awk '{print $1}')
[[ "$RUNTIME_PREFLIGHT_SHA256" =~ ^[0-9a-f]{64}$ ]]
grep -Eq '"cuda_initialized_after":false' "$RUNTIME_PREFLIGHT"
date -u +%FT%TZ > "$PREFLIGHT_ROOT/stages/03_frozen_runtime_no_cuda_smoke_ok"

CURRENT_PHASE=r40_local_security_tests
timeout --signal=TERM --kill-after=30s 600s \
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  PYTHONPATH="$R40_SOURCE:$R40_BASE_ROOT" "$REAL_PYTHON" -B \
  -m unittest discover -s "$R40_PRIMARY_ROOT/tests" -p 'test_*.py' -v \
  > "$PREFLIGHT_ROOT/logs/r40-primary-tests-27.log" 2>&1
if grep -Eq 'skipped=|\.\.\. skipped|SKIP' \
  "$PREFLIGHT_ROOT/logs/r40-primary-tests-27.log"; then
  echo "Round-40 primary test suite contained a skip" >&2
  exit 2
fi
grep -Eq '^Ran 27 tests in ' "$PREFLIGHT_ROOT/logs/r40-primary-tests-27.log"
grep -Eq '^OK$' "$PREFLIGHT_ROOT/logs/r40-primary-tests-27.log"
date -u +%FT%TZ > "$PREFLIGHT_ROOT/stages/04_r40_primary_tests_27_ok"

if find "$PRIMARY_CODE" \
  \( -type d -name __pycache__ -o -type f \( -name '*.pyc' -o -name '*.pyo' \) \) \
  -print -quit | grep -q .; then
  echo "detached preflight contaminated the immutable primary code with bytecode" >&2
  exit 2
fi
date -u +%FT%TZ > "$PREFLIGHT_ROOT/stages/05_primary_code_bytecode_absent"

# The original launcher independently rejects writable code.  Make only its
# exact, hash-checked copied snapshot read-only; do not change external assets.
chmod -R a-w "$PRIMARY_CODE"
chmod +x "$R40_SOURCE/python_proxy_env/bin/python"
mkdir -p "$LAUNCH_IDENTITY_ROOT"

export R40_PRIMARY_REAL_PYTHON="$REAL_PYTHON"
export R40_PRIMARY_EXPECTED_RUNNER="$PRIMARY_CODE/run_qcomem_qwen35_forkaudit_review_revision.py"
export R40_PRIMARY_RANK_WRAPPER="$R40_SOURCE/r39_primary_rank_entrypoint.py"
export R40_PRIMARY_CODE_ROOT="$PRIMARY_CODE"
export R40_PRIMARY_RUNTIME_ROOT="$RUNTIME_ROOT"
export R40_PRIMARY_BASE_ROOT="$R40_BASE_ROOT"
export R40_PRIMARY_SOURCE_ROOT="$R40_SOURCE"
export R40_PRIMARY_CAPTURE_ROOT="$CAPTURE_ROOT"
export R40_PRIMARY_RUNTIME_PREFLIGHT="$RUNTIME_PREFLIGHT"
export R40_PRIMARY_RUNTIME_PREFLIGHT_SHA256="$RUNTIME_PREFLIGHT_SHA256"
export R40_PRIMARY_LAUNCHER="$PRIMARY_LAUNCHER"
export R40_PRIMARY_LAUNCH_IDENTITY_ROOT="$LAUNCH_IDENTITY_ROOT"
export R40_PRIMARY_MODEL_ID=Qwen/Qwen3.5-35B-A3B
export R40_PRIMARY_MODEL_REVISION=59d61f3ce65a6d9863b86d2e96597125219dc754

CURRENT_PHASE=authorized_eight_rank_primary
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
ENV_DIR="$R40_SOURCE/python_proxy_env" \
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

CURRENT_PHASE=exact_eight_rank_finalizer
PYTHONPATH="$R40_SOURCE:$R40_BASE_ROOT" "$REAL_PYTHON" -I -B -c \
  'import runpy,sys; roots=sys.argv[1:3]; script=sys.argv[3]; sys.path[:0]=roots; sys.argv=[script,*sys.argv[4:]]; runpy.run_path(script,run_name="__main__")' \
  "$R40_SOURCE" "$R40_BASE_ROOT" "$R40_SOURCE/r39_primary_finalize.py" \
  --primary-run-root "$PRIMARY_RUN" \
  --capture-root "$CAPTURE_ROOT" \
  --code-root "$PRIMARY_CODE" \
  --runtime-root "$RUNTIME_ROOT" \
  --runtime-preflight-manifest "$RUNTIME_PREFLIGHT" \
  --expected-runtime-preflight-sha256 "$RUNTIME_PREFLIGHT_SHA256" \
  --launch-identity-root "$LAUNCH_IDENTITY_ROOT" \
  --output "$FORMAL_BINDING"

CURRENT_PHASE=terminal_closure
date -u +%FT%TZ > "$RESULT_ROOT/supervisor/formal-pipeline-returned-ok"
exec 1>&3 2>&4
exec 3>&- 4>&-
terminal_tmp="$RESULT_ROOT/terminal-files.sha256.tmp-$$"
(
  cd "$RESULT_ROOT"
  find supervisor preflight primary compiled-dispatch-capture \
    formal-binding rank-launch-identities -type f \
    ! -path '*/terminal-files.sha256' \
    ! -name 'terminal-files.sha256' \
    ! -name 'terminal-files.sha256.tmp-*' \
    ! -name COMPLETE -print0 | sort -z | xargs -0 sha256sum
) > "$terminal_tmp"
ln "$terminal_tmp" "$RESULT_ROOT/terminal-files.sha256"
rm -f "$terminal_tmp"
( set -o noclobber; : > "$RESULT_ROOT/COMPLETE" )
OUTER_FINALIZED=true
trap - EXIT ERR INT TERM
echo "R40 formal pipeline completed once: $RESULT_ROOT"
