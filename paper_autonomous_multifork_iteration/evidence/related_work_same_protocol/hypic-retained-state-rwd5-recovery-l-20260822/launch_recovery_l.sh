#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

K_ROOT=/tmp/rwd5-hypic-store-freeze-k
L_ROOT=/tmp/rwd5-hypic-store-recovery-l
MODEL_ROOT=/tmp/Qwen3.5-35B-A3B-hypic-model-view
K_MANIFEST=${K_ROOT}/SHA256SUMS
L_MANIFEST=${L_ROOT}/SHA256SUMS
EXPECTED_K_MANIFEST_SHA256=c7f0fcc0b44d6292af52f2d31a0770e7a74982c20eadc8317740106034dc3a7b
EXPECTED_L_MANIFEST_SHA256=${EXPECTED_RECOVERY_MANIFEST_SHA256:?supply externally audited recovery-L SHA256SUMS SHA256}
RUN_DIR_L=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-retained-state-rwd5-20260822l
INSTRUMENTED_REPO_L=/tmp/HYPIC-98147c0-rwd5-store-l
RECOVERY_RECEIPT_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-retained-state-rwd5-recovery-l-20260822
REPAIR_RECEIPT=${L_ROOT}/asset-mode-repair-receipt.json
K_LAUNCHER=${K_ROOT}/code/launch_hypic_retained_state_bytes_8gpu.sh

die() { printf '%s\n' "ERROR: $*" >&2; exit 1; }

for path in "$K_MANIFEST" "$L_MANIFEST" "$REPAIR_RECEIPT" "$K_LAUNCHER" "$MODEL_ROOT"; do
  [[ -e "$path" ]] || die "missing frozen path: $path"
done
[[ "$(sha256sum "$K_MANIFEST" | awk '{print $1}')" == "$EXPECTED_K_MANIFEST_SHA256" ]] || die "K manifest identity drift"
[[ "$(sha256sum "$L_MANIFEST" | awk '{print $1}')" == "$EXPECTED_L_MANIFEST_SHA256" ]] || die "L manifest identity drift"
(cd "$K_ROOT" && sha256sum -c "$K_MANIFEST") || die "K frozen files drift"
(cd "$L_ROOT" && sha256sum -c "$L_MANIFEST") || die "L frozen files drift"

check_asset() {
  local name=$1 expected_sha=$2 expected_size=$3 expected_inode=$4
  local path=${MODEL_ROOT}/${name}
  [[ -f "$path" && ! -L "$path" ]] || die "asset is not a regular non-symlink: $name"
  [[ "$(sha256sum "$path" | awk '{print $1}')" == "$expected_sha" ]] || die "asset SHA drift: $name"
  [[ "$(stat -c '%a|%u|%g|%s|%i|%d|%Y|%Z' "$path")" == "444|0|0|${expected_size}|${expected_inode}|2097177|1787372672|1787376685" ]] \
    || die "asset post-repair stat drift: $name"
}

check_asset model-artifacts.sha256 \
  d78424684a27718b44616c80c788e7f8b726feed674fe015901b87de1993f7dd 778 58755972
check_asset preprocessor_config.json \
  27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516 390 58755952
check_asset video_preprocessor_config.json \
  7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13 385 58755969

mapfile -t writable_top < <(find "$MODEL_ROOT" -maxdepth 1 -type f -perm /222 -printf '%f\n' | sort)
[[ ${#writable_top[@]} -eq 0 ]] || die "writable top-level model assets remain: ${writable_top[*]}"
[[ ! -e "$RUN_DIR_L" ]] || die "fresh L RUN_DIR already exists"
[[ ! -e "$INSTRUMENTED_REPO_L" ]] || die "fresh L instrumented repo already exists"
[[ ! -e "$RECOVERY_RECEIPT_DIR" ]] || die "recovery receipt directory already exists"

mkdir -p "$RECOVERY_RECEIPT_DIR"
cp "$REPAIR_RECEIPT" "$RECOVERY_RECEIPT_DIR/asset-mode-repair-receipt.json"
sha256sum "$K_MANIFEST" "$L_MANIFEST" "$RECOVERY_RECEIPT_DIR/asset-mode-repair-receipt.json" \
  > "$RECOVERY_RECEIPT_DIR/frozen-authority.sha256"
printf '%s\n' "preflight_passed_before_k_launcher" > "$RECOVERY_RECEIPT_DIR/STATUS"

exec env \
  FREEZE_ROOT="$K_ROOT" \
  EXPECTED_FREEZE_MANIFEST_SHA256="$EXPECTED_K_MANIFEST_SHA256" \
  MODEL_DIR="$MODEL_ROOT" \
  RUN_DIR="$RUN_DIR_L" \
  INSTRUMENTED_REPO="$INSTRUMENTED_REPO_L" \
  bash "$K_LAUNCHER"
