#!/usr/bin/env bash
set -euo pipefail
set +x

# Infrastructure-only staging for the frozen A11 formal run.  This script does
# not execute FIT, CAL, TEST, or any other scientific workload.
TARGET=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/a11_boolq_ordered_formal_20260822a_actual
STAGE=${TARGET}.provisioning-20260824e
DATASET_EXPORT=/mnt/tidal-alsh-share2/qs_dataset/liuhanzuo/10836/10959/2026_08_24_02_30
CODE=${STAGE}/code
DATA=${STAGE}/data/boolq_35b264d_all_12697.jsonl
MODEL=${STAGE}/model/Qwen2.5-7B-Instruct-a09a354

[[ -n "${PAYLOAD_ROOT:-}" ]] || { echo 'PAYLOAD_ROOT is required' >&2; exit 2; }
[[ "$PAYLOAD_ROOT" =~ ^/tmp/a11-stage-bundle\.[[:alnum:]]{6}$ ]] || { echo "unexpected payload root: $PAYLOAD_ROOT" >&2; exit 2; }
[[ $(readlink -f -- "$PAYLOAD_ROOT") == "$PAYLOAD_ROOT" ]] || { echo "non-canonical payload root: $PAYLOAD_ROOT" >&2; exit 2; }
[[ -d "$PAYLOAD_ROOT/code" && ! -L "$PAYLOAD_ROOT/code" ]] || { echo 'payload code directory is invalid' >&2; exit 2; }
[[ -d "$PAYLOAD_ROOT/model_assets" && ! -L "$PAYLOAD_ROOT/model_assets" ]] || { echo 'payload model-assets directory is invalid' >&2; exit 2; }
[[ -n "${QS_USER:-}" && -n "${QS_TOKEN:-}" ]] || { echo 'QS model credentials are unavailable' >&2; exit 2; }
[[ -d "$DATASET_EXPORT" && ! -L "$DATASET_EXPORT" ]] || { echo "dataset export is not a regular directory: $DATASET_EXPORT" >&2; exit 3; }
[[ ! -e "$TARGET" ]] || { echo "target already exists: $TARGET" >&2; exit 3; }
[[ ! -e "$STAGE" ]] || { echo "fresh provisioning path already exists: $STAGE" >&2; exit 4; }

mkdir -p "$CODE" "$(dirname "$DATA")" "$MODEL" "$STAGE/runs"

code_files=(README.md protocol.json run_boolq_ordered.py test_run_boolq_ordered.py launch_boolq_ordered_8gpu.sh qs_preview.json)
for filename in "${code_files[@]}"; do
  [[ -f "$PAYLOAD_ROOT/code/$filename" && ! -L "$PAYLOAD_ROOT/code/$filename" ]] || {
    echo "missing or non-regular bundled code file: $filename" >&2
    exit 2
  }
  cp -- "$PAYLOAD_ROOT/code/$filename" "$CODE/$filename"
done
(
  cd "$CODE"
  sha256sum -c <<'EOF'
230535fd52fc5c737099d734a628853c57e4bd1b0ccd830d23c8e324eaaa3a04  README.md
1a6593a97b67509d0afc954bef653924f7f9cdc055592d0ba75b08987b71cb31  protocol.json
f36e9ee066f7c37d455490b74f78056cb987ed382f5d8e7e35c0ecbb00f75a29  run_boolq_ordered.py
ba1d6987820334318c7f8a51e915c23d1b30f1c65d51abb8b293843033b64c94  test_run_boolq_ordered.py
8faea7e75cf716d8fdb5edf0301ddfbc5bc7b49b7d64152097dfe1661715165e  launch_boolq_ordered_8gpu.sh
a4d2ed9f3ac9e819ada35e45801170673621c250fbf5b580f02d93756027a29e  qs_preview.json
EOF
  bash -n launch_boolq_ordered_8gpu.sh
  PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -q test_run_boolq_ordered.py
)

export DATASET_EXPORT DATA
python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

source = Path(os.environ["DATASET_EXPORT"])
output = Path(os.environ["DATA"])
required = {"answer", "passage", "question", "source_index", "source_split"}
files = sorted(path for path in source.rglob("*") if path.is_file() and not path.is_symlink())
if not files:
    raise SystemExit("dataset export contains no regular files")

def exported_rows(path):
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    if not text.strip():
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if rows:
                    raise SystemExit(f"invalid JSON after records at {path}:{line_number}")
                return []
            if not isinstance(row, dict) or not required.issubset(row):
                if rows:
                    raise SystemExit(f"invalid record after records at {path}:{line_number}")
                return []
            rows.append(row)
        return rows
    if isinstance(payload, dict) and required.issubset(payload):
        return [payload]
    if isinstance(payload, list) and all(isinstance(row, dict) and required.issubset(row) for row in payload):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        nested = payload["data"]
        if all(isinstance(row, dict) and required.issubset(row) for row in nested):
            return nested
    return []

records = []
for path in files:
    rows = exported_rows(path)
    if rows:
        print(f"dataset records source={path} rows={len(rows)}")
    for row_number, row in enumerate(rows, 1):
        split = row["source_split"]
        index = row["source_index"]
        if split not in {"train", "validation"} or type(index) is not int or index < 0:
            raise SystemExit(f"invalid source identity at {path} record {row_number}")
        if type(row["answer"]) is not bool:
            raise SystemExit(f"invalid Boolean answer at {path} record {row_number}")
        records.append({
            "answer": row["answer"],
            "passage": str(row["passage"]),
            "question": str(row["question"]),
            "source_index": index,
            "source_split": split,
        })

expected_counts = {"train": 9427, "validation": 3270}
counts = {split: sum(row["source_split"] == split for row in records) for split in expected_counts}
if counts != expected_counts or len(records) != 12697:
    raise SystemExit(f"dataset count mismatch: total={len(records)} splits={counts}")
identities = {(row["source_split"], row["source_index"]) for row in records}
if len(identities) != len(records):
    raise SystemExit("duplicate source identity in dataset export")
records.sort(key=lambda row: ((0 if row["source_split"] == "train" else 1), row["source_index"]))
with output.open("w", encoding="utf-8", newline="") as handle:
    for row in records:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
digest = hashlib.sha256(output.read_bytes()).hexdigest()
if output.stat().st_size != 8835655 or digest != "13c2f4143ae320a0191c6de5be919248a20c15515f58c6deb7d3732068f2d31a":
    raise SystemExit(f"canonical dataset mismatch: bytes={output.stat().st_size} sha256={digest}")
print(f"canonical dataset verified {digest}")
PY

python3 -m pip install --no-cache-dir requests \
  -i http://pypi.devops.xiaohongshu.com/simple/ \
  --trusted-host pypi.devops.xiaohongshu.com
python3 -m pip install --no-cache-dir --no-build-isolation quicksilver-toolkit \
  -i http://pypi.devops.xiaohongshu.com/simple/ \
  --trusted-host pypi.devops.xiaohongshu.com
TOOLKIT_VERSION=$(python3 -c 'import importlib.metadata; print(importlib.metadata.version("quicksilver-toolkit"))')
export TOOLKIT_VERSION

MODEL_DL=$(mktemp -d /tmp/a11-model-dl.XXXXXX)
cleanup_temps() {
  local path
  for path in "$MODEL_DL" "$PAYLOAD_ROOT"; do
    if [[ "$path" =~ ^/tmp/a11-(model-dl|stage-bundle)\.[[:alnum:]]{6}$ ]] \
      && [[ -d "$path" && ! -L "$path" ]] \
      && [[ $(readlink -f -- "$path") == "$path" ]]; then
      rm -rf -- "$path"
    else
      echo "refusing to clean unexpected path: $path" >&2
    fi
  done
}
trap cleanup_temps EXIT
export MODEL_DL
python3 - <<'PY'
import inspect
import os

try:
    from model_tools.model import download_model
except ImportError:
    from model_tools import download_model

kwargs = {
    "save_path": os.environ["MODEL_DL"],
    "model_name": "Qwen2.5-7B-Instruct",
    "version": 2,
    "subversion": 1736233087,
    "region": "aliyun-shanghai",
    "user": os.environ["QS_USER"],
    "token": os.environ["QS_TOKEN"],
    "env": "prod",
    "max_retry": 3,
    "num_threads": 10,
}
if "zone" in inspect.signature(download_model).parameters:
    kwargs["zone"] = "cn"
download_model(**kwargs)
PY

MODEL_WRAPPED=$MODEL_DL/Qwen2.5-7B-Instruct
if [[ -d "$MODEL_WRAPPED" && ! -L "$MODEL_WRAPPED" ]]; then
  MODEL_SOURCE=$MODEL_WRAPPED
elif [[ -f "$MODEL_DL/config.json" && ! -L "$MODEL_DL/config.json" ]]; then
  MODEL_SOURCE=$MODEL_DL
else
  echo "unexpected model download layout under: $MODEL_DL" >&2
  exit 5
fi
core_files=(
  LICENSE config.json generation_config.json merges.txt
  model-00001-of-00004.safetensors
  model-00002-of-00004.safetensors
  model-00003-of-00004.safetensors
  model-00004-of-00004.safetensors
  model.safetensors.index.json tokenizer.json tokenizer_config.json vocab.json
)
for filename in "${core_files[@]}"; do
  [[ -f "$MODEL_SOURCE/$filename" && ! -L "$MODEL_SOURCE/$filename" ]] || {
    echo "missing or non-regular downloaded model file: $filename" >&2
    exit 6
  }
  cp -- "$MODEL_SOURCE/$filename" "$MODEL/$filename"
done
for filename in README.md .gitattributes; do
  [[ -f "$PAYLOAD_ROOT/model_assets/$filename" && ! -L "$PAYLOAD_ROOT/model_assets/$filename" ]] || {
    echo "missing or non-regular bundled model asset: $filename" >&2
    exit 6
  }
  cp -- "$PAYLOAD_ROOT/model_assets/$filename" "$MODEL/$filename"
done

[[ $(find "$MODEL" -mindepth 1 -maxdepth 1 -type f | wc -l) == 14 ]] || { echo 'model file count mismatch' >&2; exit 7; }
[[ $(find "$MODEL" -mindepth 1 -maxdepth 1 -type l | wc -l) == 0 ]] || { echo 'model symlink found' >&2; exit 7; }
(
  cd "$MODEL"
  sha256sum -c <<'EOF'
11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361  .gitattributes
832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e  LICENSE
f366f33bbf6bcadbb7d87f0a21a7b65584a56b8d58b0743c77c88bee625b93a6  README.md
7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c  config.json
3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f  generation_config.json
599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3  merges.txt
a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7  model-00001-of-00004.safetensors
f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185  model-00002-of-00004.safetensors
8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5  model-00003-of-00004.safetensors
1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd  model-00004-of-00004.safetensors
624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028  model.safetensors.index.json
c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539  tokenizer.json
5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583  tokenizer_config.json
ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910  vocab.json
EOF
  ledger_sha=$(LC_ALL=C sha256sum .gitattributes LICENSE README.md config.json generation_config.json merges.txt model-00001-of-00004.safetensors model-00002-of-00004.safetensors model-00003-of-00004.safetensors model-00004-of-00004.safetensors model.safetensors.index.json tokenizer.json tokenizer_config.json vocab.json | sha256sum | awk '{print $1}')
  [[ "$ledger_sha" == 3ee6c9510b7e50bfcd46d6df33cafa3e2019f13a6a09bf1d2f9e80cdfe1164e8 ]] || { echo 'model ledger hash mismatch' >&2; exit 8; }
)

export STAGE_ROOT="$STAGE"
python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

root = Path(os.environ["STAGE_ROOT"])
record = {
    "schema": "a11-authoritative-tidalfs-staging-v2",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "dataset_source": "qs-dataset:10836/version:10959/cloud:6/export:870",
    "dataset_sha256": "13c2f4143ae320a0191c6de5be919248a20c15515f58c6deb7d3732068f2d31a",
    "model_source": "qs-model:443/version:2/subversion:1736233087",
    "model_tools_version": os.environ["TOOLKIT_VERSION"],
    "model_snapshot_manifest_sha256": "3ee6c9510b7e50bfcd46d6df33cafa3e2019f13a6a09bf1d2f9e80cdfe1164e8",
    "protocol_sha256": "1a6593a97b67509d0afc954bef653924f7f9cdc055592d0ba75b08987b71cb31",
    "runner_sha256": "f36e9ee066f7c37d455490b74f78056cb987ed382f5d8e7e35c0ecbb00f75a29",
    "launcher_sha256": "8faea7e75cf716d8fdb5edf0301ddfbc5bc7b49b7d64152097dfe1661715165e",
}
(root / "STAGING_SUCCESS.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY

[[ ! -e "$TARGET" ]] || { echo "target appeared during staging: $TARGET" >&2; exit 9; }
sync
mv -T -- "$STAGE" "$TARGET"
printf 'A11_QS_INTERNAL_STAGING_SUCCESS target=%s\n' "$TARGET"
sha256sum "$TARGET/STAGING_SUCCESS.json" "$TARGET/data/boolq_35b264d_all_12697.jsonl" \
  "$TARGET/code/protocol.json" "$TARGET/code/run_boolq_ordered.py" "$TARGET/code/launch_boolq_ordered_8gpu.sh"
