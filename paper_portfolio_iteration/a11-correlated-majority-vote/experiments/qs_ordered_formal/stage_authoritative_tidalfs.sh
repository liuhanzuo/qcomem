#!/usr/bin/env bash
set -euo pipefail

TARGET=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/a11_boolq_ordered_formal_20260822a_actual
STAGE=${TARGET}.provisioning-20260824a
MODEL=${STAGE}/model/Qwen2.5-7B-Instruct-a09a354
DATA=${STAGE}/data/boolq_35b264d_all_12697.jsonl
SOURCES=${STAGE}/sources
CODE=${STAGE}/code
HF_MODEL=https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/a09a35458c702b33eeacc393d103063234e8bc28
HF_BOOLQ=https://huggingface.co/datasets/google/boolq/resolve/35b264d03638db9f4ce671b711558bf7ff0f80d5

[[ -n "${CODE_PAYLOAD_B64:-}" ]] || { echo 'CODE_PAYLOAD_B64 is required' >&2; exit 2; }
[[ ! -e "$TARGET" ]] || { echo "target already exists: $TARGET" >&2; exit 3; }
[[ ! -e "$STAGE" ]] || { echo "provisioning path already exists: $STAGE" >&2; exit 4; }

mkdir -p "$MODEL" "$SOURCES" "$CODE" "$STAGE/runs"

printf '%s' "$CODE_PAYLOAD_B64" | base64 -d | tar -xzf - -C "$CODE"
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

curl -L --fail --retry 5 --retry-all-errors -o "$SOURCES/train.parquet" "$HF_BOOLQ/data/train-00000-of-00001.parquet?download=true"
curl -L --fail --retry 5 --retry-all-errors -o "$SOURCES/validation.parquet" "$HF_BOOLQ/data/validation-00000-of-00001.parquet?download=true"
printf '%s  %s\n' \
  4f028e992c0bd4df30b9f056f4946b64f5c23028034ff0ed5ea467d8538cc623 "$SOURCES/train.parquet" \
  52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8 "$SOURCES/validation.parquet" | sha256sum -c -

python3 -c 'import pyarrow.parquet' 2>/dev/null || pip install --no-cache-dir pyarrow
export STAGE_ROOT="$STAGE"
python3 - <<'PY'
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

root = Path(os.environ["STAGE_ROOT"])
output = root / "data" / "boolq_35b264d_all_12697.jsonl"
with output.open("w", encoding="utf-8", newline="") as handle:
    for split in ("train", "validation"):
        rows = pq.read_table(root / "sources" / f"{split}.parquet").to_pylist()
        for source_index, row in enumerate(rows):
            record = {
                "answer": bool(row["answer"]),
                "passage": str(row["passage"]),
                "question": str(row["question"]),
                "source_index": source_index,
                "source_split": split,
            }
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
PY
[[ $(stat -c '%s' "$DATA") == 8835655 ]] || { echo 'dataset byte count mismatch' >&2; exit 5; }
printf '%s  %s\n' 13c2f4143ae320a0191c6de5be919248a20c15515f58c6deb7d3732068f2d31a "$DATA" | sha256sum -c -

small_files=(
  .gitattributes LICENSE README.md config.json generation_config.json merges.txt
  model.safetensors.index.json tokenizer.json tokenizer_config.json vocab.json
)
for filename in "${small_files[@]}"; do
  curl -L --fail --retry 5 --retry-all-errors -C - -o "$MODEL/$filename" "$HF_MODEL/$filename?download=true"
done

shards=(
  model-00001-of-00004.safetensors
  model-00002-of-00004.safetensors
  model-00003-of-00004.safetensors
  model-00004-of-00004.safetensors
)
pids=()
for filename in "${shards[@]}"; do
  curl -L --fail --retry 5 --retry-all-errors -C - -o "$MODEL/$filename" "$HF_MODEL/$filename?download=true" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

[[ $(find "$MODEL" -mindepth 1 -maxdepth 1 -type f | wc -l) == 14 ]] || { echo 'model file count mismatch' >&2; exit 6; }
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

python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

root = Path(os.environ["STAGE_ROOT"])
record = {
    "schema": "a11-authoritative-tidalfs-staging-v1",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "dataset_sha256": "13c2f4143ae320a0191c6de5be919248a20c15515f58c6deb7d3732068f2d31a",
    "model_snapshot_manifest_sha256": "3ee6c9510b7e50bfcd46d6df33cafa3e2019f13a6a09bf1d2f9e80cdfe1164e8",
    "protocol_sha256": "1a6593a97b67509d0afc954bef653924f7f9cdc055592d0ba75b08987b71cb31",
    "runner_sha256": "f36e9ee066f7c37d455490b74f78056cb987ed382f5d8e7e35c0ecbb00f75a29",
    "launcher_sha256": "8faea7e75cf716d8fdb5edf0301ddfbc5bc7b49b7d64152097dfe1661715165e",
}
(root / "STAGING_SUCCESS.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY

sync
mv "$STAGE" "$TARGET"
printf 'A11_TIDALFS_STAGING_SUCCESS target=%s\n' "$TARGET"
sha256sum "$TARGET/STAGING_SUCCESS.json" "$TARGET/data/boolq_35b264d_all_12697.jsonl" "$TARGET/code/protocol.json" "$TARGET/code/run_boolq_ordered.py" "$TARGET/code/launch_boolq_ordered_8gpu.sh"
