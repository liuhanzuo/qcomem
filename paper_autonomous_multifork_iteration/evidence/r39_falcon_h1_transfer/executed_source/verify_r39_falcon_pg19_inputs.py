#!/usr/bin/env python3
"""Re-tokenize frozen PG-19 objects and verify the Falcon-H1 input windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


FROZEN_INPUT_SHA256 = "d4c8341c74e4b0e2ee0969208d7a3912f86cd1fb4a1f1a906b0243944342fd4c"
TOKENIZER_SHA256 = "605c664925653e3fbf2f35ea063847db441ba5b7a6af04378880409c3ab311fc"
VOCAB_SIZE = 32784


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"


def atomic_write(path: Path, payload: bytes) -> None:
    require(not path.exists(), f"refusing to replace output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify(args: argparse.Namespace) -> dict[str, Any]:
    require(sha256_file(args.frozen) == FROZEN_INPUT_SHA256, "frozen input manifest drift")
    require(sha256_file(args.tokenizer_json) == TOKENIZER_SHA256, "Falcon tokenizer drift")
    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    import tokenizers
    from tokenizers import Tokenizer

    require(tokenizers.__version__ == "0.22.2", "tokenizers derivation version drift")
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    require(tokenizer.get_vocab_size(with_added_tokens=True) == 32768, "tokenizer vocabulary drift")
    receipts = []
    for row in frozen["rows"]:
        raw_path = args.pg19_root / row["source_object"]
        require(raw_path.is_file() and not raw_path.is_symlink(), "PG-19 source absent")
        raw = raw_path.read_bytes()
        require(len(raw) == row["source_bytes"], "PG-19 source size drift")
        require(hashlib.sha256(raw).hexdigest() == row["source_sha256"], "PG-19 source hash drift")
        text = raw.decode("utf-8", errors="strict")
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        require(len(ids) == row["full_token_count"], "Falcon full token count drift")
        require(ids[197:261] == row["document_token_ids"], "Falcon document window drift")
        require(ids[477:485] == row["queries"][0]["token_ids"], "Falcon query-0 window drift")
        require(ids[509:517] == row["queries"][1]["token_ids"], "Falcon query-1 window drift")
        selected = [*row["document_token_ids"], *row["queries"][0]["token_ids"], *row["queries"][1]["token_ids"]]
        require(all(0 <= value < VOCAB_SIZE for value in selected), "selected Falcon token OOV")
        receipts.append(
            {
                "rank": row["rank"],
                "source_id": row["source_id"],
                "source_sha256": row["source_sha256"],
                "full_token_count": len(ids),
                "selected_token_count": len(selected),
                "selected_max_token_id": max(selected),
                "verified": True,
            }
        )
    return {
        "schema_version": "r39-falcon-h1-pg19-input-verification-v1",
        "frozen_inputs_sha256": FROZEN_INPUT_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "tokenizers_version": tokenizers.__version__,
        "rank_count": len(receipts),
        "rows": receipts,
        "verified": len(receipts) == 8,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--pg19-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_write(args.output, canonical_bytes(verify(args)))


if __name__ == "__main__":
    main()
