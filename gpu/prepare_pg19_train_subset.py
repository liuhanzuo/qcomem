from __future__ import annotations

import argparse
import base64
import hashlib
import json
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


BUCKET = "deepmind-gutenberg"
PREFIX = "train/"
LIST_ENDPOINT = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
OBJECT_ENDPOINT = f"https://storage.googleapis.com/{BUCKET}"


def list_train_objects(limit: int) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit must be positive")
    selected: list[dict[str, Any]] = []
    page_token: str | None = None
    while len(selected) < limit:
        query = {"prefix": PREFIX, "maxResults": min(1000, limit * 2)}
        if page_token is not None:
            query["pageToken"] = page_token
        url = LIST_ENDPOINT + "?" + urllib.parse.urlencode(query)
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = json.load(response)
        for item in payload.get("items", []):
            if item["name"].endswith(".txt"):
                selected.append(item)
                if len(selected) == limit:
                    break
        page_token = payload.get("nextPageToken")
        if page_token is None:
            break
    if len(selected) != limit:
        raise RuntimeError(f"requested {limit} PG-19 train books, found {len(selected)}")
    return selected


def download_object(item: dict[str, Any]) -> bytes:
    encoded_name = urllib.parse.quote(item["name"], safe="/")
    with urllib.request.urlopen(f"{OBJECT_ENDPOINT}/{encoded_name}", timeout=120) as response:
        data = response.read()
    expected_size = int(item["size"])
    if len(data) != expected_size:
        raise RuntimeError(
            f"{item['name']}: expected {expected_size} bytes, downloaded {len(data)}"
        )
    expected_md5 = item.get("md5Hash")
    actual_md5 = base64.b64encode(hashlib.md5(data).digest()).decode()
    if expected_md5 is not None and actual_md5 != expected_md5:
        raise RuntimeError(f"{item['name']}: GCS MD5 mismatch")
    return data


def prepare(output: Path, manifest: Path, *, books: int) -> dict[str, Any]:
    objects = list_train_objects(books)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        for index, item in enumerate(objects):
            data = download_object(item)
            text = data.decode("utf-8")
            row = {
                "id": Path(item["name"]).stem,
                "text": text,
                "_source_bucket": BUCKET,
                "_source_object": item["name"],
                "_source_generation": item.get("generation"),
                "_source_md5_base64": item.get("md5Hash"),
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            records.append(
                {
                    "index": index,
                    "name": item["name"],
                    "size": int(item["size"]),
                    "generation": item.get("generation"),
                    "md5_base64": item.get("md5Hash"),
                }
            )
            print(f"downloaded {index + 1}/{books}: {item['name']}", flush=True)
    temporary.replace(output)
    payload = {
        "format": "qcomem_pg19_train_smoke_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_repository": "https://github.com/google-deepmind/pg19",
        "bucket": BUCKET,
        "prefix": PREFIX,
        "selection": "first N train/*.txt objects in GCS lexicographic listing order",
        "books": books,
        "objects": records,
        "jsonl": str(output),
        "jsonl_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "test_or_validation_objects_used": False,
        "purpose": "functional LoRA smoke; not the formal full-corpus training set",
    }
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a provenance-locked PG-19 train-only LoRA smoke subset"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--books", type=int, default=64)
    args = parser.parse_args()
    result = prepare(args.output, args.manifest, books=args.books)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
