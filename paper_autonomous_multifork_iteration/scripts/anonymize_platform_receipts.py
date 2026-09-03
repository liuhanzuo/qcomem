#!/usr/bin/env python3
"""Produce anonymous copies of platform receipts for the submission supplement.

The R44 panel's compliance review found that
``evidence/qcomem_mixed_validation_60item_20260812d/platform_receipt.json``
carries a corporate registry hostname, an internal scheduler hostname, cluster
and queue identifiers, internal mount paths, and the submitting account name in
cleartext.  Scrubbing the manuscript alone accomplishes nothing if that package
ships as the anonymous supplement.

This script never edits an evidence file in place.  It writes a sibling
``*.anon.json`` next to each receipt so the original stays available for
author-side provenance while only the anonymised copy is packaged.

The mapping is deterministic and consistent: the same entity always receives the
same alias, so cross-references between receipts survive scrubbing and a reviewer
can still tell that two runs shared a queue without learning which queue.  The
scientific content of a receipt -- timings, status, worker count, resource
shape -- is deliberately preserved.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# Ordered: longer, more specific patterns first so a path is rewritten before
# the account name inside it is matched on its own.
SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    (r"/mnt/[A-Za-z0-9_.\-]+/dataset/[A-Za-z0-9_.\-]+/user/[A-Za-z0-9_.\-]+", "<SHARED_FS_ROOT>"),
    (r"/mnt/[A-Za-z0-9_.\-]+", "<SHARED_FS_ROOT>"),
    (r"https?://[A-Za-z0-9_.\-]+\.[A-Za-z]{2,}(?:/[^\s\"']*)?", "<PLATFORM_URL>"),
    (r"[A-Za-z0-9_\-]+\.(?:devops|int)\.[A-Za-z0-9_\-]+\.com", "<INTERNAL_HOST>"),
    (r"qs-\d+-\d+-[A-Za-z0-9\-]+", "<POD>"),
    (r"\bliuhanzuo\b", "<AUTHOR>"),
)

#: identifier-valued keys replaced by a stable alias rather than a regex rewrite
ID_KEYS = (
    "job_id",
    "trial_id",
    "queue_id",
    "cloud_id",
    "cluster_id",
    "resource_package_id",
)

#: keys whose value is a human-readable internal name
NAME_KEYS = ("queue_name", "name", "image", "web_ui_url")


def scrub_text(value: str) -> str:
    for pattern, replacement in SUBSTITUTIONS:
        value = re.sub(pattern, replacement, value)
    return value


def alias_for(key: str, value: Any, registry: dict[tuple[str, str], str]) -> str:
    token = (key, str(value))
    if token not in registry:
        registry[token] = f"<{key.upper()}_{len(registry) + 1}>"
    return registry[token]


def scrub(node: Any, registry: dict[tuple[str, str], str], key: str | None = None) -> Any:
    if isinstance(node, dict):
        return {k: scrub(v, registry, k) for k, v in node.items()}
    if isinstance(node, list):
        return [scrub(item, registry, key) for item in node]
    if key in ID_KEYS and node is not None:
        return alias_for(key, node, registry)
    if isinstance(node, str):
        scrubbed = scrub_text(node)
        if key in NAME_KEYS and scrubbed == node and node:
            return alias_for(key, node, registry)
        return scrubbed
    return node


def residual_leaks(text: str) -> list[str]:
    """Patterns that must not survive in an anonymised receipt."""
    checks = {
        "account name": r"\bliuhanzuo\b",
        "corporate domain": r"xiaohongshu",
        "internal mount": r"/mnt/tidal",
        "bare pod name": r"qs-\d+-\d+",
    }
    return [name for name, pattern in checks.items() if re.search(pattern, text)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path("evidence")],
        help="directories to scan",
    )
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=["*platform_receipt*.json"],
        help=(
            "filename globs to anonymise.  The default covers platform receipts "
            "only; pass the evidence maps and the experiment registry explicitly "
            "when building a submission bundle, because those carry author paths "
            "too and a receipt-only scan will pass while the bundle still leaks."
        ),
    )
    parser.add_argument("--suffix", default=".anon.json")
    args = parser.parse_args()

    receipts: list[Path] = []
    for root in args.roots:
        for pattern in args.patterns:
            receipts.extend(
                p
                for p in sorted(root.rglob(pattern))
                if not p.name.endswith(args.suffix)
                and ".anon" not in p.suffixes
            )
    receipts = sorted(set(receipts))
    if not receipts:
        raise SystemExit("no platform receipts found")

    registry: dict[tuple[str, str], str] = {}
    failures: list[str] = []
    for path in receipts:
        if path.suffix.lower() == ".json":
            original = json.loads(path.read_text())
            rendered = json.dumps(scrub(original, registry), ensure_ascii=False, indent=2) + "\n"
        else:
            # TSV and other line-oriented evidence files: scrub textually, which
            # preserves column structure because every substitution is in-place.
            rendered = "\n".join(scrub_text(line) for line in path.read_text().split("\n"))
        leaks = residual_leaks(rendered)
        target = path.with_suffix("")
        target = target.with_name(target.name + args.suffix)
        target.write_text(rendered)
        status = "OK" if not leaks else f"LEAKS: {', '.join(leaks)}"
        if leaks:
            failures.append(f"{target}: {status}")
        print(f"{path} -> {target}  {status}")

    print(f"\nalias registry ({len(registry)} entries):")
    for (key, value), alias in sorted(registry.items()):
        print(f"  {key}={value!r} -> {alias}")

    if failures:
        raise SystemExit("residual identifiers survived:\n" + "\n".join(failures))
    print("\nall anonymised receipts pass the residual-identifier check")


if __name__ == "__main__":
    main()
