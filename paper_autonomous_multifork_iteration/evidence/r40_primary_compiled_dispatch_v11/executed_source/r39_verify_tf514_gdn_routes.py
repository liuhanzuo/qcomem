#!/usr/bin/env python3
"""Fail closed on the frozen Transformers-5.14.1 Qwen3.5-MoE GDN routes."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import stat
from pathlib import Path


RELATIVE_SOURCE = Path(
    "transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py"
)
EXPECTED_SOURCE_SHA256 = (
    "688d9a8f2830d6729cd2945563f38b710100c086565b97c27c94c96bd9716b9f"
)
REQUIRED_TOP_LEVEL_FUNCTIONS = {
    "torch_causal_conv1d_update",
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
}
REQUIRED_ROUTE_FRAGMENTS = (
    "self.causal_conv1d_update = causal_conv1d_update or torch_causal_conv1d_update",
    "self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule",
    "self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule",
    "if use_precomputed_states and seq_len == 1:",
    "mixed_qkv = self.causal_conv1d_update(",
    "cache_params.update_conv_state(new_conv_state, self.layer_idx)",
    "core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(",
    "core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(",
    "cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)",
)


class RouteVerificationError(RuntimeError):
    """The frozen runtime source or its mutually exclusive route shape drifted."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(runtime_root: Path) -> dict[str, object]:
    root = runtime_root.resolve()
    source = root / RELATIVE_SOURCE
    try:
        mode = source.lstat().st_mode
    except OSError as error:
        raise RouteVerificationError(f"frozen route source unavailable: {error}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise RouteVerificationError("frozen route source must be a regular non-symlink file")
    try:
        source.resolve().relative_to(root)
    except ValueError as error:
        raise RouteVerificationError("frozen route source escapes runtime root") from error
    observed_sha = _sha256(source)
    if observed_sha != EXPECTED_SOURCE_SHA256:
        raise RouteVerificationError(
            f"frozen TF-5.14.1 source SHA drift: {observed_sha}"
        )
    text = source.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(source))
    top_level_functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    if not REQUIRED_TOP_LEVEL_FUNCTIONS.issubset(top_level_functions):
        raise RouteVerificationError("frozen GDN fallback function set drift")
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    gdn_class = classes.get("Qwen3_5MoeGatedDeltaNet")
    if gdn_class is None:
        raise RouteVerificationError("frozen Qwen3.5-MoE GDN class absent")
    method_names = {
        node.name for node in gdn_class.body if isinstance(node, ast.FunctionDef)
    }
    if not {"__init__", "forward"}.issubset(method_names):
        raise RouteVerificationError("frozen GDN init/forward method set drift")
    missing = [fragment for fragment in REQUIRED_ROUTE_FRAGMENTS if fragment not in text]
    if missing:
        raise RouteVerificationError(f"frozen route fragments absent: {missing}")
    if text.count("if use_precomputed_states and seq_len == 1:") != 2:
        raise RouteVerificationError("single-token route gate count drift")
    return {
        "schema_version": "forkaudit-r39-tf514-gdn-route-static-v1",
        "status": "pass",
        "runtime_source_relative_path": RELATIVE_SOURCE.as_posix(),
        "runtime_source_sha256": observed_sha,
        "route_gate_count": 2,
        "multi_token_expected_counts": {
            "chunk_rule": 1,
            "recurrent_rule": 0,
            "functional_conv_rebind": 1,
            "inplace_conv_update": 0,
            "functional_recurrent_rebind": 1,
        },
        "cached_single_token_expected_counts": {
            "chunk_rule": 0,
            "recurrent_rule": 1,
            "functional_conv_rebind": 0,
            "inplace_conv_update": 1,
            "functional_recurrent_rebind": 1,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("static-route output must be absent")
    result = verify(args.runtime_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
