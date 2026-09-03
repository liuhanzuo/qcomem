from __future__ import annotations

"""Freeze the historical fair-v2 FP32 diagnostic context for ForkAudit RR2.

The builder reads the eight archived PG-19 fair-v2 gate shards, validates the
exact diagnostic coordinate and metric types, and emits a path-independent
manifest.  The output contains no clock, host path, inode, or mtime, so a
relocated byte-identical archive produces byte-identical output.

This is a historical-evidence builder, not an experiment runner.  The RR2
relative-L2 threshold was fixed independently at 0.005 before RR2.  Archived
``vllm_reuse_vs_fp32_dense`` diagnostics neither select nor tune that value;
they provide only a contextual check that the pre-fixed threshold is at least
twice the largest historical diagnostic.
"""

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "qcomem.forkaudit.fp32-prior-context-manifest.v1"
ARCHIVE_ID = "gpu-qwen35-vllm-paged-fair-v2-20260814c"
FAIR_PROTOCOL = "same-vllm-unified-attention-q16-single-request-v2"
COMPARISON_PATH = (
    "isolated_same_kernel.rows[].backend_compatibility_nonblocking."
    "vllm_reuse_vs_fp32_dense"
)
RAW_SHARD_DIR = "pg19-gate-shards"
RAW_SHARD_NAME = "pg19-fair-v2-shard-{rank}.json"
SCIENTIFIC_LEDGER_NAME = "scientific-artifacts.sha256"
EXPECTED_WORLD_SIZE = 8
EXPECTED_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39)
PRIOR_CONTEXT_DOCUMENT_TOKENS = 1025
PRIOR_CONTEXT_QUERY_TOKENS = 32
RR2_DOCUMENT_TOKENS = 4095
PREREGISTERED_THRESHOLD = 0.005
CONTEXT_MARGIN_MULTIPLIER = 2.0

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_RE = re.compile(r"^train/[0-9]+\.txt$")
_LEDGER_RE = re.compile(r"^([0-9a-f]{64})  (.+)$")
_METRIC_KEYS = frozenset(
    {"bitwise_exact", "finite", "max_abs", "mean_abs", "relative_l2"}
)


class CalibrationManifestError(RuntimeError):
    """The archived historical evidence violates the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationManifestError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
    except (TypeError, ValueError) as exc:
        raise CalibrationManifestError("manifest is not finite canonical JSON") from exc
    return text.encode("utf-8") + b"\n"


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CalibrationManifestError(f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise CalibrationManifestError(f"JSON contains non-finite constant {value}")


def _strict_json_bytes(payload: bytes, *, label: str) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CalibrationManifestError(f"{label} is not UTF-8 JSON") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_json_constant,
        )
    except CalibrationManifestError:
        raise
    except json.JSONDecodeError as exc:
        raise CalibrationManifestError(f"{label} is not valid JSON") from exc


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    _require(type(value) is dict, f"{label} must be an object")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    _require(type(value) is list, f"{label} must be an array")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"{label} must be an integer >= {minimum}")
    return value


def _finite_float(value: Any, *, label: str, minimum: float = 0.0) -> float:
    _require(type(value) is float, f"{label} must be a JSON floating-point number")
    _require(math.isfinite(value) and value >= minimum, f"{label} must be finite and >= {minimum}")
    return value


def _string(value: Any, *, label: str) -> str:
    _require(type(value) is str and value != "", f"{label} must be a non-empty string")
    return value


def _sha256(value: Any, *, label: str) -> str:
    digest = _string(value, label=label)
    _require(_SHA256_RE.fullmatch(digest) is not None, f"{label} must be a lowercase SHA-256")
    return digest


def _boolean(value: Any, *, label: str) -> bool:
    _require(type(value) is bool, f"{label} must be a boolean")
    return value


def _read_scientific_ledger(
    path: Path,
    *,
    expected_shards: set[str],
) -> tuple[dict[str, str], dict[str, Any]]:
    _require(path.is_file(), f"{SCIENTIFIC_LEDGER_NAME} is missing")
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CalibrationManifestError("scientific artifact ledger is not UTF-8") from exc
    _require(text.endswith("\n"), "scientific artifact ledger must end with one newline")

    shard_digests: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = _LEDGER_RE.fullmatch(line)
        _require(match is not None, f"scientific artifact ledger line {line_number} is malformed")
        assert match is not None
        digest, raw_name = match.groups()
        suffix = f"/{RAW_SHARD_DIR}/"
        if suffix not in raw_name:
            continue
        name = raw_name.rsplit("/", 1)[-1]
        _require(name in expected_shards, f"scientific artifact ledger names unexpected shard {name!r}")
        _require(name not in shard_digests, f"scientific artifact ledger repeats shard {name}")
        shard_digests[name] = digest
    _require(
        set(shard_digests) == expected_shards,
        "scientific artifact ledger does not bind exactly the eight expected shards",
    )
    normalized_rows = [
        {
            "logical_name": f"{RAW_SHARD_DIR}/{name}",
            "sha256": shard_digests[name],
        }
        for name in sorted(shard_digests)
    ]
    return shard_digests, {
        "logical_name": SCIENTIFIC_LEDGER_NAME,
        "verified_raw_shard_entries": len(normalized_rows),
        "normalized_raw_shard_entries_sha256": _sha256_bytes(
            _canonical_json_bytes(normalized_rows)
        ),
        "source_path_strings_serialized_or_hashed": False,
    }


def _metric_payload(value: Any, *, label: str) -> dict[str, Any]:
    metric = _mapping(value, label=label)
    _require(set(metric) == _METRIC_KEYS, f"{label} has an unexpected metric schema")
    bitwise_exact = _boolean(metric["bitwise_exact"], label=f"{label}.bitwise_exact")
    finite = _boolean(metric["finite"], label=f"{label}.finite")
    _require(finite, f"{label} is marked non-finite")
    max_abs = _finite_float(metric["max_abs"], label=f"{label}.max_abs")
    mean_abs = _finite_float(metric["mean_abs"], label=f"{label}.mean_abs")
    relative_l2 = _finite_float(metric["relative_l2"], label=f"{label}.relative_l2")
    return {
        "bitwise_exact": bitwise_exact,
        "finite": finite,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "relative_l2": relative_l2,
    }


def _diagnostics_from_shard(
    shard: Any,
    *,
    expected_rank: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    root = _mapping(shard, label=f"rank {expected_rank} shard")
    rank = _integer(root.get("rank"), label="shard.rank")
    _require(rank == expected_rank, f"shard rank {rank} does not match filename rank {expected_rank}")
    _require(
        _integer(root.get("world_size"), label="shard.world_size", minimum=1) == EXPECTED_WORLD_SIZE,
        "historical fair-v2 cohort world size drifted",
    )
    _require(root.get("status") == "completed_pg19_fair_v2_gate_shard", "shard status drifted")
    _require(root.get("passed") is True, "archived fair-v2 shard is not marked passed")
    _require(root.get("fair_protocol") == FAIR_PROTOCOL, "fair-v2 protocol drifted")
    _require(root.get("quantization") == "Q16", "historical fair-v2 cohort is not Q16")
    _require(root.get("single_request_only") is True, "historical fair-v2 cohort is not single-request")
    windows_sha256 = _sha256(root.get("windows_sha256"), label="shard.windows_sha256")

    static = _mapping(root.get("static"), label="shard.static")
    pg19_data_sha256 = _sha256(static.get("pg19_data_sha256"), label="static.pg19_data_sha256")
    pg19_manifest_sha256 = _sha256(
        static.get("pg19_manifest_sha256"), label="static.pg19_manifest_sha256"
    )
    _require(
        _sha256(static.get("pg19_windows_sha256"), label="static.pg19_windows_sha256")
        == windows_sha256,
        "static and shard window digests disagree",
    )
    protocol_config = _mapping(static.get("protocol_config"), label="static.protocol_config")
    _require(
        protocol_config.get("pg19_document_tokens") == PRIOR_CONTEXT_DOCUMENT_TOKENS,
        "historical cohort document length drifted",
    )
    _require(
        protocol_config.get("pg19_query_tokens") == PRIOR_CONTEXT_QUERY_TOKENS,
        "historical cohort query length drifted",
    )

    rows = _sequence(root.get("rows"), label="shard.rows")
    _require(len(rows) == 1, "each fair-v2 rank shard must contain exactly one PG-19 window")
    window = _mapping(rows[0], label="shard.rows[0]")
    window_index = _integer(window.get("window_index"), label="window.window_index")
    _require(window_index == rank, "historical window index must equal its frozen rank")
    source_object = _string(window.get("source_object"), label="window.source_object")
    _require(_SOURCE_RE.fullmatch(source_object) is not None, "historical source is not PG-19 train")
    document_tokens = _integer(window.get("document_tokens"), label="window.document_tokens", minimum=1)
    query_tokens = _integer(window.get("query_tokens"), label="window.query_tokens", minimum=1)
    _require(document_tokens == PRIOR_CONTEXT_DOCUMENT_TOKENS, "historical document length drifted")
    _require(query_tokens == PRIOR_CONTEXT_QUERY_TOKENS, "historical query length drifted")

    isolated = _mapping(window.get("isolated_same_kernel"), label="window.isolated_same_kernel")
    _require(isolated.get("passed") is True, "isolated same-kernel gate is not marked passed")
    _require(isolated.get("fair_protocol") == FAIR_PROTOCOL, "isolated protocol drifted")
    _require(
        _integer(isolated.get("layer_count"), label="isolated.layer_count", minimum=1)
        == len(EXPECTED_LAYERS),
        "isolated layer count drifted",
    )
    layer_indices = _sequence(isolated.get("layer_indices"), label="isolated.layer_indices")
    _require(
        all(type(item) is int for item in layer_indices)
        and tuple(layer_indices) == EXPECTED_LAYERS,
        "isolated full-attention layer list drifted",
    )
    layer_rows = _sequence(isolated.get("rows"), label="isolated.rows")
    _require(len(layer_rows) == len(EXPECTED_LAYERS), "isolated diagnostic row count drifted")

    diagnostics: list[dict[str, Any]] = []
    observed_layers: list[int] = []
    for row_index, raw_layer in enumerate(layer_rows):
        label = f"rank {rank} layer row {row_index}"
        layer = _mapping(raw_layer, label=label)
        layer_idx = _integer(layer.get("layer_idx"), label=f"{label}.layer_idx")
        observed_layers.append(layer_idx)
        _require(layer.get("passed") is True, f"{label} is not marked passed")
        query_sha256 = _sha256(layer.get("query_sha256"), label=f"{label}.query_sha256")
        position_ids_sha256 = _sha256(
            layer.get("position_ids_sha256"), label=f"{label}.position_ids_sha256"
        )
        mask_sha256 = _sha256(layer.get("mask_sha256"), label=f"{label}.mask_sha256")
        scaling = _finite_float(layer.get("scaling"), label=f"{label}.scaling", minimum=0.0)
        _require(scaling > 0.0, f"{label}.scaling must be positive")
        _require(layer.get("same_scale") is True, f"{label} did not bind equal scaling")
        _require(layer.get("same_post_rope_query_object") is True, f"{label} did not bind the query")

        compatibility = _mapping(
            layer.get("backend_compatibility_nonblocking"),
            label=f"{label}.backend_compatibility_nonblocking",
        )
        reuse = _metric_payload(
            compatibility.get("vllm_reuse_vs_fp32_dense"),
            label=f"{label}.vllm_reuse_vs_fp32_dense",
        )
        fresh = _metric_payload(
            compatibility.get("vllm_fresh_vs_fp32_dense"),
            label=f"{label}.vllm_fresh_vs_fp32_dense",
        )
        _require(fresh == reuse, f"{label} fresh/reuse FP32 diagnostics disagree")
        diagnostics.append(
            {
                "rank": rank,
                "window_index": window_index,
                "source_object": source_object,
                "document_tokens": document_tokens,
                "query_tokens": query_tokens,
                "layer_idx": layer_idx,
                "query_sha256": query_sha256,
                "position_ids_sha256": position_ids_sha256,
                "mask_sha256": mask_sha256,
                "scaling": scaling,
                "comparison": "vllm_reuse_vs_fp32_dense",
                "metrics": reuse,
            }
        )
    _require(tuple(observed_layers) == EXPECTED_LAYERS, "diagnostic layer ordering drifted")
    return diagnostics, {
        "windows_sha256": windows_sha256,
        "pg19_data_sha256": pg19_data_sha256,
        "pg19_manifest_sha256": pg19_manifest_sha256,
    }


def _coordinate_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["source_object"],
        row["window_index"],
        row["document_tokens"],
        row["query_tokens"],
        row["query_sha256"],
        row["layer_idx"],
        row["scaling"],
    )


def build_manifest(
    archive_root: Path,
    *,
    rr2_document_tokens: int = RR2_DOCUMENT_TOKENS,
    preregistered_threshold: float = PREREGISTERED_THRESHOLD,
) -> dict[str, Any]:
    """Validate an archive and return its path-free prior-context manifest."""

    _require(
        type(rr2_document_tokens) is int and rr2_document_tokens > 0,
        "RR2 document length must be a positive integer",
    )
    _require(
        type(preregistered_threshold) is float
        and math.isfinite(preregistered_threshold)
        and preregistered_threshold > 0.0,
        "preregistered threshold must be a positive finite float",
    )
    _require(
        preregistered_threshold == PREREGISTERED_THRESHOLD,
        "the RR2 preregistered relative-L2 threshold must remain exactly 0.005",
    )

    expected_names = {RAW_SHARD_NAME.format(rank=rank) for rank in range(EXPECTED_WORLD_SIZE)}
    shard_dir = archive_root / RAW_SHARD_DIR
    _require(shard_dir.is_dir(), f"{RAW_SHARD_DIR} is missing")
    actual_names = {path.name for path in shard_dir.glob("*.json") if path.is_file()}
    _require(actual_names == expected_names, "archive must contain exactly the eight expected raw shard JSON files")
    ledger_digests, ledger_receipt = _read_scientific_ledger(
        archive_root / SCIENTIFIC_LEDGER_NAME,
        expected_shards=expected_names,
    )

    raw_shards: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    identities: list[dict[str, str]] = []
    for rank in range(EXPECTED_WORLD_SIZE):
        name = RAW_SHARD_NAME.format(rank=rank)
        payload = (shard_dir / name).read_bytes()
        digest = _sha256_bytes(payload)
        _require(ledger_digests[name] == digest, f"scientific ledger digest mismatch for {name}")
        raw_shards.append(
            {
                "rank": rank,
                "logical_name": f"{RAW_SHARD_DIR}/{name}",
                "sha256": digest,
                "bytes": len(payload),
            }
        )
        shard = _strict_json_bytes(payload, label=name)
        rows, identity = _diagnostics_from_shard(shard, expected_rank=rank)
        diagnostics.extend(rows)
        identities.append(identity)

    _require(len(diagnostics) == EXPECTED_WORLD_SIZE * len(EXPECTED_LAYERS), "diagnostic count drifted")
    coordinate_keys = [_coordinate_key(row) for row in diagnostics]
    _require(len(coordinate_keys) == len(set(coordinate_keys)), "historical diagnostic coordinate is duplicated")
    query_digests = [row["query_sha256"] for row in diagnostics]
    _require(len(query_digests) == len(set(query_digests)), "historical query digest is duplicated")
    sources = [row["source_object"] for row in diagnostics[:: len(EXPECTED_LAYERS)]]
    _require(len(sources) == len(set(sources)), "historical source object is duplicated")

    identity_keys = ("windows_sha256", "pg19_data_sha256", "pg19_manifest_sha256")
    archive_identity: dict[str, str] = {}
    for key in identity_keys:
        values = {row[key] for row in identities}
        _require(len(values) == 1, f"archived shards disagree on {key}")
        archive_identity[key] = next(iter(values))

    maximum_row = max(
        diagnostics,
        key=lambda row: (row["metrics"]["relative_l2"], row["rank"], row["layer_idx"]),
    )
    maximum_observed = maximum_row["metrics"]["relative_l2"]
    required_margin_boundary = CONTEXT_MARGIN_MULTIPLIER * maximum_observed
    _require(math.isfinite(required_margin_boundary), "historical margin boundary is non-finite")
    _require(
        preregistered_threshold >= required_margin_boundary,
        "fixed preregistered threshold lacks the required two-times historical margin; do not run RR2",
    )

    prior_document_lengths = sorted({row["document_tokens"] for row in diagnostics})
    _require(
        rr2_document_tokens not in prior_document_lengths,
        "RR2 document length overlaps the historical context coordinate",
    )
    coordinate_projection = [
        {
            key: row[key]
            for key in (
                "source_object",
                "window_index",
                "document_tokens",
                "query_tokens",
                "query_sha256",
                "layer_idx",
                "scaling",
            )
        }
        for row in diagnostics
    ]
    coordinates_sha256 = _sha256_bytes(_canonical_json_bytes(coordinate_projection))

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": (
            "Freeze the historical pre-RR2 fair-v2 vLLM-versus-IEEE-FP32 "
            "diagnostics for a contextual margin check against an independently "
            "pre-fixed ForkAudit relative-L2 threshold."
        ),
        "archive": {
            "archive_id": ARCHIVE_ID,
            "fair_protocol": FAIR_PROTOCOL,
            "quantization": "Q16",
            "single_request_only": True,
            **archive_identity,
            "scientific_artifact_ledger": ledger_receipt,
            "raw_shards": raw_shards,
        },
        "diagnostic_definition": {
            "comparison_path": COMPARISON_PATH,
            "candidate": "vLLM unified attention over Q16 BF16 K/V",
            "reference": "dense FP32 attention",
            "backend_compatibility_was_nonblocking_in_prior_run": True,
            "fresh_and_reuse_metric_payloads_required_exactly_equal": True,
            "diagnostic_count": len(diagnostics),
            "role_in_rr2_threshold_choice": "contextual_validation_only",
            "selected_or_tuned_rr2_threshold": False,
        },
        "diagnostics": diagnostics,
        "pre_fixed_threshold_margin_check": {
            "threshold_fixed_before_rr2": True,
            "threshold_fixed_independently_of_prior_rows": True,
            "prior_rows_selected_or_tuned_threshold": False,
            "prior_archive_role": "contextual_validation_only",
            "fixed_preregistered_threshold": preregistered_threshold,
            "required_context_margin_multiplier": CONTEXT_MARGIN_MULTIPLIER,
            "maximum_observed_prior_relative_l2": maximum_observed,
            "required_margin_boundary_from_prior_maximum": required_margin_boundary,
            "fixed_threshold_to_prior_maximum_ratio": (
                preregistered_threshold / maximum_observed
            ),
            "fixed_threshold_at_least_twice_prior_maximum": True,
            "maximum_coordinate": {
                "rank": maximum_row["rank"],
                "window_index": maximum_row["window_index"],
                "source_object": maximum_row["source_object"],
                "document_tokens": maximum_row["document_tokens"],
                "query_sha256": maximum_row["query_sha256"],
                "layer_idx": maximum_row["layer_idx"],
                "scaling": maximum_row["scaling"],
            },
        },
        "rr2_disjointness_from_prior_context": {
            "ordered_fields": [
                "source_object",
                "window_index",
                "document_tokens",
                "query_tokens",
                "query_sha256",
                "layer_idx",
                "scaling",
            ],
            "coordinate_equality": "exact equality of every ordered field",
            "prior_coordinate_count": len(coordinate_projection),
            "prior_coordinates_sha256": coordinates_sha256,
            "prior_context_document_token_values": prior_document_lengths,
            "rr2_preregistered_document_tokens": rr2_document_tokens,
            "document_length_is_required_coordinate_component": True,
            "document_length_disjoint": True,
            "rr2_coordinate_disjointness_rule": (
                "An RR2 oracle diagnostic is distinct from the historical context only "
                "if its complete ordered coordinate is absent from diagnostics above. "
                "The preregistered RR2 document length 4095 is disjoint from the prior "
                "value 1025, so no RR2 coordinate can equal a prior-context coordinate."
            ),
            "prior_document_start_token_available": False,
            "prior_document_start_token_note": (
                "The archived fair-v2 shards do not serialize document_start_token; "
                "coordinate disjointness here therefore relies on the fully observed "
                "coordinate above, including the disjoint document length, and makes no "
                "claim about start-token disjointness."
            ),
        },
        "path_independence": {
            "absolute_paths_serialized": False,
            "timestamps_serialized": False,
            "filesystem_metadata_serialized": False,
            "raw_artifacts_named_by_logical_relative_name": True,
        },
    }
    serialized = _canonical_json_bytes(result)
    lowered = serialized.lower()
    _require(b"/users/" not in lowered and b"/mnt/" not in lowered, "manifest leaked a host path")
    return result


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    payload = _canonical_json_bytes(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the path-independent ForkAudit prior-FP32 context manifest"
    )
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest = build_manifest(args.archive_root)
        write_manifest(args.output, manifest)
    except CalibrationManifestError as exc:
        raise SystemExit(f"prior-context manifest rejected: {exc}") from exc
    payload = args.output.read_bytes()
    print(
        json.dumps(
            {
                "status": "forkaudit_fp32_prior_context_manifest_built",
                "output_sha256": _sha256_bytes(payload),
                "output_bytes": len(payload),
                "diagnostic_count": manifest["diagnostic_definition"]["diagnostic_count"],
                "maximum_observed_relative_l2": manifest["pre_fixed_threshold_margin_check"][
                    "maximum_observed_prior_relative_l2"
                ],
                "fixed_preregistered_threshold": manifest["pre_fixed_threshold_margin_check"][
                    "fixed_preregistered_threshold"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
