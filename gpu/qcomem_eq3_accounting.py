"""Torch-free byte accounting for the Eq. 3 group packer.

This module deliberately imports nothing from ``torch``.  Every function here
is pure integer/float arithmetic over plain dictionaries so that the byte
accounting behind Table 1 and Table 2 can be unit tested on a laptop with no
CUDA and no Transformers install, and so that the aggregator can validate an
archived shard without a GPU environment.

Format contract (Eq. 3 of the manuscript, Figure 2c):

* values are grouped into runs of ``group_size`` (64 by default) along the
  flattened element order (cache leaves) or the hidden dimension (the document
  boundary residual);
* each group stores ``group_size`` unsigned ``b``-bit codes, packed little-end
  first inside each byte by ``qcomem_torch._pack_unsigned``;
* each group additionally stores one BF16 scale and one BF16 bias, i.e. 4 bytes
  of metadata.

One group therefore occupies ``group_size * b / 8 + 4`` bytes and a component of
``n`` elements occupies ``ceil(n / group_size) * (group_size * b / 8 + 4)``
bytes.  ``assert_eq3_component_identity`` is the assertion required by the
revision protocol; it is applied to every quantized component of every emitted
row.

``b = 16`` is *not* a packed width in this format.  ``quantize_residual`` casts
to BF16 and stores the values verbatim with no group metadata, and
``quantize_tensor`` clones the source buffer *in its source dtype* with no group
metadata.  The second of those is the known dtype-inconsistency defect: Qwen3.5
GatedDeltaNet ``recurrent_states`` are FP32, so a ``bits=16`` "reference" cache
silently counts 4 bytes per element for them.  This module never asserts the
Eq. 3 identity at ``b = 16``; instead it emits, for every component, both

* ``native_reference_nbytes`` = ``elements * native_itemsize``  and
* ``bf16_reference_nbytes``   = ``elements * 2``

so that a native-dtype ratio and an all-BF16 ratio can both be computed
downstream without guessing which dtype a component was stored in.
"""

from __future__ import annotations

import random
import statistics
from typing import Any, Iterable, Mapping, Sequence


GROUP_SIZE = 64
BF16_ITEMSIZE = 2
#: one BF16 scale plus one BF16 bias per group
METADATA_NBYTES_PER_GROUP = 2 * BF16_ITEMSIZE
#: widths that are genuinely bit-packed by ``_pack_unsigned``
PACKED_BITS = (2, 4, 8)
#: every width the packer accepts; 16 is a verbatim copy, not a packed width
SUPPORTED_BITS = (2, 4, 8, 16)

STATE_TYPES = (
    "document_residual",
    "attention_key",
    "attention_value",
    "conv_state",
    "recurrent_state",
    "other",
)


class Eq3IdentityError(AssertionError):
    """Raised when a stored component does not match the Eq. 3 byte identity."""


# ---------------------------------------------------------------------------
# core format arithmetic
# ---------------------------------------------------------------------------


def group_count(elements: int, group_size: int = GROUP_SIZE) -> int:
    """``ceil(elements / group_size)`` in exact integer arithmetic."""

    if elements < 0:
        raise ValueError("elements must not be negative")
    if group_size < 1:
        raise ValueError("group_size must be positive")
    return -(-int(elements) // int(group_size))


def eq3_group_nbytes(bits: int, group_size: int = GROUP_SIZE) -> int:
    """Bytes occupied by one packed group: ``group_size * bits / 8 + 4``."""

    if bits not in PACKED_BITS:
        raise ValueError(
            f"bits must be one of {PACKED_BITS}; {bits} is not a packed width"
        )
    if group_size < 1:
        raise ValueError("group_size must be positive")
    if (group_size * bits) % 8:
        raise ValueError("group_size * bits must fill whole bytes")
    return (group_size * bits) // 8 + METADATA_NBYTES_PER_GROUP


def eq3_component_nbytes(
    elements: int, bits: int, group_size: int = GROUP_SIZE
) -> int:
    """``ceil(n / g) * (g * b / 8 + 4)`` — the Eq. 3 identity for one component."""

    return group_count(elements, group_size) * eq3_group_nbytes(bits, group_size)


def eq3_code_nbytes(elements: int, bits: int, group_size: int = GROUP_SIZE) -> int:
    """Bytes of packed codes alone, i.e. the identity minus its metadata."""

    return group_count(elements, group_size) * ((group_size * bits) // 8)


def eq3_metadata_nbytes(elements: int, group_size: int = GROUP_SIZE) -> int:
    """Bytes of BF16 scales plus BF16 biases for one component."""

    return group_count(elements, group_size) * METADATA_NBYTES_PER_GROUP


def verbatim_component_nbytes(elements: int, itemsize: int) -> int:
    """Bytes of an unpacked component stored verbatim at ``itemsize``."""

    if itemsize < 1:
        raise ValueError("itemsize must be positive")
    return int(elements) * int(itemsize)


def native_reference_nbytes(elements: int, itemsize: int) -> int:
    """Dense reference in the component's own source dtype."""

    return verbatim_component_nbytes(elements, itemsize)


def bf16_reference_nbytes(elements: int) -> int:
    """Dense reference with every component consistently counted as BF16."""

    return verbatim_component_nbytes(elements, BF16_ITEMSIZE)


def eq3_compression_ratio(bits: int, group_size: int = GROUP_SIZE) -> float:
    """Format ceiling against an all-BF16 reference, ignoring group padding."""

    return (group_size * BF16_ITEMSIZE) / eq3_group_nbytes(bits, group_size)


# ---------------------------------------------------------------------------
# component records
# ---------------------------------------------------------------------------


def component_record(
    *,
    leaf_path: str,
    layer_index: int | None,
    state_type: str,
    elements: int,
    bits: int | None,
    group_size: int,
    code_nbytes: int,
    scale_nbytes: int,
    bias_nbytes: int,
    native_itemsize: int,
    storage_nbytes: int | None = None,
    floating: bool = True,
) -> dict[str, Any]:
    """Build one auditable per-component byte record.

    ``bits is None`` marks a leaf that was never handed to the packer: either a
    leaf of an unquantized exact reference cache, or a non-floating-point cache
    leaf (``floating=False``, e.g. an integer length counter) that
    ``quantize_transformers_cache`` clones verbatim.  Those leaves carry no
    Eq. 3 identity but are still counted, so that the breakdown reconciles
    byte-for-byte against ``PackedCache.nbytes`` / ``cache_nbytes``.

    ``dtype_inconsistent_reference`` marks a floating component that is stored
    or referenced at something other than 2 bytes per element.  It fires on
    Qwen3.5 FP32 GatedDeltaNet ``recurrent_states`` in both the exact reference
    arm and in any ``bits=16`` packer output, which is exactly the accounting
    hazard the revision has to make visible rather than hide.
    """

    elements = int(elements)
    code_nbytes = int(code_nbytes)
    scale_nbytes = int(scale_nbytes)
    bias_nbytes = int(bias_nbytes)
    total_nbytes = code_nbytes + scale_nbytes + bias_nbytes
    is_packed_width = bits in PACKED_BITS
    if bits is not None and bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS} or None")
    if is_packed_width:
        expected_nbytes = eq3_component_nbytes(elements, int(bits), group_size)
        expected_code_nbytes = eq3_code_nbytes(elements, int(bits), group_size)
        expected_metadata_nbytes = eq3_metadata_nbytes(elements, group_size)
    else:
        # bits == 16 (verbatim BF16/FP32 copy) or an unquantized leaf
        expected_nbytes = verbatim_component_nbytes(elements, native_itemsize)
        expected_code_nbytes = expected_nbytes
        expected_metadata_nbytes = 0
    bf16_reference = (
        bf16_reference_nbytes(elements)
        if floating
        else verbatim_component_nbytes(elements, native_itemsize)
    )
    record = {
        "leaf_path": leaf_path,
        "layer_index": layer_index,
        "state_type": state_type,
        "elements": elements,
        "bits": bits,
        "floating": bool(floating),
        "quantized": bits is not None,
        "is_packed_width": is_packed_width,
        "group_size": int(group_size),
        "groups": group_count(elements, group_size) if is_packed_width else 0,
        "code_nbytes": code_nbytes,
        "scale_nbytes": scale_nbytes,
        "bias_nbytes": bias_nbytes,
        "metadata_nbytes": scale_nbytes + bias_nbytes,
        "total_nbytes": total_nbytes,
        "storage_nbytes": (
            int(storage_nbytes) if storage_nbytes is not None else total_nbytes
        ),
        "native_itemsize": int(native_itemsize),
        "native_reference_nbytes": native_reference_nbytes(
            elements, native_itemsize
        ),
        "bf16_reference_nbytes": bf16_reference,
        "dtype_inconsistent_reference": bool(
            not is_packed_width
            and floating
            and int(native_itemsize) != BF16_ITEMSIZE
        ),
        "eq3_expected_nbytes": expected_nbytes,
        "eq3_expected_code_nbytes": expected_code_nbytes,
        "eq3_expected_metadata_nbytes": expected_metadata_nbytes,
        "eq3_identity_ok": total_nbytes == expected_nbytes,
        "eq3_identity_checked": is_packed_width,
    }
    return record


def assert_eq3_component_identity(record: Mapping[str, Any]) -> None:
    """Assert one component equals ``ceil(n/64) * (64b/8 + 4)`` for its width.

    Verbatim (``bits == 16`` or unquantized) leaves are not subject to the
    identity; for those this function checks the weaker verbatim identity
    ``elements * native_itemsize`` instead, which is what the reference arm
    must satisfy for its ratio to be meaningful.
    """

    if record["total_nbytes"] == record["eq3_expected_nbytes"]:
        return
    raise Eq3IdentityError(
        "component {path} (layer={layer}, state={state}, bits={bits}) stores "
        "{actual} bytes but the format identity requires {expected} bytes "
        "(elements={elements}, group_size={group}): codes {code}/{code_exp}, "
        "metadata {meta}/{meta_exp}".format(
            path=record["leaf_path"],
            layer=record["layer_index"],
            state=record["state_type"],
            bits=record["bits"],
            actual=record["total_nbytes"],
            expected=record["eq3_expected_nbytes"],
            elements=record["elements"],
            group=record["group_size"],
            code=record["code_nbytes"],
            code_exp=record["eq3_expected_code_nbytes"],
            meta=record["metadata_nbytes"],
            meta_exp=record["eq3_expected_metadata_nbytes"],
        )
    )


def assert_eq3_identities(records: Iterable[Mapping[str, Any]]) -> None:
    """Assert the format identity for every component in ``records``."""

    for record in records:
        assert_eq3_component_identity(record)


def identity_violations(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return one summary dict per component that fails its byte identity."""

    violations = []
    for record in records:
        if record["total_nbytes"] == record["eq3_expected_nbytes"]:
            continue
        violations.append(
            {
                "leaf_path": record["leaf_path"],
                "layer_index": record["layer_index"],
                "state_type": record["state_type"],
                "bits": record["bits"],
                "elements": record["elements"],
                "group_size": record["group_size"],
                "actual_nbytes": record["total_nbytes"],
                "expected_nbytes": record["eq3_expected_nbytes"],
                "delta_nbytes": record["total_nbytes"]
                - record["eq3_expected_nbytes"],
            }
        )
    return violations


_SUM_FIELDS = (
    "elements",
    "code_nbytes",
    "scale_nbytes",
    "bias_nbytes",
    "metadata_nbytes",
    "total_nbytes",
    "storage_nbytes",
    "native_reference_nbytes",
    "bf16_reference_nbytes",
    "eq3_expected_nbytes",
)


def _sum_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = {field: 0 for field in _SUM_FIELDS}
    for record in records:
        for field in _SUM_FIELDS:
            totals[field] += int(record[field])
    totals["components"] = len(records)
    totals["bits"] = sorted({record["bits"] for record in records if record["bits"]})
    totals["dtype_inconsistent_components"] = sum(
        1 for record in records if record["dtype_inconsistent_reference"]
    )
    return totals


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def summarize_components(
    records: Sequence[Mapping[str, Any]],
    *,
    group_size: int = GROUP_SIZE,
    reconciliation_nbytes: int | None = None,
) -> dict[str, Any]:
    """Aggregate per-component records per state type and per layer.

    The returned dict is what a run emits for every row.  It contains the
    component list itself, per-state-type and per-layer aggregates, both
    reference counts, and the identity audit.
    """

    records = list(records)
    totals = _sum_records(records)
    by_state_type: dict[str, list[Mapping[str, Any]]] = {}
    by_layer: dict[Any, list[Mapping[str, Any]]] = {}
    for record in records:
        by_state_type.setdefault(record["state_type"], []).append(record)
        by_layer.setdefault(record["layer_index"], []).append(record)

    def _layer_sort_key(index: Any) -> tuple[int, int]:
        return (1, 0) if index is None else (0, int(index))

    violations = identity_violations(records)
    summary = {
        "format": {
            "equation": "ceil(n / g) * (g * b / 8 + 4)",
            "group_size": int(group_size),
            "metadata_nbytes_per_group": METADATA_NBYTES_PER_GROUP,
            "metadata_dtype": "bfloat16 scale + bfloat16 bias",
            "packed_widths": list(PACKED_BITS),
            "verbatim_widths": [16],
        },
        "components": records,
        "by_state_type": {
            state_type: _sum_records(group)
            for state_type, group in sorted(by_state_type.items())
        },
        "by_layer": [
            {"layer_index": index, **_sum_records(by_layer[index])}
            for index in sorted(by_layer, key=_layer_sort_key)
        ],
        "totals": totals,
        "packed_store_nbytes": totals["total_nbytes"],
        "packed_store_storage_nbytes": totals["storage_nbytes"],
        "native_dtype_reference_nbytes": totals["native_reference_nbytes"],
        "bf16_reference_nbytes": totals["bf16_reference_nbytes"],
        "native_dtype_ratio": _ratio(
            totals["native_reference_nbytes"], totals["total_nbytes"]
        ),
        "bf16_ratio": _ratio(
            totals["bf16_reference_nbytes"], totals["total_nbytes"]
        ),
        "dtype_inconsistent_components": totals["dtype_inconsistent_components"],
        "eq3_identity_violations": violations,
        "eq3_identity_ok": not violations,
        "checked_components": sum(
            1 for record in records if record["eq3_identity_checked"]
        ),
    }
    if reconciliation_nbytes is not None:
        summary["reconciliation"] = {
            "frozen_accountant_nbytes": int(reconciliation_nbytes),
            "breakdown_storage_nbytes": totals["storage_nbytes"],
            "delta_nbytes": totals["storage_nbytes"] - int(reconciliation_nbytes),
            "matches": totals["storage_nbytes"] == int(reconciliation_nbytes),
        }
    return summary


def empty_store_breakdown(group_size: int = GROUP_SIZE) -> dict[str, Any]:
    """The breakdown of an arm that retains nothing between requests."""

    summary = summarize_components([], group_size=group_size)
    summary["semantic"] = "this arm retains no cross-request state"
    return summary


# ---------------------------------------------------------------------------
# latency and throughput statistics
# ---------------------------------------------------------------------------


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = min(int(fraction * (len(ordered) - 1)), len(ordered) - 1)
    return float(ordered[index])


def decode_latency_summary(tpot_seconds: Sequence[float]) -> dict[str, Any]:
    """Per-step decode latency statistics; the full list stays in the row."""

    values = [float(value) for value in tpot_seconds]
    if not values:
        return {
            "decode_steps": 0,
            "decode_seconds_total": 0.0,
            "decode_seconds_mean": None,
            "decode_seconds_median": None,
            "decode_seconds_min": None,
            "decode_seconds_max": None,
            "decode_seconds_p90": None,
            "decode_seconds_p99": None,
            "decode_seconds_first": None,
            "decode_seconds_last": None,
            "decode_seconds_first_quarter_mean": None,
            "decode_seconds_last_quarter_mean": None,
        }
    quarter = max(len(values) // 4, 1)
    return {
        "decode_steps": len(values),
        "decode_seconds_total": float(sum(values)),
        "decode_seconds_mean": float(statistics.fmean(values)),
        "decode_seconds_median": float(statistics.median(values)),
        "decode_seconds_min": float(min(values)),
        "decode_seconds_max": float(max(values)),
        "decode_seconds_p90": _quantile(values, 0.90),
        "decode_seconds_p99": _quantile(values, 0.99),
        "decode_seconds_first": float(values[0]),
        "decode_seconds_last": float(values[-1]),
        "decode_seconds_first_quarter_mean": float(
            statistics.fmean(values[:quarter])
        ),
        "decode_seconds_last_quarter_mean": float(
            statistics.fmean(values[-quarter:])
        ),
    }


def throughput_summary(
    *,
    generated_tokens: int,
    ttft_seconds: float | None,
    tpot_seconds: Sequence[float],
    online_seconds: float | None,
    request_wall_seconds: float | None,
    end_to_end_including_build_seconds: float | None = None,
) -> dict[str, Any]:
    """Measured throughput plus the reconstructed model it must replace.

    ``reconstructed_tokens_per_second`` is ``n / (TTFT + n * median TPOT)``,
    i.e. the quantity Table 2's tok/s column was derived from.  It is emitted
    beside the measured numbers precisely so the reviewer can see the size of
    the modelling error rather than take the model on trust.
    """

    values = [float(value) for value in tpot_seconds]
    median_tpot = float(statistics.median(values)) if values else None
    reconstructed_seconds = (
        ttft_seconds + generated_tokens * median_tpot
        if ttft_seconds is not None and median_tpot is not None
        else None
    )
    measured = _ratio_float(generated_tokens, online_seconds)
    reconstructed = _ratio_float(generated_tokens, reconstructed_seconds)
    return {
        "generated_tokens": int(generated_tokens),
        "median_tpot_seconds": median_tpot,
        "online_seconds": online_seconds,
        "request_wall_seconds": request_wall_seconds,
        "end_to_end_including_build_seconds": end_to_end_including_build_seconds,
        "online_tokens_per_second": measured,
        "wall_tokens_per_second": _ratio_float(
            generated_tokens, request_wall_seconds
        ),
        "end_to_end_tokens_per_second": _ratio_float(
            generated_tokens, end_to_end_including_build_seconds
        ),
        "reconstructed_model_seconds": reconstructed_seconds,
        "reconstructed_tokens_per_second": reconstructed,
        "reconstructed_over_measured": (
            reconstructed / measured
            if measured not in (None, 0) and reconstructed is not None
            else None
        ),
        "instrumentation_overhead_seconds": (
            request_wall_seconds - online_seconds
            if request_wall_seconds is not None and online_seconds is not None
            else None
        ),
    }


def _ratio_float(numerator: float, denominator: float | None) -> float | None:
    if denominator is None or denominator <= 0:
        return None
    return float(numerator) / float(denominator)


# ---------------------------------------------------------------------------
# generation-length sweep planning
# ---------------------------------------------------------------------------


def arm_name(config: str, max_new_tokens: int) -> str:
    """Stable name for one (configuration, generation length) cell."""

    return f"{config}@n{int(max_new_tokens)}"


def parse_arm_name(name: str) -> tuple[str, int]:
    config, separator, suffix = name.rpartition("@n")
    if not separator or not suffix.isdigit():
        raise ValueError(f"not an arm name: {name}")
    return config, int(suffix)


def parse_config_length_limits(
    values: Sequence[str] | None,
) -> dict[str, int]:
    """Parse ``name=limit`` restrictions on which lengths a config may run at.

    ``dense-recompute`` recomputes the whole sequence for every token, so its
    cost is quadratic in the generation length.  Running it at 512 new tokens
    on a 4k document is not a useful measurement, it is a way to lose a job.
    The sweep therefore supports capping specific configurations.
    """

    limits: dict[str, int] = {}
    for value in values or ():
        name, separator, limit = str(value).partition("=")
        if not separator or not name.strip():
            raise ValueError(f"expected NAME=LIMIT, got {value!r}")
        limits[name.strip()] = int(limit)
    return limits


def sweep_arms(
    config_names: Sequence[str],
    lengths: Sequence[int],
    *,
    config_length_limits: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Expand configurations x generation lengths into the arm list."""

    if not config_names:
        raise ValueError("at least one configuration is required")
    if not lengths:
        raise ValueError("at least one generation length is required")
    if len(set(config_names)) != len(config_names):
        raise ValueError("configuration names must be unique")
    ordered_lengths = [int(length) for length in lengths]
    if any(length < 1 for length in ordered_lengths):
        raise ValueError("generation lengths must be positive")
    if len(set(ordered_lengths)) != len(ordered_lengths):
        raise ValueError("generation lengths must be unique")
    limits = dict(config_length_limits or {})
    unknown = sorted(set(limits) - set(config_names))
    if unknown:
        raise ValueError(f"length limits name unknown configs: {unknown}")
    arms = []
    for config in config_names:
        limit = limits.get(config)
        for length in ordered_lengths:
            if limit is not None and length > limit:
                continue
            arms.append(
                {
                    "arm": arm_name(config, length),
                    "config": config,
                    "max_new_tokens": length,
                }
            )
    if not arms:
        raise ValueError("every arm was excluded by a length limit")
    return arms


def shuffled_arm_orders(
    arm_names: Sequence[str], *, repeats: int, seed: int
) -> list[list[str]]:
    """Per-repeat permutation of the arms, mirroring ``shuffled_config_orders``.

    The same ``random.Random(seed + repeat).shuffle`` construction is used as
    in ``qcomem_deployment.shuffled_config_orders`` so that the interleaving
    discipline of the published deployment runs carries over unchanged.
    """

    if repeats < 1:
        raise ValueError("repeats must be positive")
    orders = []
    for repeat in range(repeats):
        order = list(arm_names)
        random.Random(seed + repeat).shuffle(order)
        orders.append(order)
    return orders


# ---------------------------------------------------------------------------
# row validation, shared by the runner and the aggregator
# ---------------------------------------------------------------------------


REQUIRED_ROW_FIELDS = (
    "arm",
    "config",
    "mode",
    "workload_id",
    "repeat",
    "max_new_tokens_requested",
    "generated_tokens",
    "ttft_seconds",
    "tpot_seconds",
    "decode_latency",
    "throughput",
    "store_breakdown",
    "persistent_document_nbytes",
    "eos_policy",
)


def validate_row(row: Mapping[str, Any]) -> list[str]:
    """Return a list of human-readable problems with one emitted row."""

    problems = []
    for field in REQUIRED_ROW_FIELDS:
        if field not in row or row[field] is None:
            problems.append(f"missing field {field}")
    if problems:
        return problems
    breakdown = row["store_breakdown"]
    if not breakdown.get("eq3_identity_ok", False):
        problems.append(
            "eq3 identity violations: "
            f"{breakdown.get('eq3_identity_violations')}"
        )
    generated = int(row["generated_tokens"])
    # ``max_new_tokens_effective`` differs from the requested sweep length only
    # when the LongBench per-dataset generation limit was deliberately applied.
    requested = int(
        row.get("max_new_tokens_effective", row["max_new_tokens_requested"])
    )
    if generated > requested:
        problems.append(
            f"generated {generated} exceeds effective cap {requested}"
        )
    steps = len(row["tpot_seconds"])
    if steps not in {max(generated - 1, 0), generated}:
        problems.append(
            f"decode step count {steps} is inconsistent with {generated} tokens"
        )
    if row["eos_policy"] == "ignore" and generated != requested:
        problems.append(
            "eos_policy=ignore must always reach the cap, "
            f"got {generated} of {requested}"
        )
    if row["eos_policy"] == "stop" and generated < requested and not row.get(
        "eos_stopped"
    ):
        problems.append("short generation is not marked eos_stopped")
    return problems


def summarize_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Median/mean summary of one arm over its per-item rows."""

    if not rows:
        raise ValueError("an arm summary needs at least one row")

    def median_of(getter) -> float | None:
        values = [getter(row) for row in rows]
        values = [float(value) for value in values if value is not None]
        return float(statistics.median(values)) if values else None

    def mean_of(getter) -> float | None:
        values = [getter(row) for row in rows]
        values = [float(value) for value in values if value is not None]
        return float(statistics.fmean(values)) if values else None

    generated = [int(row["generated_tokens"]) for row in rows]
    return {
        "arm": rows[0]["arm"],
        "config": rows[0]["config"],
        "mode": rows[0]["mode"],
        "max_new_tokens_requested": rows[0]["max_new_tokens_requested"],
        "rows": len(rows),
        "generated_tokens_median": float(statistics.median(generated)),
        "generated_tokens_min": min(generated),
        "generated_tokens_max": max(generated),
        "reached_cap_fraction": sum(
            1
            for row in rows
            if int(row["generated_tokens"])
            == int(row["max_new_tokens_requested"])
        )
        / len(rows),
        "store_nbytes_median": median_of(
            lambda row: row["persistent_document_nbytes"]
        ),
        "store_native_reference_nbytes_median": median_of(
            lambda row: row["store_breakdown"]["native_dtype_reference_nbytes"]
        ),
        "store_bf16_reference_nbytes_median": median_of(
            lambda row: row["store_breakdown"]["bf16_reference_nbytes"]
        ),
        "ttft_seconds_median": median_of(lambda row: row["ttft_seconds"]),
        "decode_seconds_median": median_of(
            lambda row: row["decode_latency"]["decode_seconds_median"]
        ),
        "decode_seconds_first_quarter_mean": mean_of(
            lambda row: row["decode_latency"]["decode_seconds_first_quarter_mean"]
        ),
        "decode_seconds_last_quarter_mean": mean_of(
            lambda row: row["decode_latency"]["decode_seconds_last_quarter_mean"]
        ),
        "online_seconds_median": median_of(
            lambda row: row["throughput"]["online_seconds"]
        ),
        "request_wall_seconds_median": median_of(
            lambda row: row["throughput"]["request_wall_seconds"]
        ),
        "online_tokens_per_second_median": median_of(
            lambda row: row["throughput"]["online_tokens_per_second"]
        ),
        "wall_tokens_per_second_median": median_of(
            lambda row: row["throughput"]["wall_tokens_per_second"]
        ),
        "reconstructed_tokens_per_second_median": median_of(
            lambda row: row["throughput"]["reconstructed_tokens_per_second"]
        ),
        "reconstructed_over_measured_median": median_of(
            lambda row: row["throughput"]["reconstructed_over_measured"]
        ),
        "f1_mean": mean_of(lambda row: row.get("f1")),
        "f1_items": sum(1 for row in rows if row.get("f1") is not None),
    }
