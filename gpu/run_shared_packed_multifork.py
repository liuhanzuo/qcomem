"""C1 runner: one packed entry shared across N>1 concurrent requests, audited.

This is a new runner beside ``run_deployment_bench.py`` and
``run_deployment_length_sweep.py``, not a replacement for either.  It reuses
their workload construction verbatim (identical LongBench slicing, identical
prompt protocol, identical SHA/revision guards) and reuses
``qcomem_deployment``'s persistent-state builder and published Read path
unchanged.

What it adds
------------

* the ``shared-packed-view`` fork mode: the quantized depth-split entry is
  dequantized **once** and forked by N concurrent requests, with immutable
  document tensors shared, mutable conv/recurrent state rebound to private
  storage at the registered transition, and the attention tail owned per
  request;
* the ForkAudit contract instantiated on that path -- the manuscript's seven
  targets plus the three packed-entry obligations it names as untested --
  with coverage recorded separately from verdict;
* per-request transient working-set fields for **both** arms (Q-CoMem and
  full prefix): dequantized/materialized bytes, peak transient allocation, and
  steady-state resident bytes, emitted as first-class row fields rather than
  as diagnostics;
* a semantic-equivalence gate: every request's N>1 output must be token-for-token
  identical to what the published N=1 private-materialization path emits on the
  same inputs, and any discrepancy is recorded and surfaced rather than
  dropped.

Arms measured at every fanout
-----------------------------

``qcomem-shared-packed``
    one entry, one dequantized view, N requests.

``qcomem-private-materialize``
    the published Read path at the same fanout: N full private dequantized
    copies.  This is the arm behind Tables 1 and 2 and it stays reproducible.

``full-prefix``
    the exact full-model prefix cache, deep-cloned per request.  Present so
    the working set exists for both methods, which is what Eq. 1's
    method-independence premise assumes and the paper never measured.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from qcomem_deployment import (
    DEFAULT_MIXED_LAYER_BITS,
    NvmlProcessSampler,
    build_persistent_state,
    config_asdict,
    environment_metadata,
    load_mixed_policy,
    parse_deployment_config,
    parse_layer_bits,
)
from qcomem_multifork_accounting import (
    AGGREGATE_SCHEMA,
    PROTOCOL,
    REBIND_POLICIES,
    SHARD_SCHEMA,
    TAIL_POLICIES,
    compare_token_traces,
    crossover_request_count,
    fanout_plan,
    format_mib,
    request_ids,
    summarize_multifork_rows,
    validate_multifork_row,
    working_set_row,
)
from qcomem_shared_packed_fork import (
    prepare_shared_packed_entry,
    run_full_prefix_multifork,
    run_shared_packed_multifork,
)
from qcomem_shared_packed_forkaudit import (
    audit_shared_packed_multifork,
    published_private_reference_traces,
    run_audited_shared_packed_multifork,
    run_shared_packed_multifork_gate,
)
from qcomem_torch import TorchSplitCausalLM
from run_deployment_bench import (
    batch_prefix,
    longbench_workloads,
    synthetic_workloads,
    visible_nvml_index,
)
from run_downstream import answer_f1, atomic_json


DEFAULT_FANOUTS = (1, 2, 4)
DEFAULT_CONFIG = "qcomem-d7-frozen-static"


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# request construction
# ---------------------------------------------------------------------------


def build_request_queries(
    *,
    workloads: Sequence[dict[str, Any]],
    workload_index: int,
    fanout: int,
    query_source: str,
    device: str = "cuda",
) -> tuple[list[tuple[str, torch.Tensor]], dict[str, Any]]:
    """Build ``fanout`` distinct queries against one shared document.

    ``cross-item`` gives request ``r`` the question of workload
    ``(workload_index + r) % len(workloads)``, i.e. different LongBench
    questions asked of the same document.  When the rank holds fewer workloads
    than the fanout it would otherwise repeat a question, so it falls back to
    ``truncated-window`` and records that it did.

    ``truncated-window`` gives request ``r`` the first ``L - r`` tokens of the
    item's own question, which is the two-distinct-query-windows protocol the
    earlier transfer experiment used.

    Only request ``r00`` carries the document's own matched question; F1 is
    reported for that request only, and every other request records which
    workload its query came from.
    """

    ids = request_ids(fanout)
    base = workloads[workload_index]
    base_query = base["query_tokens"]
    length = int(base_query.reshape(-1).shape[0])
    effective_source = query_source
    if query_source == "cross-item" and len(workloads) < fanout:
        effective_source = "truncated-window"
    if effective_source == "truncated-window" and length <= fanout:
        raise ValueError(
            f"truncated-window needs a query longer than the fanout "
            f"({length} tokens for fanout {fanout})"
        )

    queries: list[tuple[str, torch.Tensor]] = []
    provenance: list[dict[str, Any]] = []
    for index, request_id in enumerate(ids):
        if effective_source == "cross-item":
            source = workloads[(workload_index + index) % len(workloads)]
            tokens = batch_prefix(source["query_tokens"], 1 << 30)
            provenance.append(
                {
                    "request_id": request_id,
                    "query_source_workload_id": source["workload_id"],
                    "query_matches_document": index == 0,
                    "query_tokens": int(tokens.numel()),
                }
            )
        else:
            tokens = batch_prefix(base_query, length - index)
            provenance.append(
                {
                    "request_id": request_id,
                    "query_source_workload_id": base["workload_id"],
                    "query_matches_document": index == 0,
                    "query_tokens": int(tokens.numel()),
                }
            )
        queries.append((request_id, tokens.to(device)))
    distinct = len({tuple(tokens.reshape(-1).tolist()) for _, tokens in queries})
    metadata = {
        "query_source_requested": query_source,
        "query_source_effective": effective_source,
        "distinct_query_count": distinct,
        "queries_are_distinct": distinct == fanout,
        "provenance": provenance,
    }
    return queries, metadata


# ---------------------------------------------------------------------------
# one measured cell
# ---------------------------------------------------------------------------


def _trace_row(
    *,
    trace,
    arm: str,
    fork_mode: str,
    entry_retained_nbytes: int,
    shared_view_nbytes: int,
) -> dict[str, Any]:
    request_traces = trace.request_traces
    return working_set_row(
        arm=arm,
        request_count=len(request_traces),
        entry_retained_nbytes=entry_retained_nbytes,
        shared_view_nbytes=shared_view_nbytes,
        per_request_materialized_nbytes=[
            row.materialized_nbytes for row in request_traces
        ],
        per_request_steady_resident_nbytes=[
            row.steady_resident_nbytes for row in request_traces
        ],
        measured_baseline_allocated_nbytes=trace.baseline_allocated_nbytes,
        measured_peak_allocated_nbytes=trace.peak_allocated_nbytes,
        measured_steady_allocated_nbytes=trace.steady_allocated_nbytes,
    ) | {
        "fork_mode": fork_mode,
        "rebind_policy": trace.rebind_policy,
        "tail_policy": trace.tail_policy,
        "setup_seconds": trace.setup_seconds,
        "transition_seconds": trace.transition_seconds,
        "decode_seconds": trace.decode_seconds,
        "phase_allocated_nbytes": dict(trace.phase_allocated_nbytes),
        "request_traces": [row.summary() for row in request_traces],
        "transient_concat_peak_nbytes_max": max(
            (row.transient_concat_peak_nbytes for row in request_traces), default=0
        ),
    }


@torch.inference_mode()
def measure_multifork_workload(
    *,
    adapter,
    tokenizer,
    config,
    workloads: Sequence[dict[str, Any]],
    workload_index: int,
    fanouts: Sequence[int],
    repeat: int,
    args,
    eos_ids: set[int],
) -> list[dict[str, Any]]:
    """Measure every arm at every fanout for one document, audited once.

    The fanouts of one document are measured together for two reasons.  The
    cross-N prefix-consistency target needs request ``r00``'s traces at two or
    more fanouts, which do not exist until the last fanout has run; and the
    published N=1 reference traces are computed once and reused, so the shared
    and private arms are compared against exactly the same reference.

    The query list is built once at the largest fanout and sliced.  Both query
    sources are prefix-consistent -- request ``r`` gets the same query at every
    fanout -- so slicing is the same as rebuilding.
    """

    workload = workloads[workload_index]
    document = workload["document_tokens"].cuda()
    max_fanout = max(fanouts)
    all_queries, query_metadata = build_request_queries(
        workloads=workloads,
        workload_index=workload_index,
        fanout=max_fanout,
        query_source=args.query_source,
    )
    stop_ids = set() if args.eos_policy == "ignore" else set(eos_ids)
    torch.cuda.empty_cache()

    packed_state = build_persistent_state(
        adapter,
        config,
        document,
        group_size=args.group_size,
        fork_strategy="deep-clone",
    )
    entry_retained = int(packed_state.stored_nbytes)

    # The published Read path, called unchanged, one request at a time.
    reference = published_private_reference_traces(
        adapter,
        config,
        document,
        all_queries,
        packed_state,
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=stop_ids,
    )

    audited_by_fanout: dict[int, dict[str, Any]] = {}
    private_by_fanout: dict[int, Any] = {}
    full_prefix_by_fanout: dict[int, Any] = {}
    full_prefix_retained: dict[int, int] = {}
    traces_by_fanout: dict[int, dict[str, list[int]]] = {}

    for fanout in fanouts:
        queries = list(all_queries[:fanout])
        audited = run_audited_shared_packed_multifork(
            adapter,
            packed_state,
            document,
            queries,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=stop_ids,
            rebind_policy=args.rebind_policy,
            tail_policy=args.tail_policy,
            policy_identity={
                "config": config.name,
                "group_size": args.group_size,
                "max_new_tokens": args.max_new_tokens,
            },
            private_reference_traces={
                request_id: reference[request_id] for request_id, _ in queries
            },
        )
        audited_by_fanout[fanout] = audited
        traces_by_fanout[fanout] = audited["trace"].token_traces()

        private_entry = prepare_shared_packed_entry(
            packed_state, share_mode="private-materialize"
        )
        private_by_fanout[fanout] = run_shared_packed_multifork(
            adapter,
            private_entry,
            queries,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=stop_ids,
        )
        del private_entry

        full_prefix_state = adapter.write_full_prefix(document)
        full_prefix_retained[fanout] = int(full_prefix_state.stored_nbytes)
        full_prefix_by_fanout[fanout] = run_full_prefix_multifork(
            adapter,
            full_prefix_state,
            queries,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=stop_ids,
        )
        del full_prefix_state
        torch.cuda.empty_cache()

    common = {
        "protocol": PROTOCOL,
        "config": config.name,
        "effective_config": config_asdict(config),
        "workload_id": workload["workload_id"],
        "workload_kind": workload["kind"],
        "dataset": workload["dataset"],
        "source_index": workload["source_index"],
        "source_id": workload["source_id"],
        "source_revision": workload["source_revision"],
        "document_tokens": int(document.numel()),
        "repeat": repeat,
        "max_new_tokens": args.max_new_tokens,
        "eos_policy": args.eos_policy,
        "group_size": args.group_size,
        "fanouts_measured": list(fanouts),
    }
    references = workload["references"]
    matched = all_queries[0][0]
    rows: list[dict[str, Any]] = []

    for fanout in fanouts:
        queries = list(all_queries[:fanout])
        audited = audited_by_fanout[fanout]
        shared_entry = audited["entry"]
        shared_trace = audited["trace"]
        # Re-evaluate the contract now that every fanout's traces exist, so the
        # cross-N target is non-vacuous.  No execution is repeated.
        audit = audit_shared_packed_multifork(
            **audited["audit_inputs"], traces_by_fanout=traces_by_fanout
        )
        private_trace = private_by_fanout[fanout]
        full_prefix_trace = full_prefix_by_fanout[fanout]
        fanout_reference = {
            request_id: reference[request_id] for request_id, _ in queries
        }

        arm_rows = {
            "qcomem-shared-packed": _trace_row(
                trace=shared_trace,
                arm="qcomem-shared-packed",
                fork_mode=shared_entry.effective_share_mode,
                entry_retained_nbytes=entry_retained,
                shared_view_nbytes=int(shared_entry.shared_view_nbytes),
            ),
            "qcomem-private-materialize": _trace_row(
                trace=private_trace,
                arm="qcomem-private-materialize",
                fork_mode="private-materialize",
                entry_retained_nbytes=entry_retained,
                shared_view_nbytes=0,
            ),
            "full-prefix": _trace_row(
                trace=full_prefix_trace,
                arm="full-prefix",
                fork_mode="private-materialize",
                entry_retained_nbytes=full_prefix_retained[fanout],
                shared_view_nbytes=0,
            ),
        }
        equivalences = {
            "qcomem-shared-packed": audit["detail"]["cross_arm_equivalence"],
            "qcomem-private-materialize": compare_token_traces(
                reference=fanout_reference,
                candidate=private_trace.token_traces(),
                reference_label="n1-private-materialize",
                candidate_label=f"n{fanout}-private-materialize",
            ),
            # never a gate: the full-prefix arm consumes the document/query
            # boundary differently and the Qwen3.5 recurrence is sensitive to it
            "full-prefix": compare_token_traces(
                reference=fanout_reference,
                candidate=full_prefix_trace.token_traces(),
                reference_label="n1-private-materialize",
                candidate_label=f"n{fanout}-full-prefix",
            ),
        }
        crossover = crossover_request_count(
            left=arm_rows["qcomem-shared-packed"]["resident_model"],
            right=arm_rows["full-prefix"]["resident_model"],
            max_request_count=args.crossover_search_limit,
        )
        traces_by_arm = {
            "qcomem-shared-packed": shared_trace,
            "qcomem-private-materialize": private_trace,
            "full-prefix": full_prefix_trace,
        }
        for arm, arm_row in arm_rows.items():
            merged = {**common, **arm_row}
            merged["query_metadata"] = {
                **query_metadata,
                "request_count": fanout,
                "provenance": query_metadata["provenance"][:fanout],
            }
            merged["semantic_equivalence"] = equivalences[arm]
            if arm == "qcomem-shared-packed":
                merged["ownership_ledger"] = audit["ownership"]["final_ledger"]
                merged["forkaudit"] = {
                    "contract_summary": audit["contract_summary"],
                    "target_rows": audit["target_rows"],
                    "sharing_window": audit["sharing_window"],
                    "sharing_efficiency_at_window": audit["ownership"][
                        "sharing_efficiency_at_window"
                    ],
                    "sharing_efficiency_at_setup": audit["ownership"][
                        "sharing_efficiency_at_setup"
                    ],
                    "mask_size_call_forms": audit["ownership"][
                        "mask_size_call_forms"
                    ],
                    "trusted_computing_base": audit["trusted_computing_base"],
                }
                if not args.drop_receipt_details:
                    merged["forkaudit"]["detail"] = audit["detail"]
                    merged["forkaudit"]["ownership"] = audit["ownership"]
                merged["resident_crossover_vs_full_prefix"] = crossover
                merged["entry_components"] = (
                    shared_entry.deployment_memory_components()
                )
            else:
                merged["ownership_ledger"] = {
                    "predicate_id": "ALL_MUTABLE_CACHE_STORAGE_PAIRWISE_DISJOINT",
                    "request_count": fanout,
                    "shared_entry_nbytes": 0,
                    "total_private_nbytes": sum(
                        arm_row["per_request_steady_resident_nbytes"]
                    ),
                    "per_request": {},
                    "pairwise": [],
                    "passed": None,
                    "non_vacuous": False,
                    "semantic": (
                        "this arm shares nothing by construction, so no "
                        "shared/private split is measured; the byte totals are "
                        "the private copies"
                    ),
                }
            prediction = tokenizer.decode(
                traces_by_arm[arm].token_traces()[matched], skip_special_tokens=True
            ).strip()
            merged["matched_request_id"] = matched
            merged["prediction"] = prediction
            merged["references"] = references
            merged["f1"] = (
                max(answer_f1(prediction, item) for item in references)
                if references
                else None
            )
            problems = validate_multifork_row(merged)
            if problems:
                merged["row_validation_problems"] = problems
                if args.strict_accounting:
                    raise SystemExit(
                        f"row for {arm} / {workload['workload_id']} at "
                        f"N={fanout} is invalid: {problems}"
                    )
            rows.append(merged)

    del packed_state, audited_by_fanout, private_by_fanout, full_prefix_by_fanout
    torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "C1: one quantized depth-split entry shared across N>1 concurrent "
            "requests, with the ForkAudit contract instantiated on that path"
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument(
        "--workload", choices=("longbench", "synthetic"), required=True
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--expected-data-sha256")
    parser.add_argument("--expected-source-revision")
    parser.add_argument("--expected-source-indices", type=int, nargs="*")
    parser.add_argument("--expected-workloads", type=int)
    parser.add_argument("--protocol-label", default="c1-shared-packed-multifork")
    parser.add_argument("--limit-per-dataset", type=int, default=4)
    parser.add_argument("--source-index-start", type=int, default=6)
    parser.add_argument("--source-index-end", type=int, default=35)
    parser.add_argument(
        "--exclude-source-indices", type=int, nargs="*", default=(4, 5)
    )
    parser.add_argument("--allow-test-v2", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument(
        "--context-lengths", type=int, nargs="+", default=(4096, 8192)
    )
    parser.add_argument("--synthetic-repetitions", type=int, default=2)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mixed-layer-bits",
        default=",".join(str(bit) for bit in DEFAULT_MIXED_LAYER_BITS),
    )
    parser.add_argument("--mixed-policy-file", type=Path)
    parser.add_argument("--mixed-policy-name", default="same_memory_as_frozen")
    parser.add_argument(
        "--fanouts",
        type=int,
        nargs="+",
        default=list(DEFAULT_FANOUTS),
        help="request counts to measure; at least one must exceed 1",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument(
        "--tail-policy", choices=TAIL_POLICIES, default="borrowed-prefix"
    )
    parser.add_argument(
        "--rebind-policy", choices=REBIND_POLICIES, default="transition"
    )
    parser.add_argument(
        "--query-source",
        choices=("cross-item", "truncated-window"),
        default="cross-item",
        help=(
            "how the N distinct queries against one document are built; "
            "cross-item falls back to truncated-window when the rank holds "
            "fewer workloads than the fanout, and records that it did"
        ),
    )
    parser.add_argument("--eos-policy", choices=("ignore", "stop"), default="ignore")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--gate-document-tokens", type=int, default=256)
    parser.add_argument("--gate-query-tokens", type=int, default=32)
    parser.add_argument("--gate-new-tokens", type=int, default=4)
    parser.add_argument("--gate-fanout", type=int, default=2)
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--crossover-search-limit", type=int, default=4096)
    parser.add_argument(
        "--drop-receipt-details",
        action="store_true",
        help="omit the per-tensor audit detail from the shard; verdicts stay",
    )
    parser.add_argument(
        "--no-strict-accounting",
        dest="strict_accounting",
        action="store_false",
        help="record row validation problems instead of raising on them",
    )
    parser.set_defaults(strict_accounting=True)
    return parser


def resolve_config(args):
    mixed_layer_bits = parse_layer_bits(args.mixed_layer_bits)
    residual_bits = 4
    policy_metadata: dict[str, Any] = {
        "source": "cli-default",
        "residual_bits": residual_bits,
        "cache_layer_bits": list(mixed_layer_bits),
    }
    if args.mixed_policy_file is not None:
        import hashlib

        residual_bits, mixed_layer_bits = load_mixed_policy(
            args.mixed_policy_file, args.mixed_policy_name
        )
        policy_metadata = {
            "source": str(args.mixed_policy_file),
            "sha256": hashlib.sha256(
                args.mixed_policy_file.read_bytes()
            ).hexdigest(),
            "policy_name": args.mixed_policy_name,
            "residual_bits": residual_bits,
            "cache_layer_bits": list(mixed_layer_bits),
        }
    config = parse_deployment_config(args.config, mixed_layer_bits=mixed_layer_bits)
    if config.mode != "qcomem":
        raise SystemExit(
            f"--config must name a Q-CoMem split arm; {args.config} is "
            f"mode {config.mode}"
        )
    return config, policy_metadata


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if not (0 <= args.rank < args.world_size):
        raise SystemExit("rank must be within world size")
    fanouts = fanout_plan(args.fanouts, require_multifork=True)
    if args.gate_fanout < 2:
        raise SystemExit("the preflight gate requires a fanout of at least 2")

    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.cuda.set_device(0)
    random.seed(args.seed + args.rank)
    torch.manual_seed(args.seed + args.rank)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.workload == "longbench":
        workloads, workload_metadata = longbench_workloads(tokenizer, args)
    else:
        workloads, workload_metadata = synthetic_workloads(tokenizer, args)
    if args.expected_data_sha256 is not None and (
        workload_metadata.get("data_sha256") != args.expected_data_sha256
    ):
        raise SystemExit("LongBench data SHA256 does not match the frozen protocol")
    if args.expected_source_revision is not None and (
        workload_metadata.get("source_revisions") != [args.expected_source_revision]
    ):
        raise SystemExit(
            "LongBench source revision does not match the frozen protocol"
        )
    if args.expected_source_indices is not None:
        actual = sorted({int(item["source_index"]) for item in workloads})
        if actual != sorted(set(args.expected_source_indices)):
            raise SystemExit(
                f"source indices {actual} do not match frozen "
                f"{sorted(set(args.expected_source_indices))}"
            )
    if args.expected_workloads is not None and len(workloads) != args.expected_workloads:
        raise SystemExit(
            f"expected {args.expected_workloads} workloads, found {len(workloads)}"
        )
    shard_workloads = workloads[args.rank :: args.world_size]
    if not shard_workloads:
        raise SystemExit("rank has no workload shard")

    config, mixed_policy = resolve_config(args)

    load_started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().cuda()
    _sync()
    model_load_seconds = time.perf_counter() - load_started
    model_allocated_bytes = torch.cuda.memory_allocated()
    total_device_bytes = torch.cuda.get_device_properties(0).total_memory
    adapter = TorchSplitCausalLM(model)
    nvml = NvmlProcessSampler(visible_nvml_index())

    eos_value = tokenizer.eos_token_id
    eos_ids = {int(eos_value)} if isinstance(eos_value, int) else set(eos_value or [])
    stop_ids = set() if args.eos_policy == "ignore" else set(eos_ids)

    first = shard_workloads[0]
    gate_document = batch_prefix(
        first["document_tokens"], args.gate_document_tokens
    ).cuda()
    gate_query_length = int(first["query_tokens"].reshape(-1).shape[0])
    gate_query_budget = min(args.gate_query_tokens, gate_query_length)
    if gate_query_budget <= args.gate_fanout:
        raise SystemExit(
            f"the gate needs a query longer than its fanout: {gate_query_budget} "
            f"tokens for fanout {args.gate_fanout}"
        )
    gate_queries = [
        (
            request_id,
            batch_prefix(first["query_tokens"], gate_query_budget - index).cuda(),
        )
        for index, request_id in enumerate(request_ids(args.gate_fanout))
    ]
    gate_full_prefix = adapter.write_full_prefix(gate_document)

    destination = args.run_dir / f"multifork-shard-{args.rank}.json"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    gates: dict[str, Any] = {
        "shared_packed_multifork_gate": run_shared_packed_multifork_gate(
            adapter,
            config,
            gate_document,
            gate_queries,
            group_size=args.group_size,
            max_new_tokens=args.gate_new_tokens,
            eos_token_ids=stop_ids,
            rebind_policy=args.rebind_policy,
            tail_policy=args.tail_policy,
            full_prefix_state=gate_full_prefix,
        )
    }
    gates_passed = all(bool(gate.get("passed")) for gate in gates.values())
    header = {
        "schema": SHARD_SCHEMA,
        "aggregate_schema": AGGREGATE_SCHEMA,
        "protocol": PROTOCOL,
        "rank": args.rank,
        "world_size": args.world_size,
        "model": str(args.model),
        "model_load_seconds": model_load_seconds,
        "model_cuda_allocated_baseline_bytes": model_allocated_bytes,
        "device_total_memory_bytes": total_device_bytes,
        "environment": environment_metadata(model),
        "transformers": transformers.__version__,
        "nvml_sampler": nvml.metadata(),
        "workload": args.workload,
        "workload_metadata": workload_metadata,
        "config": config_asdict(config),
        "mixed_policy": mixed_policy,
        "protocol_settings": {
            "label": args.protocol_label,
            "family": "c1-shared-packed-multifork",
            "fanouts": fanouts,
            "max_new_tokens": args.max_new_tokens,
            "tail_policy": args.tail_policy,
            "rebind_policy": args.rebind_policy,
            "query_source": args.query_source,
            "eos_policy": args.eos_policy,
            "group_size": args.group_size,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "seed": args.seed,
            "gate_document_tokens": args.gate_document_tokens,
            "gate_query_tokens": args.gate_query_tokens,
            "gate_new_tokens": args.gate_new_tokens,
            "gate_fanout": args.gate_fanout,
            "expected_data_sha256": args.expected_data_sha256,
            "expected_source_revision": args.expected_source_revision,
        },
        "gates": gates,
    }
    if not gates_passed:
        atomic_json(destination, {**header, "status": "gate_failed", "rows": []})
        raise SystemExit("C1 shared-packed multifork gate failed")
    if args.gate_only:
        atomic_json(destination, {**header, "status": "gate_passed", "rows": []})
        print(f"C1_GATE_PASSED {destination}", flush=True)
        return

    del gate_document, gate_queries, gate_full_prefix
    torch.cuda.empty_cache()

    for _ in range(args.warmups):
        warm_queries, _ = build_request_queries(
            workloads=shard_workloads,
            workload_index=0,
            fanout=max(fanouts),
            query_source=args.query_source,
        )
        warm_document = shard_workloads[0]["document_tokens"].cuda()
        warm_packed = build_persistent_state(
            adapter,
            config,
            warm_document,
            group_size=args.group_size,
            fork_strategy="deep-clone",
        )
        warm_entry = prepare_shared_packed_entry(
            warm_packed,
            share_mode="shared-packed-view",
            rebind_policy=args.rebind_policy,
            tail_policy=args.tail_policy,
        )
        run_shared_packed_multifork(
            adapter,
            warm_entry,
            warm_queries,
            max_new_tokens=min(4, args.max_new_tokens),
            eos_token_ids=stop_ids,
        )
        del warm_packed, warm_entry, warm_queries, warm_document
        torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    result = {**header, "status": "running", "rows": rows}
    atomic_json(destination, result)
    for workload_index, workload in enumerate(shard_workloads):
        for repeat in range(args.repeats):
            cell = measure_multifork_workload(
                adapter=adapter,
                tokenizer=tokenizer,
                config=config,
                workloads=shard_workloads,
                workload_index=workload_index,
                fanouts=fanouts,
                repeat=repeat,
                args=args,
                eos_ids=eos_ids,
            )
            rows.extend(cell)
            result["rows"] = rows
            atomic_json(destination, result)
            for shared in [
                row for row in cell if row["arm"] == "qcomem-shared-packed"
            ]:
                print(
                    json.dumps(
                        {
                            "rank": args.rank,
                            "workload": workload["workload_id"],
                            "fanout": shared["request_count"],
                            "repeat": repeat,
                            "entry_retained_mib": format_mib(
                                shared["entry_retained_nbytes"]
                            ),
                            "shared_view_mib": format_mib(
                                shared["shared_dequantized_view_nbytes"]
                            ),
                            "transient_materialized_mib": format_mib(
                                shared["transient_materialized_nbytes_total"]
                            ),
                            "peak_transient_mib": format_mib(
                                shared["peak_transient_allocation_nbytes"]
                            ),
                            "contract_status": shared["forkaudit"][
                                "contract_summary"
                            ]["overall_contract_status"],
                            "open_targets": shared["forkaudit"][
                                "contract_summary"
                            ]["open_targets"],
                            "token_identical": shared["semantic_equivalence"][
                                "token_sequences_identical"
                            ],
                        }
                    ),
                    flush=True,
                )
    result["arm_summaries"] = summarize_multifork_rows(rows)
    result["status"] = "completed"
    result["rows"] = rows
    atomic_json(destination, result)
    print(f"SAVED {destination}", flush=True)


if __name__ == "__main__":
    main()
