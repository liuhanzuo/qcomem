from __future__ import annotations

"""Pre-registerable captured-boundary attention/GDN oracle producer.

The producer records candidate outputs and the exact post-transformation inputs
at two narrow numerical boundaries.  It intentionally makes no end-to-end or
capture-independence claim; all numerical decisions are made later by the
candidate-import-free NumPy reference process.
"""

import argparse
import hashlib
import inspect
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


CAPTURE_SCHEMA = "forkaudit-r30-expanded-oracle-capture-v1"
INPUT_SCHEMA = "forkaudit-r30-expanded-oracle-input-manifest-v1"
PREREG_SCHEMA = "forkaudit-r30-expanded-oracle-preregistration-v1"
MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
MODEL_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
MODEL_WEIGHT_LEDGER_SHA256 = "8314a82c9188b9b817193e039b0b0eb0636b328512f19b0c12455853b7e20014"
PG19_DATA_SHA256 = "ef18893b6bfb0f4b8cb29eab85ccf2a0ef1fdb44606e9742a70405cca564e18c"
PG19_MANIFEST_SHA256 = "5d789d67aa239f089e92de8a4267b86d2f1d2723d5f1370970883738f5f89a9c"
WINDOWS_SHA256 = "cca6c3643b3e77178cac3c906ef0202c686a23de9578287adef95a5d8af16aa9"
FULL_LAYERS = tuple(range(3, 40, 4))
LINEAR_LAYERS = tuple(index for index in range(40) if index not in FULL_LAYERS)
SELECTED_GDN_LAYERS = (0, 1, 2, 4, 8, 12, 16, 20, 24, 28, 34, 38)
CASE_SPECS = (
    {"case_id": "PG19-W1-Q0", "window_index": 1, "query_bank_index": 0},
    {"case_id": "PG19-W6-Q2", "window_index": 6, "query_bank_index": 2},
)
DOCUMENT_TOKENS = 256
QUERY_TOKENS = 8
PAGE_SIZE = 128


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return sha256_bytes(raw)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _save_array(root: Path, row_id: str, name: str, value: torch.Tensor) -> dict[str, Any]:
    array = value.detach().contiguous().cpu().float().numpy().astype(np.float32, copy=False)
    relative = Path("sidecars") / row_id / f"{name}.npy"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    return {
        "relative_path": relative.as_posix(),
        "sha256": sha256_file(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "nbytes": int(array.nbytes),
    }


def _build_inputs(
    *, model_dir: Path, pg19_data: Path, pg19_manifest: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from qcomem_joint_policy import audit_pg19_train_calibration, build_pg19_calibration_windows
    from qcomem_vllm_paged_multifork_resident import build_pg19_train_query_bank
    from transformers import AutoTokenizer

    require(sha256_file(pg19_data) == PG19_DATA_SHA256, "PG-19 data SHA drift")
    require(sha256_file(pg19_manifest) == PG19_MANIFEST_SHA256, "PG-19 manifest SHA drift")
    records, _audit = audit_pg19_train_calibration(
        pg19_data,
        pg19_manifest,
        expected_data_sha256=PG19_DATA_SHA256,
        expected_manifest_sha256=PG19_MANIFEST_SHA256,
        minimum_books=8,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    windows, windows_sha = build_pg19_calibration_windows(
        records,
        tokenizer,
        books=8,
        document_tokens=4096,
        query_tokens=32,
        stride=263,
        candidate_windows_per_book=8,
        seed=20260819,
    )
    require(windows_sha == WINDOWS_SHA256, "frozen PG-19 window selection drift")
    built: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for spec in CASE_SPECS:
        window = windows[int(spec["window_index"])]
        queries, query_audit = build_pg19_train_query_bank(
            records,
            tokenizer,
            window,
            document_tokens=4096,
            query_tokens=32,
            count=4,
            query_stride=64,
        )
        document = window.document_ids[:DOCUMENT_TOKENS].unsqueeze(0).contiguous()
        query = queries[int(spec["query_bank_index"])][:, :QUERY_TOKENS].contiguous()
        require(tuple(document.shape) == (1, DOCUMENT_TOKENS), "document shape drift")
        require(tuple(query.shape) == (1, QUERY_TOKENS), "query shape drift")
        built.append({"case_id": spec["case_id"], "document": document, "query": query})
        receipts.append(
            {
                **spec,
                "source_object": str(window.source_object),
                "source_id": str(window.source_id),
                "document_token_ids_sha256": sha256_tensor(document),
                "query_token_ids_sha256": sha256_tensor(query),
                "full_window_document_sha256": sha256_tensor(window.document_ids),
                "query_bank_sha256": str(query_audit["query_bank_sha256"]),
            }
        )
    require(len({row["source_object"] for row in receipts}) == len(CASE_SPECS), "input cases reuse one source object")
    manifest = {
        "schema_version": INPUT_SCHEMA,
        "selection_created_without_candidate_execution": True,
        "model_revision": MODEL_REVISION,
        "dataset": "PG-19 train-only frozen 64-book artifact",
        "pg19_data_sha256": PG19_DATA_SHA256,
        "pg19_manifest_sha256": PG19_MANIFEST_SHA256,
        "windows_sha256": WINDOWS_SHA256,
        "window_algorithm": "books=8,document=4096,query=32,stride=263,candidates=8,seed=20260819",
        "document_tokens": DOCUMENT_TOKENS,
        "query_tokens": QUERY_TOKENS,
        "cases": receipts,
    }
    return built, manifest


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    _inputs, manifest = _build_inputs(
        model_dir=args.model_dir,
        pg19_data=args.pg19_data,
        pg19_manifest=args.pg19_manifest,
    )
    require(not args.output.exists(), "input manifest path already exists")
    atomic_json(args.output, manifest)
    return manifest


def _validate_preregistration(
    prereg: dict[str, Any], *, input_manifest_path: Path, reference_path: Path | None = None
) -> None:
    require(prereg.get("schema_version") == PREREG_SCHEMA, "preregistration schema drift")
    require(prereg.get("created_before_candidate_execution") is True, "preregistration is not pre-execution")
    require(prereg.get("post_execution_amendment_allowed") is False, "preregistration permits amendment")
    require(prereg.get("model_id") == MODEL_ID and prereg.get("model_revision") == MODEL_REVISION, "model binding drift")
    require(prereg.get("model_weight_ledger_raw_sha256") == MODEL_WEIGHT_LEDGER_SHA256, "model ledger binding drift")
    require(prereg.get("input_manifest_raw_sha256") == sha256_file(input_manifest_path), "input manifest binding drift")
    require(prereg.get("document_tokens") == DOCUMENT_TOKENS and prereg.get("query_tokens") == QUERY_TOKENS, "token geometry drift")
    require(tuple(prereg.get("full_attention_layers", ())) == FULL_LAYERS, "attention layer selection drift")
    require(tuple(prereg.get("gdn_layers", ())) == SELECTED_GDN_LAYERS, "GDN layer selection drift")
    require(prereg.get("input_cases") == list(CASE_SPECS), "input case selection drift")
    bindings = prereg.get("code_bindings", {})
    require(bindings.get("producer_raw_sha256") == sha256_file(Path(__file__)), "producer binding drift")
    if reference_path is not None:
        require(bindings.get("reference_raw_sha256") == sha256_file(reference_path), "reference binding drift")


def _resolve_native_query_scale(query: torch.Tensor, kernel: Any, kwargs: dict[str, Any]) -> tuple[float, str]:
    explicit = kwargs.get("scale", kwargs.get("query_scale"))
    if explicit is not None:
        require(isinstance(explicit, (int, float)), "GDN query scale is not numeric")
        return float(explicit), "explicit_call_keyword"
    qualname = str(getattr(kernel, "__qualname__", getattr(kernel, "__name__", "")))
    require(qualname.endswith("torch_chunk_gated_delta_rule"), "unknown implicit-scale GDN kernel")
    return float(int(query.shape[-1]) ** -0.5), "native_default_inverse_sqrt_key_width"


def _capture_reference_qk_inputs(
    query: torch.Tensor, key: torch.Tensor, kernel: Any, use_norm: bool
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    require(use_norm is True, "expanded sweep requires native q/k L2 normalization")
    kernel_globals = getattr(kernel, "__globals__", None)
    require(isinstance(kernel_globals, dict), "native GDN kernel globals unavailable")
    native_l2norm = kernel_globals.get("l2norm")
    require(callable(native_l2norm), "native GDN l2norm helper unavailable")
    with torch.no_grad():
        q = native_l2norm(query.detach().clone(), dim=-1, eps=1e-6)
        k = native_l2norm(key.detach().clone(), dim=-1, eps=1e-6)
    return q.detach().clone(), k.detach().clone(), {
        "native_use_qk_l2norm_in_kernel": True,
        "reference_inputs_post_native_qk_l2norm": True,
        "query_key_capture_boundary": "post_native_qk_l2norm_pre_fp32_recurrence",
        "use_qk_l2norm_in_reference": False,
        "qk_preprocessor_module": str(getattr(native_l2norm, "__module__", "")),
        "qk_preprocessor_qualname": str(getattr(native_l2norm, "__qualname__", "")),
    }


class NativeGDNKernelCapture:
    def __init__(self, backbone: Any) -> None:
        self.backbone = backbone
        self.rows: dict[int, dict[str, Any]] = {}
        self.originals: list[tuple[Any, str, Any]] = []

    def __enter__(self) -> "NativeGDNKernelCapture":
        for layer_index in LINEAR_LAYERS:
            mixer = getattr(self.backbone.layers[layer_index], "linear_attn", None)
            require(mixer is not None, f"linear layer {layer_index} has no linear_attn")
            kernel_module = getattr(mixer, "chunk_gated_delta_rule", None)
            require(kernel_module is not None, f"linear layer {layer_index} exposes no chunk kernel")
            if hasattr(kernel_module, "forward"):
                owner, attribute, kernel = kernel_module, "forward", kernel_module.forward
            else:
                owner, attribute, kernel = mixer, "chunk_gated_delta_rule", kernel_module

            def wrapped(
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                g: torch.Tensor,
                beta: torch.Tensor,
                *args: Any,
                _layer_index: int = layer_index,
                _kernel: Any = kernel,
                **kwargs: Any,
            ) -> Any:
                require(not args, "native GDN kernel used unsupported positional extras")
                selected = _layer_index in SELECTED_GDN_LAYERS
                before: dict[str, Any] | None = None
                if selected:
                    initial = kwargs.get("initial_state")
                    require(isinstance(initial, torch.Tensor), "selected GDN transition lacks initial state")
                    q, k, boundary = _capture_reference_qk_inputs(
                        query, key, _kernel, kwargs.get("use_qk_l2norm_in_kernel", False)
                    )
                    before = {
                        "query": q,
                        "key": k,
                        "value": value.detach().clone(),
                        "g": g.detach().clone(),
                        "beta": beta.detach().clone(),
                        "initial_state": initial.detach().clone(),
                        "boundary": boundary,
                    }
                result = _kernel(query, key, value, g=g, beta=beta, **kwargs)
                if selected:
                    require(isinstance(result, (tuple, list)) and len(result) >= 2, "GDN kernel result schema drift")
                    require(_layer_index not in self.rows, f"GDN layer {_layer_index} invoked twice")
                    assert before is not None
                    scale, scale_source = _resolve_native_query_scale(query, _kernel, kwargs)
                    self.rows[_layer_index] = {
                        **before,
                        "candidate_output": result[0].detach().clone(),
                        "candidate_state": result[1].detach().clone(),
                        "kernel_semantics": {
                            **before["boundary"],
                            "query_scale": scale,
                            "query_scale_source": scale_source,
                            "query_key_width": int(query.shape[-1]),
                            "output_final_state": kwargs.get("output_final_state"),
                            "callable_module": str(getattr(_kernel, "__module__", "")),
                            "callable_qualname": str(getattr(_kernel, "__qualname__", "")),
                            "callable_signature": str(inspect.signature(_kernel)),
                        },
                    }
                return result

            self.originals.append((owner, attribute, kernel))
            setattr(owner, attribute, wrapped)
        require(len(self.originals) == len(LINEAR_LAYERS), "GDN hook coverage drift")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for owner, attribute, kernel in self.originals:
            setattr(owner, attribute, kernel)
        self.originals.clear()


@dataclass
class AppendCollector:
    row_id: str
    event: dict[str, Any] | None = None

    def __call__(self, event: Mapping[str, Any]) -> str:
        require(self.event is None, f"{self.row_id} append observed twice")
        require(event.get("append_event_index") == 0, f"{self.row_id} append index drift")
        key = event.get("key_states")
        value = event.get("value_states")
        require(isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor) and key.shape == value.shape, "append K/V missing")
        capture_id = f"{self.row_id}-append-0"
        self.event = {
            "capture_id": capture_id,
            "key": key.detach().contiguous().cpu().clone(),
            "value": value.detach().contiguous().cpu().clone(),
            "append_event_index": 0,
            "sequence_length_before": int(event["sequence_length_before"]),
            "sequence_length_after": int(event["sequence_length_after"]),
            "appended_tokens_before": int(event["appended_tokens_before"]),
            "appended_tokens_after": int(event["appended_tokens_after"]),
        }
        return capture_id


@dataclass
class AttentionCollector:
    append_collectors: Mapping[int, AppendCollector]
    rows: dict[int, dict[str, Any]]

    def __call__(self, event: Mapping[str, Any]) -> None:
        layer = int(event["layer_idx"])
        require(layer in FULL_LAYERS and layer not in self.rows, "attention call selection/order drift")
        append = self.append_collectors[layer].event
        require(append is not None and event.get("append_capture_id") == append["capture_id"], "attention/append binding drift")
        query = event.get("query_cpu")
        candidate = event.get("candidate_output_cpu")
        positions = event.get("position_ids_cpu")
        require(
            event.get("observer_schema") == "qcomem-forkaudit-call-observer-v2"
            and event.get("attention_mask_is_none") is True
            and isinstance(query, torch.Tensor)
            and isinstance(candidate, torch.Tensor)
            and isinstance(positions, torch.Tensor),
            "attention boundary capture missing",
        )
        self.rows[layer] = {
            "query": query.detach().contiguous().clone(),
            "candidate_output": candidate.detach().contiguous().clone(),
            "query_positions": positions.detach().contiguous().reshape(-1).clone(),
            "effective_scaling": float(event["effective_scaling"]),
            "kernel_audit": dict(event["kernel_audit"]),
            "append_capture_id": str(event["append_capture_id"]),
        }


def _persistent_document_kv(persistent: Any, layer_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    arena = persistent.layers[layer_index].arena
    table = arena.document_block_table.detach().cpu()
    batches_key: list[torch.Tensor] = []
    batches_value: list[torch.Tensor] = []
    for batch_index in range(int(arena.batch_size)):
        block_ids = [int(value) for value in table[batch_index].reshape(-1)]
        key = torch.cat([arena.key_cache[physical].detach() for physical in block_ids], dim=0)[: int(arena.document_length)]
        value = torch.cat([arena.value_cache[physical].detach() for physical in block_ids], dim=0)[: int(arena.document_length)]
        batches_key.append(key.permute(1, 0, 2).contiguous().cpu())
        batches_value.append(value.permute(1, 0, 2).contiguous().cpu())
    return torch.stack(batches_key), torch.stack(batches_value)


def capture(args: argparse.Namespace) -> dict[str, Any]:
    prereg_raw = args.preregistration.read_bytes()
    prereg = json.loads(prereg_raw)
    _validate_preregistration(prereg, input_manifest_path=args.input_manifest)
    require(sha256_file(args.model_weight_ledger) == MODEL_WEIGHT_LEDGER_SHA256, "model weight ledger SHA drift")
    frozen_input_manifest = json.loads(args.input_manifest.read_bytes())
    inputs, rebuilt_manifest = _build_inputs(
        model_dir=args.model_dir, pg19_data=args.pg19_data, pg19_manifest=args.pg19_manifest
    )
    require(rebuilt_manifest == frozen_input_manifest, "rebuilt input manifest differs from frozen manifest")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "3", "candidate must expose physical GPU3 only")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "candidate process must see exactly one GPU")
    torch.cuda.set_device(0)

    from qcomem_qwen35_functional_stack import audit_qwen35_functional_stack_plan
    from qcomem_qwen35_vllm_paged_integration import convert_all_qwen35_full_layers_to_vllm_q16
    from qcomem_vllm_paged_fair_control import SHARED_REUSE
    from qcomem_vllm_paged_kernel import _resolve_vllm_unified_attention, audit_frozen_kernel_environment
    from qcomem_vllm_paged_multifork_resident import (
        GDN_BORROW_IMMUTABLE_BASE,
        MultiForkHitLedger,
        build_resident_request_group,
        register_multifork_backend,
    )
    from run_qcomem_qwen35_vllm_paged_multifork_resident import _build_document_cache, _resolve_backbone, _unregister_backend
    from transformers import AutoModelForImageTextToText

    identity = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name,memory.total", "--format=csv,noheader,nounits"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    ).stdout.strip().splitlines()[3]
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().cuda()
    backbone = _resolve_backbone(model)
    plan = audit_qwen35_functional_stack_plan(model)
    require(tuple(plan.full_attention_layer_indices) == FULL_LAYERS, "full-layer plan drift")
    require(tuple(plan.linear_layer_indices) == LINEAR_LAYERS, "linear-layer plan drift")
    kernel_environment = audit_frozen_kernel_environment()
    require(kernel_environment.get("matches_frozen_environment") is True, "frozen kernel environment drift")
    kernel = _resolve_vllm_unified_attention()

    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=False)
    attention_rows: list[dict[str, Any]] = []
    gdn_rows: list[dict[str, Any]] = []
    case_receipts: list[dict[str, Any]] = []
    with torch.inference_mode():
        for case in inputs:
            case_id = str(case["case_id"])
            document = case["document"].to(device="cuda:0", dtype=torch.int64)
            query = case["query"].to(device="cuda:0", dtype=torch.int64)
            persistent = _build_document_cache(backbone, document)
            conversion = convert_all_qwen35_full_layers_to_vllm_q16(
                persistent, plan, page_size=PAGE_SIZE, max_append_tokens=QUERY_TOKENS, max_request_forks=1
            )
            require(conversion.max_request_forks == 1, "conversion fork count drift")
            group = build_resident_request_group(
                persistent,
                plan,
                resident_count=1,
                policy=SHARED_REUSE,
                gdn_base_policy=GDN_BORROW_IMMUTABLE_BASE,
            )
            request = group.requests[0]
            append_collectors: dict[int, AppendCollector] = {}
            for layer in FULL_LAYERS:
                sequence = request.layers[layer].sequence
                sequence.strict_mask_check = False
                collector = AppendCollector(f"ATTN-{case_id}-L{layer:02d}")
                sequence.append_observer = collector
                append_collectors[layer] = collector
            attention_observer = AttentionCollector(append_collectors, {})
            ledger = MultiForkHitLedger(
                plan,
                request,
                request_index=0,
                resident_count=1,
                request_policy=SHARED_REUSE,
                expected_calls_per_layer=1,
                initial_query_tokens=QUERY_TOKENS,
                kernel=kernel,
                strict_position_values=True,
                call_observer=attention_observer,
            )
            backend = register_multifork_backend(ledger)
            original_attention = backbone.config._attn_implementation
            try:
                with NativeGDNKernelCapture(backbone) as gdn_observer:
                    backbone.config._attn_implementation = backend
                    output = backbone(input_ids=query, past_key_values=request, use_cache=True)
                    require(output.past_key_values is request, "query returned a different cache")
                    torch.cuda.synchronize()
                dispatch_receipt = ledger.verify_complete()
            finally:
                backbone.config._attn_implementation = original_attention
                _unregister_backend(backend)
                for layer in FULL_LAYERS:
                    request.layers[layer].sequence.append_observer = None
            require(tuple(sorted(attention_observer.rows)) == FULL_LAYERS, "attention row coverage drift")
            require(tuple(sorted(gdn_observer.rows)) == SELECTED_GDN_LAYERS, "GDN row coverage drift")

            for layer in FULL_LAYERS:
                row_id = f"ATTN-{case_id}-L{layer:02d}"
                observed = attention_observer.rows[layer]
                append = append_collectors[layer].event
                assert append is not None
                document_key, document_value = _persistent_document_kv(persistent, layer)
                key = torch.cat((document_key, append["key"]), dim=2)
                value = torch.cat((document_value, append["value"]), dim=2)
                positions = observed["query_positions"].to(torch.int64)
                require(list(positions.tolist()) == list(range(DOCUMENT_TOKENS, DOCUMENT_TOKENS + QUERY_TOKENS)), f"{row_id} query positions drift")
                arrays = {
                    name: _save_array(output_root, row_id, name, tensor)
                    for name, tensor in {
                        "query": observed["query"],
                        "key": key,
                        "value": value,
                        "candidate_output": observed["candidate_output"],
                    }.items()
                }
                attention_rows.append(
                    {
                        "row_id": row_id,
                        "case_id": case_id,
                        "layer_index": layer,
                        "query_positions": [int(item) for item in positions.tolist()],
                        "key_positions": list(range(DOCUMENT_TOKENS + QUERY_TOKENS)),
                        "softmax_scale": float(observed["effective_scaling"]),
                        "append_capture_id": observed["append_capture_id"],
                        "arrays": arrays,
                    }
                )
            for layer in SELECTED_GDN_LAYERS:
                row_id = f"GDN-{case_id}-L{layer:02d}"
                observed = gdn_observer.rows[layer]
                arrays = {
                    name: _save_array(output_root, row_id, name, observed[name])
                    for name in (
                        "query", "key", "value", "g", "beta", "initial_state",
                        "candidate_output", "candidate_state",
                    )
                }
                semantics = dict(observed["kernel_semantics"])
                require(semantics["query_scale"] == 128 ** -0.5, f"{row_id} query scale drift")
                require(semantics["output_final_state"] is True, f"{row_id} final state missing")
                gdn_rows.append(
                    {
                        "row_id": row_id,
                        "case_id": case_id,
                        "layer_index": layer,
                        "kernel_semantics": semantics,
                        "arrays": arrays,
                    }
                )
            case_receipts.append(
                {
                    "case_id": case_id,
                    "attention_rows": len(FULL_LAYERS),
                    "gdn_rows": len(SELECTED_GDN_LAYERS),
                    "dispatch_verified": bool(dispatch_receipt["verified"]),
                    "candidate_full_attention_calls": int(dispatch_receipt["total_calls"]),
                }
            )
            del output, request, group, persistent
            torch.cuda.empty_cache()

    result = {
        "schema_version": CAPTURE_SCHEMA,
        "status": "captured-no-numerical-pass-fields",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_weight_ledger_raw_sha256": MODEL_WEIGHT_LEDGER_SHA256,
        "producer_raw_sha256": sha256_file(Path(__file__)),
        "preregistration_raw_sha256": sha256_bytes(prereg_raw),
        "input_manifest_raw_sha256": sha256_file(args.input_manifest),
        "hardware": {"cuda_visible_devices": "3", "physical_gpu3": identity},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "numpy": np.__version__,
            "kernel_environment": kernel_environment,
        },
        "coverage": {
            "input_cases": len(CASE_SPECS),
            "attention_layers_per_case": len(FULL_LAYERS),
            "attention_rows": len(attention_rows),
            "attention_query_positions": len(attention_rows) * QUERY_TOKENS,
            "gdn_layers_per_case": len(SELECTED_GDN_LAYERS),
            "gdn_rows": len(gdn_rows),
            "gdn_token_transitions": len(gdn_rows) * QUERY_TOKENS,
        },
        "case_receipts": case_receipts,
        "attention_rows": attention_rows,
        "gdn_rows": gdn_rows,
        "claim_boundary": (
            "captured post-RoPE full-attention and post-native-q/k-normalization recurrent-core boundaries only; "
            "capture honesty, projections, convolution, normalization, downstream logits, and end-to-end semantics remain outside scope"
        ),
    }
    atomic_json(output_root / "capture-manifest.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--model-dir", type=Path, required=True)
    prepare_parser.add_argument("--pg19-data", type=Path, required=True)
    prepare_parser.add_argument("--pg19-manifest", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    capture_parser = sub.add_parser("capture")
    capture_parser.add_argument("--model-dir", type=Path, required=True)
    capture_parser.add_argument("--model-weight-ledger", type=Path, required=True)
    capture_parser.add_argument("--pg19-data", type=Path, required=True)
    capture_parser.add_argument("--pg19-manifest", type=Path, required=True)
    capture_parser.add_argument("--input-manifest", type=Path, required=True)
    capture_parser.add_argument("--preregistration", type=Path, required=True)
    capture_parser.add_argument("--output-dir", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    result = prepare(args) if args.command == "prepare" else capture(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
