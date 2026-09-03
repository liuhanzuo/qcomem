from __future__ import annotations

"""Capture and independently replay selected live Qwen3.5 GDN transitions."""

import argparse
import hashlib
import inspect
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


CAPTURE_SCHEMA = "qcomem-gdn-transition-capture-v1"
MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
MODEL_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
FULL_LAYERS = tuple(range(3, 40, 4))
LINEAR_LAYERS = tuple(index for index in range(40) if index not in FULL_LAYERS)
SELECTED_LAYERS = (0, 10, 20, 38)
DOCUMENT_TOKENS = 128
QUERY_TOKENS = 4
PAGE_SIZE = 128
PG19_DATA_SHA256 = "ef18893b6bfb0f4b8cb29eab85ccf2a0ef1fdb44606e9742a70405cca564e18c"
PG19_MANIFEST_SHA256 = "5d789d67aa239f089e92de8a4267b86d2f1d2723d5f1370970883738f5f89a9c"
WINDOWS_SHA256 = "cca6c3643b3e77178cac3c906ef0202c686a23de9578287adef95a5d8af16aa9"


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
    payload = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return sha256_bytes(payload)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def _resolve_native_query_scale(
    query: torch.Tensor,
    kernel: Any,
    kwargs: dict[str, Any],
) -> tuple[float, str]:
    """Resolve the scale actually applied by the captured native kernel.

    Qwen3.5's ``torch_chunk_gated_delta_rule`` applies ``1/sqrt(K)`` after
    optional q/k normalization even though the scale is not exposed as a call
    argument.  Treating the absent keyword as 1.0 records the call boundary
    incorrectly and makes an otherwise matching independent output replay
    differ by ``1 - 1/sqrt(K)``.  Unknown implicit-scale kernels fail closed.
    """

    explicit = kwargs.get("scale", kwargs.get("query_scale"))
    if explicit is not None:
        require(isinstance(explicit, (int, float)), "native query scale is not numeric")
        return float(explicit), "explicit_call_keyword"
    qualname = str(getattr(kernel, "__qualname__", getattr(kernel, "__name__", "")))
    require(
        qualname.endswith("torch_chunk_gated_delta_rule"),
        "unrecognized native kernel with implicit query scale",
    )
    key_width = int(query.shape[-1])
    require(key_width > 0, "native query key width must be positive")
    return float(key_width**-0.5), "native_default_inverse_sqrt_key_width"


def _capture_reference_qk_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    kernel: Any,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Freeze the effective q/k tensors at the recurrent-core boundary.

    The native Qwen3.5 kernel applies its l2 normalization while q/k are still
    BF16, before converting the recurrence inputs to FP32.  Re-normalizing the
    serialized FP32 values in NumPy is therefore a different floating-point
    program.  Capture the native preprocessor's outputs explicitly and keep
    normalization outside the independent recurrence claim.
    """

    require(isinstance(use_qk_l2norm_in_kernel, bool), "native q/k normalization flag is not bool")
    if not use_qk_l2norm_in_kernel:
        return query.detach().clone(), key.detach().clone(), {
            "native_use_qk_l2norm_in_kernel": False,
            "reference_inputs_post_native_qk_l2norm": False,
            "query_key_capture_boundary": "raw_native_kernel_arguments",
        }
    kernel_globals = getattr(kernel, "__globals__", None)
    require(isinstance(kernel_globals, dict), "native kernel globals are unavailable")
    native_l2norm = kernel_globals.get("l2norm")
    require(callable(native_l2norm), "native q/k l2norm helper is unavailable")
    with torch.no_grad():
        reference_query = native_l2norm(query.detach().clone(), dim=-1, eps=1e-6)
        reference_key = native_l2norm(key.detach().clone(), dim=-1, eps=1e-6)
    for name, captured, original in (
        ("query", reference_query, query),
        ("key", reference_key, key),
    ):
        require(isinstance(captured, torch.Tensor), f"native normalized {name} is not tensor")
        require(captured.shape == original.shape, f"native normalized {name} shape drift")
        require(captured.dtype == original.dtype, f"native normalized {name} dtype drift")
        require(captured.device == original.device, f"native normalized {name} device drift")
    return reference_query.detach().clone(), reference_key.detach().clone(), {
        "native_use_qk_l2norm_in_kernel": True,
        "reference_inputs_post_native_qk_l2norm": True,
        "query_key_capture_boundary": "post_native_qk_l2norm_pre_fp32_recurrence",
        "qk_preprocessor_module": str(
            getattr(native_l2norm, "__module__", type(native_l2norm).__module__)
        ),
        "qk_preprocessor_qualname": str(
            getattr(native_l2norm, "__qualname__", type(native_l2norm).__qualname__)
        ),
        "qk_preprocessor_signature": str(inspect.signature(native_l2norm)),
    }


class NativeGDNKernelCapture:
    """Wrap each loaded module's native chunk kernel without changing its result."""

    def __init__(self, backbone: Any, selected_layers: tuple[int, ...]) -> None:
        self.backbone = backbone
        self.selected_layers = selected_layers
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
                selected = _layer_index in self.selected_layers
                before = None
                boundary_receipt = None
                if selected:
                    initial = kwargs.get("initial_state")
                    require(isinstance(initial, torch.Tensor), "selected query transition lacks an initial state")
                    use_norm = kwargs.get("use_qk_l2norm_in_kernel", False)
                    reference_query, reference_key, boundary_receipt = (
                        _capture_reference_qk_inputs(query, key, _kernel, use_norm)
                    )
                    before = {
                        "raw_query": query.detach().clone(),
                        "raw_key": key.detach().clone(),
                        "query": reference_query,
                        "key": reference_key,
                        "value": value.detach().clone(),
                        "g": g.detach().clone(),
                        "beta": beta.detach().clone(),
                        "initial_state": initial.detach().clone(),
                    }
                result = _kernel(query, key, value, g=g, beta=beta, **kwargs)
                if selected:
                    require(
                        isinstance(result, (tuple, list)) and len(result) >= 2,
                        "native GDN kernel did not return output and final state",
                    )
                    candidate_output, candidate_state = result[0], result[1]
                    require(isinstance(candidate_output, torch.Tensor), "candidate GDN output is not tensor")
                    require(isinstance(candidate_state, torch.Tensor), "candidate GDN final state is not tensor")
                    require(_layer_index not in self.rows, f"selected layer {_layer_index} invoked twice")
                    assert before is not None
                    assert boundary_receipt is not None
                    use_norm = kwargs.get("use_qk_l2norm_in_kernel", False)
                    require(isinstance(use_norm, bool), "native q/k normalization flag is not bool")
                    query_scale, query_scale_source = _resolve_native_query_scale(
                        query, _kernel, kwargs
                    )
                    self.rows[_layer_index] = {
                        **before,
                        "candidate_output": candidate_output.detach().clone(),
                        "candidate_state": candidate_state.detach().clone(),
                        "kernel_semantics": {
                            **boundary_receipt,
                            "use_qk_l2norm_in_kernel": False,
                            "query_scale": float(query_scale),
                            "query_scale_source": query_scale_source,
                            "query_key_width": int(query.shape[-1]),
                            "output_final_state": kwargs.get("output_final_state"),
                            "callable_module": str(getattr(_kernel, "__module__", type(_kernel).__module__)),
                            "callable_qualname": str(getattr(_kernel, "__qualname__", type(_kernel).__qualname__)),
                            "callable_signature": str(inspect.signature(_kernel)),
                            "keyword_names": sorted(kwargs),
                        },
                    }
                return result

            self.originals.append((owner, attribute, kernel))
            setattr(owner, attribute, wrapped)
        require(len(self.originals) == 30, "native GDN hook coverage differs from 30 layers")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for owner, attribute, kernel in self.originals:
            setattr(owner, attribute, kernel)
        self.originals.clear()


def _build_inputs(args: argparse.Namespace) -> tuple[Any, torch.Tensor, torch.Tensor, dict[str, Any]]:
    from qcomem_joint_policy import audit_pg19_train_calibration, build_pg19_calibration_windows
    from qcomem_vllm_paged_multifork_resident import build_pg19_train_query_bank
    from transformers import AutoTokenizer

    require(sha256_file(args.pg19_data) == PG19_DATA_SHA256, "PG19 data SHA drift")
    require(sha256_file(args.pg19_manifest) == PG19_MANIFEST_SHA256, "PG19 manifest SHA drift")
    records, data_audit = audit_pg19_train_calibration(
        args.pg19_data,
        args.pg19_manifest,
        expected_data_sha256=PG19_DATA_SHA256,
        expected_manifest_sha256=PG19_MANIFEST_SHA256,
        minimum_books=8,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
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
    require(windows_sha == WINDOWS_SHA256, "frozen lifecycle window selection drift")
    window = windows[0]
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
    query = queries[0][:, :QUERY_TOKENS].contiguous()
    require(tuple(document.shape) == (1, DOCUMENT_TOKENS), "document slice shape drift")
    require(tuple(query.shape) == (1, QUERY_TOKENS), "query slice shape drift")
    return tokenizer, document, query, {
        "data_audit": data_audit,
        "windows_sha256": windows_sha,
        "source_object": window.source_object,
        "source_id": str(window.source_id),
        "full_window_document_sha256": sha256_tensor(window.document_ids),
        "full_query_bank_audit": query_audit,
        "document_token_ids_sha256": sha256_tensor(document),
        "query_token_ids_sha256": sha256_tensor(query),
    }


def capture(args: argparse.Namespace) -> dict[str, Any]:
    prereg_raw = args.preregistration.read_bytes()
    prereg = json.loads(prereg_raw)
    amendment_raw = args.scope_amendment.read_bytes()
    amendment = json.loads(amendment_raw)
    require(prereg.get("schema_version") == "qcomem-gdn-transition-preregistration-v1", "prereg schema drift")
    require(sha256_bytes(amendment_raw) == args.expected_scope_amendment_sha256, "scope amendment SHA drift")
    require(amendment.get("schema_version") == "qcomem-gdn-transition-oracle-rerun-amendment-v1", "scope amendment schema drift")
    require(amendment.get("parent_preregistration_raw_sha256") == sha256_bytes(prereg_raw), "scope amendment/prereg binding drift")
    require(all(amendment.get("unchanged", {}).values()), "scope amendment changed a preregistered coordinate or threshold")
    require(amendment.get("corrected_runner_sha256") == sha256_file(Path(__file__)), "scope amendment/runner binding drift")
    require(tuple(row["layer_index"] for row in prereg["selected_rows"]) == SELECTED_LAYERS, "selected layer drift")
    require(prereg["document_tokens"] == DOCUMENT_TOKENS, "document token count drift")
    require(prereg["query_tokens"] == QUERY_TOKENS, "query token count drift")
    require(sha256_file(args.model_weight_ledger) == args.expected_model_weight_ledger_sha256, "model weight ledger SHA drift")
    _tokenizer, document_cpu, query_cpu, input_receipt = _build_inputs(args)

    torch.cuda.set_device(0)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    identity = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name,memory.total", "--format=csv,noheader,nounits"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    ).stdout.strip().splitlines()[0]
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
    from run_qcomem_qwen35_vllm_paged_multifork_resident import (
        _build_document_cache,
        _resolve_backbone,
        _unregister_backend,
    )
    from transformers import AutoModelForImageTextToText

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
    require(tuple(plan.full_attention_layer_indices) == FULL_LAYERS, "full layer plan drift")
    require(tuple(plan.linear_layer_indices) == LINEAR_LAYERS, "linear layer plan drift")
    kernel_environment = audit_frozen_kernel_environment()
    require(kernel_environment.get("matches_frozen_environment") is True, "frozen kernel environment drift")
    kernel = _resolve_vllm_unified_attention()
    document = document_cpu.to(device="cuda:0", dtype=torch.int64)
    query = query_cpu.to(device="cuda:0", dtype=torch.int64)
    with torch.inference_mode():
        persistent = _build_document_cache(backbone, document)
        conversion = convert_all_qwen35_full_layers_to_vllm_q16(
            persistent,
            plan,
            page_size=PAGE_SIZE,
            max_append_tokens=QUERY_TOKENS,
            max_request_forks=1,
        )
        require(conversion.max_request_forks == 1, "oracle conversion fork count drift")
        group = build_resident_request_group(
            persistent,
            plan,
            resident_count=1,
            policy=SHARED_REUSE,
            gdn_base_policy=GDN_BORROW_IMMUTABLE_BASE,
        )
        request = group.requests[0]
        for layer_index in FULL_LAYERS:
            request.layers[layer_index].sequence.strict_mask_check = False
        ledger = MultiForkHitLedger(
            plan,
            request,
            request_index=0,
            resident_count=1,
            request_policy=SHARED_REUSE,
            expected_calls_per_layer=1,
            initial_query_tokens=QUERY_TOKENS,
            kernel=kernel,
        )
        backend = register_multifork_backend(ledger)
        original_attention = backbone.config._attn_implementation
        try:
            with NativeGDNKernelCapture(backbone, SELECTED_LAYERS) as observer:
                backbone.config._attn_implementation = backend
                output = backbone(input_ids=query, past_key_values=request, use_cache=True)
                require(output.past_key_values is request, "query returned a different cache")
                torch.cuda.synchronize()
            dispatch_receipt = ledger.verify_complete()
        finally:
            backbone.config._attn_implementation = original_attention
            _unregister_backend(backend)
    require(tuple(sorted(observer.rows)) == SELECTED_LAYERS, "selected native GDN rows were not all observed")

    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=False)
    rows = []
    for selected in prereg["selected_rows"]:
        layer_index = int(selected["layer_index"])
        captured = observer.rows[layer_index]
        expected_shapes = {
            "raw_query": [1, QUERY_TOKENS, 32, 128],
            "raw_key": [1, QUERY_TOKENS, 32, 128],
            "query": [1, QUERY_TOKENS, 32, 128],
            "key": [1, QUERY_TOKENS, 32, 128],
            "value": [1, QUERY_TOKENS, 32, 128],
            "g": [1, QUERY_TOKENS, 32],
            "beta": [1, QUERY_TOKENS, 32],
            "initial_state": [1, 32, 128, 128],
            "candidate_output": [1, QUERY_TOKENS, 32, 128],
            "candidate_state": [1, 32, 128, 128],
        }
        arrays = {}
        for name, shape in expected_shapes.items():
            require(list(captured[name].shape) == shape, f"{selected['row_id']} {name} shape drift")
            arrays[name] = _save_array(output_root, selected["row_id"], name, captured[name])
        semantics = captured["kernel_semantics"]
        require(semantics["native_use_qk_l2norm_in_kernel"] is True, "native q/k normalization semantics drift")
        require(semantics["reference_inputs_post_native_qk_l2norm"] is True, "reference q/k capture boundary drift")
        require(semantics["use_qk_l2norm_in_kernel"] is False, "reference would normalize q/k twice")
        require(semantics["query_scale"] == 128**-0.5, "native query scale semantics drift")
        require(semantics["query_scale_source"] == "native_default_inverse_sqrt_key_width", "native query scale source drift")
        require(semantics["output_final_state"] is True, "native kernel did not emit final state")
        rows.append(
            {
                "row_id": selected["row_id"],
                "layer_index": layer_index,
                "arrays": arrays,
                "kernel_semantics": semantics,
            }
        )
    result = {
        "schema_version": CAPTURE_SCHEMA,
        "status": "captured-no-pass-field",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_weight_ledger_raw_sha256": args.expected_model_weight_ledger_sha256,
        "preregistration_raw_sha256": sha256_bytes(prereg_raw),
        "scope_amendment_raw_sha256": sha256_bytes(amendment_raw),
        "runner_raw_sha256": sha256_file(Path(__file__)),
        "input_receipt": input_receipt,
        "hardware": {"cuda_visible_devices": visible, "nvidia_smi": identity},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "numpy": np.__version__,
            "kernel_environment": kernel_environment,
        },
        "dispatch_receipt": dispatch_receipt,
        "rows": rows,
    }
    atomic_json(output_root / "capture-manifest.json", result)
    return result


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    reference_source = Path(args.reference_module)
    source_text = reference_source.read_text(encoding="utf-8")
    require("import torch" not in source_text, "reference module imports torch")
    require("from qcomem" not in source_text and "import qcomem" not in source_text, "reference module imports candidate code")
    from qcomem_gdn_transition_oracle_reference import evaluate_capture

    amendment_raw = args.scope_amendment.read_bytes()
    amendment = json.loads(amendment_raw)
    require(sha256_bytes(amendment_raw) == args.expected_scope_amendment_sha256, "scope amendment SHA drift")
    require(amendment.get("parent_preregistration_raw_sha256") == sha256_file(args.preregistration), "scope amendment/prereg binding drift")
    require(amendment.get("corrected_runner_sha256") == sha256_file(Path(__file__)), "scope amendment/aggregator binding drift")
    require(amendment.get("corrected_reference_sha256") == sha256_file(reference_source), "scope amendment/reference binding drift")
    result = evaluate_capture(args.capture_manifest, args.preregistration, args.output)
    require(result.get("capture_manifest_raw_sha256") is not None, "reference omitted capture binding")
    capture = json.loads(args.capture_manifest.read_bytes())
    require(capture.get("scope_amendment_raw_sha256") == sha256_bytes(amendment_raw), "capture/scope amendment binding drift")
    result["reference_source_raw_sha256"] = sha256_file(reference_source)
    result["aggregator_runner_raw_sha256"] = sha256_file(Path(__file__))
    atomic_json(args.output, result)
    require(result["all_clean_rows_pass"] is True, "one or more clean GDN oracle rows failed")
    require(result["all_seeded_wrong_transitions_rejected"] is True, "one or more seeded wrong transitions escaped")
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    capture_parser = sub.add_parser("capture")
    capture_parser.add_argument("--model-dir", type=Path, required=True)
    capture_parser.add_argument("--model-weight-ledger", type=Path, required=True)
    capture_parser.add_argument("--expected-model-weight-ledger-sha256", required=True)
    capture_parser.add_argument("--pg19-data", type=Path, required=True)
    capture_parser.add_argument("--pg19-manifest", type=Path, required=True)
    capture_parser.add_argument("--preregistration", type=Path, required=True)
    capture_parser.add_argument("--scope-amendment", type=Path, required=True)
    capture_parser.add_argument("--expected-scope-amendment-sha256", required=True)
    capture_parser.add_argument("--output-dir", type=Path, required=True)
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--capture-manifest", type=Path, required=True)
    aggregate_parser.add_argument("--preregistration", type=Path, required=True)
    aggregate_parser.add_argument("--scope-amendment", type=Path, required=True)
    aggregate_parser.add_argument("--expected-scope-amendment-sha256", required=True)
    aggregate_parser.add_argument("--reference-module", type=Path, required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    result = capture(args) if args.command == "capture" else aggregate(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
