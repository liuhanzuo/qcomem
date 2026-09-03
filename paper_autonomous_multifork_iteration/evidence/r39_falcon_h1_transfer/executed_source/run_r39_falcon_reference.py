#!/usr/bin/env python3
"""Candidate-import-free official DynamicCache reference for Falcon-H1 R39."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import platform
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import torch


PROTOCOL = "forkaudit-falcon-h1-hybrid-transformers-transfer-v1"
MODEL_ID = "tiiuae/Falcon-H1-0.5B-Base"
HF_MODEL_REVISION = "59fb76e8c5d3fc7441b062be638e1ba0afd5c687"
MODELSCOPE_REVISION = "a475c769e108fd1dc6cfe41e342305d36431ef20"
WORLD_SIZE = 8
VOCAB_SIZE = 32784
FAMILY_ORDER = ("kv_key", "kv_value", "conv", "mamba2_recurrent")
OFFICIAL_MODELING_SHA256 = "e90bf774524e9b66284ad1c5528c35339271a187f58f16ba2d45c97f4bc6b5bd"
OFFICIAL_CACHE_UTILS_SHA256 = "ee7902fbd031ed332b5e26d07756a33f09b5c90a435b8363b9330876dc33ce0e"
OFFICIAL_MASKING_UTILS_SHA256 = "5f48e428ea02d1b6008acb45c147fcdb4eba89deea69627744662aa05da1b9f2"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_bound_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    require(sha256_bytes(raw) == expected_sha256, f"{label} raw SHA-256 drift")
    value = json.loads(raw)
    require(isinstance(value, dict), f"{label} is not an object")
    return value


def int64_le_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(dtype=torch.int64, device="cpu").contiguous().view(-1)
    digest = hashlib.sha256()
    for item in value.tolist():
        digest.update(struct.pack("<q", int(item)))
    return digest.hexdigest()


def tensor_raw_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().contiguous()
    return b"" if value.numel() == 0 else value.view(torch.uint8).cpu().numpy().tobytes(order="C")


def family_row(layer_index: int, family: str, tensor: torch.Tensor) -> dict[str, Any]:
    require(isinstance(tensor, torch.Tensor) and tensor.numel() > 0, "reference state tensor absent")
    require(tensor.is_contiguous(), "reference state tensor is non-contiguous")
    return {
        "layer_index": layer_index,
        "family": family,
        "state_index": 0 if family in {"conv", "mamba2_recurrent"} else None,
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "content_sha256": sha256_bytes(tensor_raw_bytes(tensor)),
    }


def expected_shape(family: str, sequence_length: int) -> list[int]:
    return {
        "kv_key": [1, 2, sequence_length, 64],
        "kv_value": [1, 2, sequence_length, 64],
        "conv": [1, 1792, 4],
        "mamba2_recurrent": [1, 24, 64, 128],
    }[family]


def official_cache_family_receipt(cache: Any, sequence_length: int) -> dict[str, Any]:
    layers = getattr(cache, "layers", None)
    require(isinstance(layers, list) and len(layers) == 36, "official cache geometry drift")
    rows = []
    for layer_index, layer in enumerate(layers):
        require(set(getattr(layer, "conv_states", {})) == {0}, "official conv index drift")
        require(set(getattr(layer, "recurrent_states", {})) == {0}, "official recurrent index drift")
        require(getattr(layer, "is_conv_states_initialized", {}).get(0) is True, "official conv absent")
        require(
            getattr(layer, "is_recurrent_states_initialized", {}).get(0) is True,
            "official recurrent absent",
        )
        require(getattr(layer, "has_previous_state", {}).get(0) is True, "official previous-state flag absent")
        tensors = {
            "kv_key": getattr(layer, "keys", None),
            "kv_value": getattr(layer, "values", None),
            "conv": layer.conv_states[0],
            "mamba2_recurrent": layer.recurrent_states[0],
        }
        for family in FAMILY_ORDER:
            row = family_row(layer_index, family, tensors[family])
            require(row["shape"] == expected_shape(family, sequence_length), "official family shape drift")
            dtype = "torch.float32" if family == "mamba2_recurrent" else "torch.bfloat16"
            require(row["dtype"] == dtype, "official family dtype drift")
            rows.append(row)
    expected = [(layer, family) for layer in range(36) for family in FAMILY_ORDER]
    require([(row["layer_index"], row["family"]) for row in rows] == expected, "reference census drift")
    return {
        "schema_version": "r39-falcon-h1-composed-state-family-receipt-v1",
        "split_depth": 18,
        "expected_sequence_length": sequence_length,
        "expected_family_count": 144,
        "observed_family_count": len(rows),
        "complete": True,
        "rows": rows,
        "rows_sha256": sha256_bytes(canonical_bytes(rows)),
    }


class LogitBundle:
    def __init__(self) -> None:
        self.payload = bytearray()
        self.records: list[dict[str, Any]] = []
        self.ids: set[str] = set()

    def add(self, record_id: str, logits: torch.Tensor) -> dict[str, Any]:
        require(record_id not in self.ids, "duplicate reference record ID")
        value = logits.detach().float().cpu().contiguous()
        require(tuple(value.shape) == (1, VOCAB_SIZE), "reference full-vocabulary shape drift")
        require(bool(torch.isfinite(value).all().item()), "non-finite reference logits")
        raw = value.numpy().astype("<f4", copy=False).tobytes(order="C")
        record = {
            "record_id": record_id,
            "offset_bytes": len(self.payload),
            "nbytes": len(raw),
            "shape": [1, VOCAB_SIZE],
            "dtype": "float32-le",
            "content_sha256": sha256_bytes(raw),
            "argmax": int(torch.argmax(value, dim=-1).item()),
        }
        self.payload.extend(raw)
        self.records.append(record)
        self.ids.add(record_id)
        return record

    def write(self, path: Path) -> dict[str, Any]:
        require(self.records, "empty reference sidecar")
        raw = bytes(self.payload)
        atomic_write(path, raw)
        return {
            "schema_version": "r39-falcon-full-vocabulary-fp32-sidecar-v1",
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
            "record_count": len(self.records),
            "records": self.records,
            "terminal_closure": {
                "first_offset_bytes": 0,
                "last_end_offset_bytes": len(raw),
                "exact_byte_coverage": True,
            },
        }


def gpu_identity(expected_uuid: str) -> dict[str, Any]:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == expected_uuid, "GPU isolation drift")
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={expected_uuid}",
            "--query-gpu=uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    fields = [item.strip() for item in result.stdout.strip().split(",")]
    require(len(fields) == 3 and fields[0] == expected_uuid and "H20" in fields[1], "H20 identity drift")
    properties = torch.cuda.get_device_properties(0)
    require([properties.major, properties.minor] == [9, 0], "H20 compute capability drift")
    return {
        "uuid": fields[0],
        "name": fields[1],
        "total_memory_mib": int(fields[2]),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }


def force_naive_reference_dispatch() -> dict[str, Any]:
    import inspect

    require(
        os.environ.get("USE_HUB_KERNELS", "").upper() == "NO",
        "reference Hub kernels not disabled",
    )
    from transformers.models.falcon_h1 import modeling_falcon_h1

    require(hasattr(modeling_falcon_h1, "is_fast_path_available"), "reference dispatch flag absent")
    before = getattr(modeling_falcon_h1, "is_fast_path_available")
    require(isinstance(before, bool), "reference dispatch flag is not boolean")
    path = Path(inspect.getsourcefile(modeling_falcon_h1) or "")
    require(path.is_file() and sha256_file(path) == OFFICIAL_MODELING_SHA256, "reference modeling source drift")
    modeling_falcon_h1.is_fast_path_available = False
    return {
        "official_module": "transformers.models.falcon_h1.modeling_falcon_h1",
        "official_method": "FalconH1Mixer.torch_forward",
        "pre_force_fast_path_available": before,
        "fast_path_observed_after_force": bool(modeling_falcon_h1.is_fast_path_available),
        "modeling_source_sha256": sha256_file(path),
        "optional_module_presence": {
            name: importlib.util.find_spec(name) is not None
            for name in ("mamba_ssm", "causal_conv1d", "flash_attn")
        },
        "package_installs_optional_kernels": False,
        "use_hub_kernels_environment": os.environ["USE_HUB_KERNELS"],
    }


def verify_official_sources() -> dict[str, str]:
    import inspect

    from transformers import cache_utils, masking_utils
    from transformers.models.falcon_h1 import modeling_falcon_h1

    rows = {}
    for name, module, expected in (
        ("modeling_falcon_h1.py", modeling_falcon_h1, OFFICIAL_MODELING_SHA256),
        ("cache_utils.py", cache_utils, OFFICIAL_CACHE_UTILS_SHA256),
        ("masking_utils.py", masking_utils, OFFICIAL_MASKING_UTILS_SHA256),
    ):
        path = Path(inspect.getsourcefile(module) or "")
        require(path.is_file(), f"reference official source absent: {name}")
        digest = sha256_file(path)
        require(digest == expected, f"reference official source drift: {name}")
        rows[name] = digest
    return rows


def source_is_candidate_import_free(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    banned = ("falcon" + "_h1_adapter", "run_r39_falcon_" + "candidate", "qcomem" + "_torch")
    forbidden_import = any(
        any(name == token or name.startswith(token + ".") for token in banned)
        for name in imported
    )
    forbidden_call = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in {"__import__", "exec", "eval"}:
            forbidden_call = True
        if isinstance(node.func, ast.Attribute):
            dotted = []
            value: Any = node.func
            while isinstance(value, ast.Attribute):
                dotted.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name):
                dotted.append(value.id)
            name = ".".join(reversed(dotted))
            if name in {"importlib.import_module", "importlib.util.spec_from_file_location"}:
                forbidden_call = True
    return not forbidden_import and not forbidden_call


def geometry_receipt(model: Any) -> dict[str, Any]:
    config = model.config
    layers = model.model.layers
    layer_types = tuple(getattr(config, "layer_types", ()))
    components = [
        {
            "layer_index": index,
            "has_self_attention": hasattr(layer, "self_attn"),
            "has_mamba2_mixer": hasattr(layer, "mamba"),
            "has_feed_forward": hasattr(layer, "feed_forward"),
        }
        for index, layer in enumerate(layers)
    ]
    matches = (
        config.model_type == "falcon_h1"
        and len(layers) == 36
        and config.hidden_size == 1024
        and config.vocab_size == VOCAB_SIZE
        and layer_types == ("hybrid",) * 36
        and all(all(row[key] for key in ("has_self_attention", "has_mamba2_mixer", "has_feed_forward")) for row in components)
    )
    return {
        "model_type": config.model_type,
        "num_hidden_layers": len(layers),
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "layer_types": list(layer_types),
        "components": components,
        "matches_registered": matches,
    }


def run_reference(args: argparse.Namespace) -> None:
    require(
        os.environ.get("USE_HUB_KERNELS", "").upper() == "NO",
        "USE_HUB_KERNELS=NO absent before Transformers import",
    )
    require(bool(sys.flags.isolated), "reference requires Python isolated mode")
    candidate_modules = (
        "falcon_h1_adapter",
        "run_r39_falcon_candidate",
        "qcomem_torch",
    )
    require(
        all(importlib.util.find_spec(name) is None for name in candidate_modules),
        "candidate or Q16 module is importable in reference process",
    )
    static = load_bound_json(args.static, args.expected_static_sha256, "static preregistration")
    source = load_bound_json(args.source_manifest, args.expected_source_sha256, "source manifest")
    authority = load_bound_json(args.model_authority, args.expected_model_authority_sha256, "model authority")
    assignment = load_bound_json(args.gpu_assignment, args.expected_gpu_assignment_sha256, "GPU assignment")
    require(static["protocol"] == source["protocol"] == PROTOCOL, "protocol drift")
    require(static["model"]["repo_id"] == MODEL_ID, "model ID drift")
    require(static["model"]["revision"] == HF_MODEL_REVISION, "HF revision drift")
    require(authority["revision"] == HF_MODEL_REVISION, "model authority revision drift")
    row = assignment["rows"][args.rank]
    require(row["rank"] == args.rank and row["uuid"] == args.expected_gpu_uuid, "GPU assignment drift")
    torch.cuda.set_device(0)
    hardware = gpu_identity(args.expected_gpu_uuid)
    torch.use_deterministic_algorithms(True)

    import transformers
    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache

    require(transformers.__version__ == "5.14.1", "Transformers version drift")
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_root),
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
        trust_remote_code=False,
    )
    model.eval().cuda()
    official_sources = verify_official_sources()
    dispatch = force_naive_reference_dispatch()
    require(dispatch["fast_path_observed_after_force"] is False, "reference fast path remains enabled")
    geometry = geometry_receipt(model)
    require(geometry["matches_registered"], "reference model geometry drift")

    frozen = static["rank_inputs"][args.rank]
    document = torch.tensor([frozen["document_token_ids"]], dtype=torch.int64, device="cuda:0")
    queries = [torch.tensor([query["token_ids"]], dtype=torch.int64, device="cuda:0") for query in frozen["queries"]]
    require(tuple(document.shape) == (1, 64), "document shape drift")
    require(len(queries) == 2 and all(tuple(query.shape) == (1, 8) for query in queries), "query shape drift")
    require(int64_le_sha256(document) == frozen["document_token_ids_int64_le_sha256"], "document digest drift")
    for query, query_frozen in zip(queries, frozen["queries"]):
        require(int64_le_sha256(query) == query_frozen["token_ids_int64_le_sha256"], "query digest drift")

    sidecar = LogitBundle()
    trajectories = []
    with torch.inference_mode():
        for request_index, query in enumerate(queries):
            cache = DynamicCache(config=model.config)
            model(input_ids=document, past_key_values=cache, use_cache=True, logits_to_keep=1)
            output = model(input_ids=query, past_key_values=cache, use_cache=True, logits_to_keep=1)
            generated = []
            steps = []
            sequence_length = 72
            for step in range(2):
                record = sidecar.add(f"reference/request-{request_index}/step-{step}", output.logits[:, -1, :])
                generated.append(record["argmax"])
                cache_receipt = official_cache_family_receipt(cache, sequence_length)
                steps.append(
                    {
                        "step": step,
                        "record_id": record["record_id"],
                        "logit_sha256": record["content_sha256"],
                        "generated_token_id": record["argmax"],
                        "cache_family_receipt": cache_receipt,
                    }
                )
                if step + 1 < 2:
                    token = torch.tensor([[record["argmax"]]], dtype=torch.int64, device="cuda:0")
                    output = model(input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1)
                    sequence_length += 1
            trajectories.append(
                {
                    "request_index": request_index,
                    "query_token_ids_int64_le_sha256": int64_le_sha256(query),
                    "generated_token_ids": generated,
                    "steps": steps,
                }
            )
        sidecar_receipt = sidecar.write(args.sidecar)

    own_source = Path(__file__).read_text(encoding="utf-8")
    require(source_is_candidate_import_free(own_source), "reference source imports candidate code")
    shard = {
        "schema_version": "r39-falcon-h1-reference-shard-v1",
        "protocol": PROTOCOL,
        "rank": args.rank,
        "world_size": WORLD_SIZE,
        "scientific_run_valid": True,
        "identity": {
            "static_manifest_sha256": args.expected_static_sha256,
            "source_manifest_sha256": args.expected_source_sha256,
            "model_authority_sha256": args.expected_model_authority_sha256,
            "gpu_assignment_sha256": args.expected_gpu_assignment_sha256,
            "model_id": MODEL_ID,
            "hf_revision": HF_MODEL_REVISION,
            "modelscope_revision": MODELSCOPE_REVISION,
            "transformers_version": transformers.__version__,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "hardware": hardware,
            "geometry": geometry,
            "dispatch": dispatch,
            "official_source_sha256": official_sources,
            "reference_implementation": {
                "candidate_import_free": True,
                "candidate_modules_unavailable_on_sys_path": True,
                "isolated_python_mode": bool(sys.flags.isolated),
                "official_api": "AutoModelForCausalLM + transformers.cache_utils.DynamicCache",
                "source_sha256": sha256_file(Path(__file__)),
                "same_chunk_schedule": [64, 8, 1],
            },
            "input": {
                "source_id": frozen["source_id"],
                "source_object": frozen["source_object"],
                "document_token_ids_int64_le_sha256": int64_le_sha256(document),
                "query_token_ids_int64_le_sha256": [int64_le_sha256(query) for query in queries],
            },
        },
        "trajectories": trajectories,
        "sidecar": sidecar_receipt,
        "claim_boundary": static["claim_boundary"],
    }
    atomic_write(args.output, canonical_bytes(shard))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--model-authority", type=Path, required=True)
    parser.add_argument("--gpu-assignment", type=Path, required=True)
    parser.add_argument("--expected-static-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-model-authority-sha256", required=True)
    parser.add_argument("--expected-gpu-assignment-sha256", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_reference(parse_args())
