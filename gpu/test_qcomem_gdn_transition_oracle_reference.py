from __future__ import annotations

import numpy as np
import torch

from qcomem_gdn_transition_oracle_reference import FAULTS, metrics, recurrent_transition
from run_qcomem_qwen35_gdn_transition_oracle import (
    _capture_reference_qk_inputs,
    _resolve_native_query_scale,
)


def _fixture() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(20260819)
    query = rng.normal(0, 0.2, (1, 4, 4, 8)).astype(np.float32)
    key = rng.normal(0, 0.2, (1, 4, 4, 8)).astype(np.float32)
    value = rng.normal(0, 0.2, (1, 4, 4, 8)).astype(np.float32)
    g = -rng.uniform(0.01, 0.2, (1, 4, 4)).astype(np.float32)
    beta = rng.uniform(0.1, 0.9, (1, 4, 4)).astype(np.float32)
    initial = rng.normal(0, 0.1, (1, 4, 8, 8)).astype(np.float32)
    return query, key, value, g, beta, initial


def test_reference_is_deterministic_and_finite() -> None:
    first = recurrent_transition(*_fixture())
    second = recurrent_transition(*_fixture())
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert np.isfinite(first[0]).all()
    assert np.isfinite(first[1]).all()


def test_every_preregistered_fault_changes_the_transition() -> None:
    clean_output, clean_state = recurrent_transition(*_fixture())
    for fault in FAULTS:
        wrong_output, wrong_state = recurrent_transition(*_fixture(), fault=fault)
        output_error = metrics(clean_output, wrong_output)
        state_error = metrics(clean_state, wrong_state)
        assert (
            output_error["relative_l2"] > 1e-4
            or state_error["relative_l2"] > 1e-4
        ), fault


def test_native_chunk_default_query_scale_is_inverse_sqrt_width() -> None:
    def torch_chunk_gated_delta_rule() -> None:
        return None

    query = torch.empty((1, 4, 32, 128), dtype=torch.float32)
    scale, source = _resolve_native_query_scale(
        query, torch_chunk_gated_delta_rule, {}
    )
    assert scale == 128**-0.5
    assert source == "native_default_inverse_sqrt_key_width"


def test_explicit_native_query_scale_takes_precedence() -> None:
    def arbitrary_kernel() -> None:
        return None

    query = torch.empty((1, 4, 32, 128), dtype=torch.float32)
    scale, source = _resolve_native_query_scale(
        query, arbitrary_kernel, {"query_scale": 0.25}
    )
    assert scale == 0.25
    assert source == "explicit_call_keyword"


def test_reference_qk_inputs_use_native_bfloat16_preprocessor() -> None:
    namespace: dict[str, object] = {"torch": torch}
    exec(
        "def l2norm(x, dim=-1, eps=1e-6):\n"
        "    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)\n"
        "def torch_chunk_gated_delta_rule():\n"
        "    return None\n",
        namespace,
    )
    kernel = namespace["torch_chunk_gated_delta_rule"]
    native_l2norm = namespace["l2norm"]
    query = torch.arange(1, 17, dtype=torch.bfloat16).reshape(1, 1, 2, 8)
    key = torch.flip(query, dims=(-1,))
    captured_query, captured_key, receipt = _capture_reference_qk_inputs(
        query, key, kernel, True
    )
    assert torch.equal(captured_query, native_l2norm(query, dim=-1, eps=1e-6))
    assert torch.equal(captured_key, native_l2norm(key, dim=-1, eps=1e-6))
    assert receipt["native_use_qk_l2norm_in_kernel"] is True
    assert receipt["reference_inputs_post_native_qk_l2norm"] is True
    assert receipt["query_key_capture_boundary"] == "post_native_qk_l2norm_pre_fp32_recurrence"


def test_reference_qk_capture_fails_closed_without_native_preprocessor() -> None:
    namespace: dict[str, object] = {}
    exec("def torch_chunk_gated_delta_rule():\n    return None\n", namespace)
    query = torch.ones((1, 1, 2, 8), dtype=torch.bfloat16)
    try:
        _capture_reference_qk_inputs(
            query, query, namespace["torch_chunk_gated_delta_rule"], True
        )
    except RuntimeError as error:
        assert "l2norm helper is unavailable" in str(error)
    else:
        raise AssertionError("missing native preprocessor was accepted")
