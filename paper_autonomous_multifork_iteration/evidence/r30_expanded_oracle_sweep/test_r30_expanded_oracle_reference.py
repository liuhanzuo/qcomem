from __future__ import annotations

import numpy as np

from r30_expanded_oracle_reference import ATTENTION_FAULTS, GDN_FAULTS, dense_attention, recurrent_transition


def test_reference_shapes_and_faults() -> None:
    rng = np.random.default_rng(20260825)
    query = rng.normal(size=(1, 4, 3, 8)).astype(np.float32)
    key = rng.normal(size=(1, 2, 7, 8)).astype(np.float32)
    value = rng.normal(size=(1, 2, 7, 8)).astype(np.float32)
    clean = dense_attention(query, key, value, [4, 5, 6], list(range(7)), 8 ** -0.5)
    assert clean.shape == (1, 3, 4, 8)
    assert all(not np.array_equal(clean, dense_attention(query, key, value, [4, 5, 6], list(range(7)), 8 ** -0.5, fault=fault)) for fault in ATTENTION_FAULTS)

    q = rng.normal(size=(1, 3, 2, 4)).astype(np.float32)
    k = rng.normal(size=(1, 3, 2, 4)).astype(np.float32)
    v = rng.normal(size=(1, 3, 2, 4)).astype(np.float32)
    g = -np.abs(rng.normal(size=(1, 3, 2))).astype(np.float32)
    beta = rng.uniform(size=(1, 3, 2)).astype(np.float32)
    state = rng.normal(size=(1, 2, 4, 4)).astype(np.float32)
    output, final = recurrent_transition(q, k, v, g, beta, state, 0.5)
    assert output.shape == (1, 3, 2, 4) and final.shape == state.shape
    assert all(not np.array_equal(output, recurrent_transition(q, k, v, g, beta, state, 0.5, fault=fault)[0]) for fault in GDN_FAULTS)
