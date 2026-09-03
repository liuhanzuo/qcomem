# Independent GDN transition oracle design decision

Date: 2026-08-19  
Issue: meta-review A6 / R4-006

## Alternatives considered

1. **Repeat the candidate Torch transition in a second Torch helper.** Cheapest, but it shares the same framework reduction path and is not sufficiently independent.
2. **Capture native recurrent-rule inputs and replay the explicit recurrence in NumPy FP32.** Selected. It removes candidate imports, separates GPU capture from CPU reconstruction, supports frozen rows/tolerances and seeded faults, and directly tests the common-mode arithmetic concern.
3. **Reimplement the complete GDN layer (projections, causal convolution, recurrence, gated normalization and output projection).** Broader but substantially increases accidental specification drift and review cost; deferred unless the bounded recurrent-core oracle exposes a discrepancy.

## Selected design and boundary

The scientific run uses one existing Qwen3.5/vLLM query path only to capture the actual native `chunk_gated_delta_rule` inputs and outputs at layers 0, 10, 20 and 38 after a 128-token PG-19 train document prefix. A separate NumPy-only module reconstructs the four-token recurrence from q, k, v, g, beta and the incoming FP32 state. It never imports Torch or the candidate implementation. Four pre-assigned wrong-transition rules test non-vacuous sensitivity.

This experiment can support agreement of four selected recurrent-core transitions with an independent FP32 reference. It cannot support independent correctness of projections, causal convolution, gating normalization, output projection, all 30 GDN layers or end-to-end logits.

The experiment is new and isolated. It does not rerun the KV/GDN ownership factorial, detector matrix, lifecycle transfer, attention oracle, memory table or mutation campaign.
