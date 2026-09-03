# R27-TERRA-C

Access mode: PDF only. Model: `gpt-5.6-terra`.

## Summary

ForkAudit is a receipt-based audit contract for shared-prefix inference in hybrid LLMs, separating token-indexed KV ownership from mutable recurrent GDN state. It evaluates a 2x2 ownership factorial on one Qwen3.5 configuration and combines replayable lifecycle/storage receipts, cross-arm equality, selected FP32/NumPy operator checks, and targeted faults. Its central claim is narrow: conditional evidence under trusted producer capture for one sequential stack.

## Strengths

- Unusually clear claim discipline and assurance hierarchy.
- Concrete lifecycle phases, pointer-free range rules, tail-COW requirements, and failure conditions.
- Strong experimental hygiene within the stated scope.
- Candid disclosure of common-mode failure, trusted capture, unbound dispatch, selected-oracle coverage, and absence of concurrent kernels.
- Legible tables and figures that communicate the audit architecture well.

## Weaknesses

Must fix:

- The novelty case remains insufficiently compelling without a sharper conceptual advance, guarantee, or same-system comparison to prior testing/auditing practice.
- Validation is too narrow to establish broad practical value: one model, one hardware family, one sequential stream, one main partial-tail geometry, and selected operator rows/transitions.
- Receipt-complete evidence depends on honest producer capture and mostly digests; an external capture/attestation path or more rigorous threat model is needed.

Optional improvements:

- Condense the paper around the contract, evidence hierarchy, and one decisive validation.
- Compare against a strong same-system audit/test baseline using real or blinded faults.
- Clarify or shorten deployment/related-work tables.

## Questions

1. Which bug classes does ForkAudit catch beyond a strong differential/metamorphic test plus allocator instrumentation?
2. How would storage/lifecycle witnesses work under concurrency, continuous batching, eviction, and multi-document reuse?
3. Can an independent party capture rather than only replay receipts?
4. What is the adoption cost and which pieces are reusable?
5. Why should the selected oracle rows transfer to untested layers, shapes, and schedules?

## Reproducibility

Moderate. The PDF gives detailed geometry, artifact paths, schemas, replay targets, and decision rules, but depends on a large model/H20 environment and producer-captured evidence. A small runnable reference and independent-capture instructions would help.

## Rating

Overall: **5/10** (borderline). Confidence: **4/5**.
