# R27-TERRA-B

Access mode: PDF only. Model: `gpt-5.6-terra`.

## Summary

ForkAudit is a receipt-based audit contract for request isolation in hybrid LLM inference, separating immutable/shared attention KV from mutable recurrent GDN state. It evaluates a 2x2 KV/GDN ownership factorial on one Qwen3.5 configuration, using ledger replay, selected FP32/NumPy operator checks, and targeted fault injections. Its central claim is conditional: receipt-level evidence under trusted producer capture, not end-to-end or independent verification.

## Strengths

- The assurance levels and limitations are unusually clear.
- Lifecycle phases, storage-range predicates, tail COW, callable receipts, and memory denominators are concrete.
- The factorial, cross-fanout checks, selected operator oracles, and positive controls form a coherent narrow evidence stack.
- Memory accounting avoids exaggerated capacity claims.
- The appendix is detailed, organized, visually readable, and includes an artifact/claim map.

## Weaknesses

Must fix:

- The result remains fundamentally self-attested; the paper needs a sharper threat model and stronger independent validation beyond disciplined instrumentation/regression testing.
- Validation is too narrow for a general audit contract: one model/runtime/hardware family, one main partial-tail geometry, N<=32, sequential one-stream execution, and selected operator checks.
- The fault study only shows nine designed mutants reaching intended gates, not behavior on realistic/unseen faults, bypasses, or false positives.

Optional improvements:

- Reduce density and repetition around the threat model and contribution.
- Justify oracle selection/tolerances and report sensitivity or distributions.
- Quantify instrumentation, storage, runtime, and integration overhead.
- Shorten conventional deployment/context sections.

## Questions

1. Which adversaries or failure modes are covered under honest capture?
2. What independently verifies external byte bindings?
3. How does the contract behave under concurrency, batching, eviction, or multi-document trees?
4. Why are eight attention rows and four GDN transitions sufficient?
5. What are the end-to-end capture/storage/replay overheads?
6. Can an evaluator recreate the principal result without privileged access to the original producer environment?

## Reproducibility

Moderate. Geometry, schema, decision rules, paths, and replay commands are precise, but most evidence is digest-based and reproduction depends on a pinned large-model/GPU environment plus trusted capture. It is more reproducible than independently verifiable.

## Rating

Overall: **4/10**. Confidence: **4/5**.
