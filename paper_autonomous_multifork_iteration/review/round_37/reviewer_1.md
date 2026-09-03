# Reviewer 1

## Summary

ForkAudit is a fail-closed, phase-aware trace-validation framework for forked hybrid LLM inference with shared KV cache and mutable GDN state. It records lifecycle, storage, copy-on-write, call-contract, and semantic receipts; replays ownership predicates; and explicitly distinguishes trace coverage from predicate verdicts. On a fixed Qwen3.5/H20 setup, it reports complete passing evidence for six of seven targets, while correctly leaving compiled-dispatch provenance partial. The paper also includes designed fault campaigns, a retrospective aliasing bug, bounded numerical replay checks, and separate memory-quality context experiments.

## Strengths

- The problem is real and under-addressed: output equality alone does not establish isolation of mutable hybrid state.
- The contract is unusually explicit about its assumptions, mandatory evidence slots, phase-specific predicates, and assurance boundary.
- The storage-witness schema and distinction between coverage and verdict are technically clear and potentially useful for regression testing.
- Evaluation is careful in several respects: factorial ownership configurations, fixed positive controls, an executor-separated fault exercise, a historical bug, and well-labeled non-pooled auxiliary cohorts.
- The paper is transparent about substantial limitations, especially the trusted producer/capture path, partial dispatch coverage, and non-production schedule.
- Reproducibility reporting is detailed, with manifests, hashes, sidecars, and stated replay boundaries.

## Weaknesses

- The central assurance remains conditional on the candidate-side producer faithfully enumerating, labeling, and capturing all relevant state. The out-of-process observer reconstructs relations but still trusts producer-selected slots and CUDA-IPC semantics. Thus the evidence does not independently establish that the trace corresponds to the actual execution state; it validates consistency of a trusted trace.
- External validity is very limited: one model, one hardware family, a single page size and partial-tail geometry, sequential batch-one calls, N <= 32, and no native continuous/ragged batching, eviction, realistic scheduler behavior, or in-flight cancellation.
- The numerical checks begin after producer-captured post-RoPE/post-normalization boundaries and cover selected operators (only 12/30 GDN layers in the expanded sweep). They do not provide an independent end-to-end execution oracle.
- The fault evidence demonstrates sensitivity to deliberately constructed mutations, including faults designed around the framework's predicate vocabulary. It does not characterize false positives, recall on unseen faults, or performance on a natural bug corpus. The retrospective defect is valuable but only one case, and a conventional persistent-base invariant also catches it.
- Novelty is principally an integration and engineering contribution: trace binding, mutation testing, storage-range checking, and metamorphic relations have strong precedents. The paper needs a sharper argument for what conceptual advance—not just a rigorous combination—is new enough for ICLR.
- The deployment Store–F1 and related-system tables are clearly caveated, but make the paper diffuse and risk distracting from the auditing contribution. They do not strengthen the main validation claim.
- Presentation is polished but overly dense. The main paper has many cohorts, tables, and qualifications, making it difficult to identify the minimal core contribution and decisive evidence.

## Questions for the authors

1. What concrete mechanism prevents an incomplete or incorrectly mapped producer enumeration from yielding a complete trace, beyond trusting the producer in the TCB? Can you validate enumeration independently at least at the framework/runtime boundary?
2. Can the method be demonstrated on a second hybrid architecture/runtime and on genuine continuous-batching or concurrent serving traces?
3. How much incremental implementation and runtime overhead does capture impose, and which subset of receipts is practical for CI?
4. Can you evaluate against a blinded, naturally occurring bug corpus or compare detection/localization against standard invariants and differential testing?
5. Why should the CoMem/Store–F1 and broad contextual benchmarking remain in the main paper rather than be reduced to a short motivating appendix?
6. Will the complete pinned artifacts, executable source, and raw enough state to independently recompute—not merely hash-check—the primary claims be publicly available?

## Ethical concerns

No major direct ethical issue is apparent. The use of PG-19 training data and deployment-efficiency results should nevertheless clearly document data licensing and acknowledge that cheaper inference can enable greater model deployment. The stated non-security assurance boundary is appropriate and should remain prominent to prevent operational overclaiming.

## Overall score

**5 / 10 — Borderline / weak reject.**

This is careful, technically competent systems-validation work with a useful auditing formulation. However, the evidence remains fundamentally trace- and producer-trust-relative, and the empirical scope plus novelty case are currently too narrow for a strong ICLR acceptance.

## Confidence

**4 / 5 — High confidence.**

## Final recommendation

**Weak Reject.** I would be positive on a revised version that narrows the paper around the core audit contribution, demonstrates cross-stack and realistic-serving applicability, and supplies stronger independent evidence for trace completeness and fault-detection value.

