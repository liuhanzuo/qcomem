Reviewed the specified PDF (SHA-256 matched) :codex-file-citation{path="/Users/liuhanzuo/MacLLM-Bench/paper_autonomous_multifork_iteration/review/round_38/pdf_only_input/forkaudit.pdf" purpose="source"}

## Summary

ForkAudit is a phase-indexed, fail-closed trace-validation protocol for ownership isolation when a shared KV prefix coexists with mutable GDN recurrent state. It records lifecycle, storage, COW, dispatch, and semantic receipts, then replays predefined predicates. On one Qwen3.5-35B-A3B/H20 stack it reports six complete passing targets, one dispatch target with partial coverage, 96 ownership configurations, targeted fault controls, and a retrospective alias repair. Separate, explicitly non-pooled measurements report retained-state versus F1 and allocator deltas.

## Strongest verified contribution

The strongest substantiated contribution is the unusually explicit, trace-relative ownership contract: it distinguishes coverage from a pass verdict, tests physical storage relationships at setup/transition/final phases, and clearly states the trusted capture assumptions. The controlled historical case (p. 8, Table 2) compellingly illustrates that token/logit/final-state equality can coexist with a persistent-base alias.

## Strengths

- The threat model and conditional guarantee are unusually clear (Sec. 3, pp. 3–4). The paper does not oversell trace replay as security attestation.
- The ownership formulation addresses a real systems-testing gap: output equality alone is insufficient for detecting mutable-state aliasing.
- The separation of logical state, allocator deltas, retained Store, process memory, and capacity is careful (Sec. 4.4, p. 6; Table 3, p. 9).
- The paper labels designed faults as positive controls rather than estimating detection rates (Sec. 4.5, p. 6; Tables 1 and 8, pp. 7 and 17).
- Figures 1–3 and the main tables are legible and technically polished.

## Weaknesses

### Critical issues

- None relative to the paper’s narrowly stated, conditional claims. The manuscript consistently discloses its boundaries rather than claiming a general correctness or security proof.

### Major issues

1. **The central assurance remains dependent on the producer whose behavior is being audited.** The producer enumerates slots and semantically binds them; capture, schema, framework storage semantics, and honest event completeness remain in the TCB (Sec. 3, p. 3; Eq. 2 and discussion, p. 4). The out-of-process observer still trusts producer-side enumeration (p. 8; Table 6, p. 15). Thus, “complete coverage” establishes completeness only relative to the declared producer trace, not independently observed execution. This is a material evidence limitation for an audit framework.

2. **External validity is very narrow.** The primary experiment fixes one hybrid adapter, H20 hardware family, BF16/Q16 KV, 128-token pages, 4,095-token documents, a 32-token transition, \(N\in\{1,8,32\}\), eight greedy outputs, and batch-one sequential execution on one stream/rank (Sec. 4.1, p. 5; Table 4, p. 13; Appendix H, pp. 24–25). The two-stream experiment measures call-interval overlap, not kernel overlap or in-flight cancellation (p. 8). This supports a case study, not a broadly deployable hybrid-serving audit.

3. **Fault evidence demonstrates intended sensitivity, not comparative or natural-bug utility.** The nine primary controls are created for named gates, the gate-suppression matrix uses these same constructed faults, and the five “held-out” faults are held out from executor preparation but not from predicate vocabulary (Sec. 4.5, p. 6; Table 1, p. 7; Table 8, p. 17). The sole historical defect is caught by a conventional persistent-base invariant as well (p. 8, Table 2). There is no empirical comparison against existing lifecycle, metamorphic, differential, or invariant-based test suites on a natural regression corpus. This limits both the novelty and the practical evidence for the proposed integration.

4. **Independent numerical and dispatch validation is incomplete.** Dispatch provenance explicitly binds only a Python callable, not per-call compiled binaries or autotuning choices (Sec. 3, p. 4; Sec. 5.1, p. 6; Table 6, p. 15). The numerical oracles begin from producer-captured post-RoPE or post-normalization boundaries, cover selected attention/GDN rows, and do not validate upstream activation construction or end-to-end behavior (p. 7; Tables 21–23, pp. 23–24). These are useful checks, but insufficient to elevate the system-level assurance.

### Minor issues

- The separate Store–F1 panel is responsibly caveated, but its eight-item slice and reported point estimates make it weak support for the “deployment” framing; timing variability is not presented despite three timing repetitions in one cohort (Table 3, p. 9; Table 5, pp. 13–14).
- The appendix is exhaustive but overlong and table-dense. The contextual tables on published systems and related transfers (Tables 12–16, pp. 19–22) dilute the main ownership-audit narrative; p. 19 also has substantial unused space.

## Questions for the authors

1. Can an independently instrumented backend enumerate relevant allocations/events and bind the compiled kernel/autotuning choice, so that the producer is not responsible for the essential completeness claim?
2. How does ForkAudit compare with conventional lifecycle invariants, metamorphic/differential tests, and existing cache test suites on naturally occurring regressions or version-migration failures?
3. Does the protocol transfer to another hybrid architecture/runtime and to native continuous/ragged batching, multiple documents, different page sizes/precisions, and genuinely in-flight cancellation?
4. For the Store–F1 results, can the authors provide per-item outcomes and dispersion, plus a larger non-cherry-picked benchmark slice?

## Ethical concerns

No direct human-subject or obvious harmful-data concern is evident. The manuscript appropriately states that it is not a security attestation and acknowledges that efficiency could scale deployment (Appendix B, p. 12). A practical concern is that users could overinterpret trace-relative passing results as runtime correctness unless the TCB and unobserved-execution limitations remain prominent in any release.

## Scores

- **Soundness: 3/4.** Good within the explicit trace-relative contract: definitions, controls, and limitations are rigorous. It is not higher because completeness and semantics depend on the trusted producer/capture path, incomplete dispatch binding, and selected-boundary oracles.
- **Presentation: 3/4.** Clear threat-model disclosure and strong figures, but dense terminology, repeated caveats, and a large contextual appendix obscure the central claim.
- **Contribution: 2/4.** The integration of lifecycle, storage ownership, and fail-closed coverage semantics is useful, but evidence is confined to one custom stack and lacks comparative natural-regression evaluation.

**Overall Rating: 4/10 — marginally below the acceptance threshold.** The paper is careful and technically thoughtful, but its evidence currently supports a promising, bounded case-study audit rather than a generally validated systems contribution.

**Confidence: 4/5.** The limitations are explicitly documented in the manuscript; confidence is reduced only because the underlying artifact was not independently executed in this PDF-only review.

**Score ceiling under current evidence: 6/10.** Even giving full credit to every reported result, the single-stack scope, producer-dependent trace completeness, and absence of comparative/natural-bug evidence cap this at weak-accept level.

Evidence that would raise the score: independent artifact replay; trace collection independent of producer enumeration and compiled-dispatch binding; multi-runtime/multi-architecture evaluations with native concurrency; and a preregistered comparison on naturally occurring regressions against strong baselines.

Evidence that would lower the score: inability to reproduce the reports; evidence that unrecorded compiled dispatch changes outcomes; traces missing events/allocations; or natural regressions that bypass the registered predicates.

**Final recommendation: Reject, borderline (4/10).**

This review was produced by an isolated AI subagent for internal pre-submission quality control.
