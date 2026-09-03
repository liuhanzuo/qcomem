## Meta-review (Round 35)

This synthesis is not an additional score. The three reviewer scores are each **4 (weak reject)**, and all three recommend **Reject**.

### Panel outcome and consensus

The panel is strongly aligned: ForkAudit is viewed as careful, technically coherent, unusually transparent about its assurance boundary, and potentially useful as a trace-relative validation protocol. However, reviewers agree that the current evidence does not substantiate a broadly significant systems-validation contribution. The decisive concerns are a broad trusted producer/capture boundary, fixed-stack and non-production-like evaluation, and insufficient evidence that the full protocol materially outperforms simpler conventional invariant/testing suites on realistic defects.

### Agreed strengths

- Exceptionally explicit, honest assurance boundary: trace validation is separated from attestation/security, coverage from verdicts, and observed from unobserved behavior.
- Technically coherent phase-aware ownership contract, including immutable prefixes, private mutable state, tail copy-on-write, recurrent rebinding, and semantic relations beyond token/logit equality.
- Concrete, replayable-in-principle pointer-free storage witnesses and conservative byte-range overlap rules.
- Careful reporting, clear tables/appendices, useful controls, and candid limitations.
- The historical regression, mutation controls, selected numerical checks, and process-separated observer add value within the stated bounded claim.

### Agreed decision blockers

1. **Trusted-producer dependence.** The validator checks consistency of producer-supplied receipts, slot inventories, storage identities, and captured boundaries; it does not independently establish complete or correct capture. The observer does not resolve this because it still depends on producer-selected state and capture semantics.

2. **Narrow empirical scope.** Evidence is limited to one Qwen3.5/H20 configuration, fixed geometry, sequential batch-one execution, and limited fan-out. It does not demonstrate native continuous/ragged batching, scheduler lifecycle behavior, cancellation/eviction, optimized recurrent kernels, or realistic multi-document serving.

3. **Insufficient comparative bug-detection evidence.** Most injected faults are aligned with the protocol’s registered predicates. The designer–executor split does not make the method’s vocabulary held out, and the one retrospective alias defect is also caught by a conventional persistent-base invariant. No convincing comparison against a strong baseline suite—persistent-base guards, allocator/storage checks, differential testing, and conventional runtime tests—establishes the incremental value of the complete protocol.

4. **Incomplete independent end-to-end validation.** Semantic and numerical checks begin at candidate-produced/captured boundaries or compare arms sharing common components. They support bounded consistency, but cannot rule out common-mode upstream construction/capture errors.

5. **Contribution/positioning remains incremental.** Reviewers see the main contribution as a thoughtful integration and instrumentation protocol using known ownership, trace-validation, metamorphic-testing, and numerical-replay ideas. The paper does not yet demonstrate capabilities unavailable from a simpler combined baseline.

Artifact availability and live reproducibility also remain unverified from the reviewed material, although reviewers consistently praised the documentation and replay specification.

### Disagreements

There are no material disagreements on outcome or the core blockers. The differences are matters of emphasis:

- Reviewers 1 and 2 emphasize practical scope, overhead, CI usability, and realistic scheduling/continuous batching.
- Reviewer 3 most directly emphasizes the need to quantify incremental value over a simpler invariant suite and notes that “exact” semantic checks remain candidate-boundary dependent.
- Reviewers 1 and 2 more explicitly call for an additional runtime/architecture and a broader defect corpus; reviewer 3 frames the same need around independently discovered regressions and baseline comparison.

### Repairability assessment

Presentation can be improved through prose/layout alone: compress repeated boundary caveats, foreground the minimal adoption path and central methodological insight, clarify novelty positioning, and make the bounded trace-relative claim more immediately legible.

The decision blockers are **not honestly repairable by prose or layout alone**. Addressing them requires materially new evidence: evaluation on realistic continuous/ragged batching and lifecycle conditions, a second relevant runtime/implementation, quantified capture/replay overhead, stronger independent producer/capture cross-checks where feasible, and comparative evaluation against a specified conventional baseline on independently sourced or meaningfully held-out defects.

### Plateau / next action

The current paper has reached a revision plateau for text-only changes. Do not continue polishing toward acceptance without new experiments. The appropriate next action is to decide whether the authors can run a focused evidence expansion: realistic serving/scheduling coverage, a strong baseline comparison, and a more independent defect/capture-validation study. If those experiments are unavailable, the paper should be positioned and evaluated as a narrowly scoped engineering/trace-validation case study rather than as broad ownership validation.
