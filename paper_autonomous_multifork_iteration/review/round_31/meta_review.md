## 1. Score summary and consensus

All three reviewers assign **4/10 (borderline reject)** with **4/5 confidence** and recommend **Reject**. Consensus: technically careful and unusually transparent, but insufficiently novel and empirically supported for acceptance.

## 2. Shared strengths

- Clear, disciplined assurance boundary and limitation disclosure.
- Concrete phase-aware ownership obligations for immutable KV and mutable recurrent state, including transition rebinding and tail COW.
- Strong trace schema/reporting discipline: typed receipts, coverage-versus-verdict separation, normalized storage witnesses, hashes, and explicit measurement denominators.
- Informative targeted fault/gate-suppression study showing output equality can miss designed ownership failures.
- Thoughtful reproducibility documentation and cautious interpretation of memory/overhead findings.

## 3. Shared blocking weaknesses

- **Fundamental:** replay validates the consistency of a trusted, mandatory-complete producer’s trace—not independent correctness of live execution. Coherent omission/fabrication, transient restored mutations, and common-mode capture/runtime faults remain invisible.
- Extremely narrow external validity: one fixed stack and geometry, short decoding, sequential batch-one/single-stream operation, without realistic continuous/ragged batching, eviction, concurrency, cancellation, or multi-document serving.
- Novelty is not convincingly separated from an integration of existing tracing, COW, hashing, invariant checks, and differential/metamorphic testing; no credible baseline comparison establishes incremental value.
- Faults are tailored positive controls rather than held-out or naturally occurring failures; false-positive/false-negative behavior is not established.
- Numerical and systems evidence remain partial: captured-boundary checks share upstream implementation assumptions, coverage is limited, and memory/overhead results are not production-capacity evidence.

## 4. Actionable manuscript-only revisions

- Reframe the central claim consistently as **trace-consistency validation under a trusted-capture contract**, not validation of live ownership behavior.
- Put the trust/threat model, undetectable failure modes, and shared-observer limitation prominently in the abstract, introduction, and conclusion.
- Sharpen the novelty statement against conventional invariant logging, storage checks, and differential testing; state the claimed incremental contribution precisely without asserting unmeasured superiority.
- Compress or move tangential supporting sections that do not strengthen the central ForkAudit case study.
- Make operational limitations of memory and timing results visually and rhetorically prominent: no process-level capacity result, non-production full-copy controls, and narrowly scoped overhead measurements.
- Clearly distinguish designed sensitivity controls from bug-detection-rate evidence.

These changes improve honesty and readability, but do not address the core decision blockers.

## 5. New evidence required to change the decision

Ranked by expected decision impact:

1. **Independent observation/capture path:** external process, lower-level allocator/kernel telemetry, or hardware-supported capture, with quantified agreement/disagreement against ForkAudit. This most directly addresses the fundamental trusted-producer limitation.
2. **Realistic deployment evaluation:** native continuous/ragged batching, multiple documents, varied prefix/tail lengths, eviction/re-admission, concurrency, and cancellation; ideally on a materially different hybrid runtime/model.
3. **Strong comparative baseline:** conventional lifecycle assertions, storage checks, hashing, and differential tests, compared on assurance gained, engineering cost, overhead, and failures caught.
4. **Held-out realistic fault evaluation:** non-tailored and adversarial/common-mode failure modes, with false-alarm and miss characterization rather than only gate-aligned mutations.
5. **Independent artifact execution and release:** reproducible capture/replay package with raw supporting trials, pinned environment, and outside re-execution.
6. **Broader numerical and operational measurement:** more layers/boundaries, longer runs, and process-level memory/admitted-capacity measurements.

## 6. Plateau assessment

**Another wording/reorganization-only round is very unlikely to raise consensus above 4/10.** Presentation is already a strength; the reviewers’ objections are not principally confusion or overstatement.

Fixable presentation issues include claim framing, novelty positioning, section focus, and clearer caveats for memory/fault evidence. The evidence limits are fundamental: self-attested capture, lack of external validation, narrow deployment scope, absence of a convincing baseline, and tailored fault evidence. Better prose can make the paper more defensible, but cannot convert these limitations into acceptance-level evidence.

## 7. Meta-verdict

**Reject at the current evidence level.** Preserve the paper’s unusually rigorous disclosure, but treat it as a bounded trace-validation case study unless new independent capture, comparative, and realistic-serving evidence is obtained.
