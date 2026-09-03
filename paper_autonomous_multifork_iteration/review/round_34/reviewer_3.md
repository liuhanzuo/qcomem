(1) Summary and claimed contributions

ForkAudit is an offline, fail-closed trace-validation protocol for detecting state-ownership errors when LLM continuations fork from a shared prefix with hybrid KV and recurrent state. It combines phase-indexed storage/call receipts, ownership predicates, semantic equivalence checks, selected captured-boundary numerical oracles, and designed fault injections. The evaluation is explicitly limited to a fixed Qwen3.5/H20 stack and declared observation points.

(2) Strengths

- The paper defines its assurance boundary unusually clearly: producer enumeration, capture, storage semantics, and replay are acknowledged as TCB components rather than hidden assumptions.
- The phase-aware ownership contract is technically concrete, including pointer-free interval reconstruction, tail-COW checks, GDN transition checks, and explicit distinction between coverage and verdict.
- The empirical protocol contains useful positive controls and carefully distinguishes first-gate localization from a defect-detection rate. The suppressed-gate experiment compellingly illustrates that matching tokens and, often, logits do not establish ownership correctness.
- The writing, tables, and claim-to-cohort authorization are exceptionally clear. Limitations are stated prominently rather than buried.

(3) Weaknesses

- The core evidence is fundamentally trace-relative and depends on the same producer to enumerate the slots and produce much of the evidence being replayed. The process-separated observer improves relation reconstruction but does not independently establish capture completeness, state semantics, or dispatch behavior.
- Generality and practical significance are limited: one model family, one hardware family, sequential batch-one execution, limited fan-out, a prescribed partial-tail geometry, and no native continuous/ragged batching, realistic serving scheduling, eviction, or in-flight cancellation.
- The method is largely an integration of established testing, logging, metamorphic, and ownership-checking ideas. The paper does not convincingly establish that the resulting protocol is broadly deployable or that it detects realistic regressions beyond deliberately constructed mutations.
- Numerical checks begin after important upstream boundaries and cover selected operator instances; cross-arm semantic equality also risks common-mode agreement. The paper acknowledges this, but it limits the strength of the empirical validation.
- Reproducibility is asserted through artifact paths and replay claims, but the PDF alone does not make the artifacts, environment, or raw evidence independently assessable.

(4) Questions for the authors

- What engineering overhead does mandatory capture impose in a realistic CI workflow, and which receipts are essential for a useful lower-cost deployment?
- Can the method be evaluated on independently discovered historical bugs or a concealed fault suite whose construction is not informed by the predicate vocabulary?
- How would the contract and witnesses change for continuous batching, asynchronous reclamation, native recurrent kernels, and multi-document prefix trees?
- Is there any independently produced capture or end-to-end differential execution that reduces reliance on trusted producer slot enumeration?

(5) Reproducibility and ethics

The manuscript gives unusually detailed protocol, schema, artifact-map, and replay descriptions, which supports auditability. However, based on the PDF alone, reproducibility remains unverified because the referenced packages and raw artifacts are not available for inspection. The ethics discussion is appropriate and identifies the limited system-behavior scope; no notable human-subject concern is apparent.

(6) Overall score—4

Weak reject. This is a careful and clearly scoped systems-validation contribution with strong internal methodology, but its novelty and external significance are constrained by reliance on a trusted capture path, narrowly constructed controls, and a highly restricted execution setting.

(7) Confidence—4

The paper is detailed, candid about its limitations, and sufficient to assess its technical scope, though evaluating the claimed replay artifacts would be necessary to resolve reproducibility and implementation-strength questions.

(8) Verdict—Reject
