## (1) Summary and claimed contributions

This review is based solely on :codex-file-citation{path="/Users/liuhanzuo/MacLLM-Bench/paper_autonomous_multifork_iteration/review/round_36/pdf_only_input/forkaudit.pdf" purpose="source"}.

ForkAudit is an offline, fail-closed trace-validation contract for hybrid LLM prefix sharing with read-only KV and mutable GDN recurrent state. It records phase-indexed ownership, lifecycle, append/COW, call, and semantic receipts; replays pointer-free storage predicates; and separates trace coverage from verdicts. On one Qwen3.5/H20 stack, it reports complete passes for six targets, partial dispatch coverage, exact cross-arm/cross-fan-out semantic relations, targeted fault controls, a bounded numerical oracle, and a retrospective alias-defect reproduction/repair.

## (2) Strengths

- The paper identifies a real gap: output equality cannot establish isolation of mutable state. The decomposition into prefix immutability, request/base and request/peer ownership, tail COW, and phase-specific rebinding is clear and useful.
- It is unusually careful about conditionality. The TCB, observation-point scope, lack of compiled-dispatch binding, and non-security/non-attestation boundary are stated plainly rather than hidden.
- The trace schema, mandatory-record rule, normalized storage intervals, and coverage-versus-verdict distinction form a coherent, actionable protocol.
- Evaluation is disciplined within its fixed setting: factorial ownership configurations, fault-to-gate mapping, gate-suppression experiments, selected independent CPU/NumPy computations, and a retrospective defect case directly support the central motivation.
- The manuscript is exceptionally detailed and visually readable; tables clearly delimit what each cohort does and does not authorize.

## (3) Weaknesses

- The central assurance is necessarily close to “the trusted instrumented producer faithfully reported that its own registered predicates held.” Faithful producer enumeration, capture, tensor/storage semantics, and semantic binding remain in the TCB. The process-separated observer does not remove this dependence. This is a reasonable engineering boundary, but substantially limits the strength and novelty of the claimed validation.
- Evidence is confined to one model/runtime/hardware stack, sequential batch-one execution, one main partial-tail geometry, N ≤ 32, and selected captured operator boundaries. The paper explicitly acknowledges this, but the evaluation does not establish that ForkAudit transfers cleanly to other hybrid architectures, cache managers, optimized recurrent kernels, or native serving schedules.
- The fault studies are designed mutations, and the “designer–executor-separated” faults are still drawn from an existing predicate vocabulary. They demonstrate intended gate sensitivity, not detection recall, false-positive behavior, or robustness to realistic unforeseen failures. The historical case is informative but only one defect, and a conventional persistent-base invariant also catches it.
- Much of the technical statement is definitional: once complete, honest trace capture and correct replay are assumed, the pass implication follows directly. A stronger contribution would provide a more formal link from framework-level storage semantics and event instrumentation to the audited ownership property, or independently validate capture completeness.
- The paper’s breadth creates some dilution: extensive contextual deployment and related-work panels are carefully disclaimed, but contribute little to evaluating the core contract. The PDF lists paths and replay claims, yet does not itself establish public artifact availability or permit assessment of those artifacts.

## (4) Questions for the authors

1. What minimal adapter interface and engineering effort are required to apply ForkAudit to a different hybrid model/runtime, especially one with fused recurrent kernels or a production continuous-batching scheduler?
2. Can you demonstrate capture completeness or cross-check producer slot enumeration using a mechanism meaningfully less coupled to the candidate state-management path?
3. How does the protocol behave for legitimate aliasing/view patterns that make conservative intervals overlap, and what false-positive rate or diagnostic burden results in practice?
4. Why should the reader regard the five designer–executor-separated faults as materially stronger evidence than conventional predesigned mutation tests when the predicate vocabulary remains known?
5. Will the complete replay package, source revisions, raw sidecars, and environment specification be anonymously available at review time?

## (5) Reproducibility and ethics

The PDF provides unusually strong reproducibility documentation: frozen versions, geometry, hardware, schemas, artifact paths, and replay boundaries. Still, reproducibility is only partially assessable from the submission because most tensors are hashes rather than archived data, no independent model re-execution is claimed, and artifact accessibility cannot be confirmed from the PDF. Ethics concerns appear limited: the study uses PG-19 training data and no human subjects, and it appropriately notes that systems efficiency may scale deployment without claiming safety, privacy, or bias assessment.

## (6) Overall score—state exactly one allowed integer and concise justification

4. The paper presents a thoughtful and carefully scoped systems-audit methodology with strong within-stack controls, but its principal guarantee rests on a broad trusted producer/capture path and its empirical support is too narrow to justify acceptance as a broadly validated ICLR contribution.

## (7) Confidence—state one integer 1–5 and concise justification

4. The paper is detailed enough to assess its protocol and stated evidence boundaries, though confidence is limited by the inability to inspect or execute the referenced artifacts and by the specialized systems setting.

## (8) Verdict—Accept or Reject

Reject
