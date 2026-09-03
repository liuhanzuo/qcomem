## Summary

I reviewed all 27 pages, including rendered figures, tables, references, and appendices. The frozen PDF matches the expected SHA-256 `064a6fd55eda24a58c082d4ed8893a187df22a1adc9983d7472edf77b03facf3`. :codex-file-citation{path="/Users/liuhanzuo/MacLLM-Bench/paper_autonomous_multifork_iteration/review/round_38/pdf_only_input/forkaudit.pdf" purpose="source"}

The paper presents ForkAudit, a phase-aware trace contract for validating ownership and isolation of shared KV cache and mutable recurrent/GDN state when a prefix is forked into multiple requests. Its seven targets cover identity, prefix immutability, private ownership, tail copy-on-write, dispatch provenance, cross-arm equivalence, and cross-fan-out consistency. The implementation is evaluated on one Qwen3.5-35B-A3B/H20 stack using 96 ownership configurations, designed mutations, five designer–executor-separated faults, one historical alias regression, selected numerical replay, lifecycle extensions, and separate retained-memory measurements.

The manuscript is exceptionally explicit about its boundaries. However, the central evidence remains conditional on a broad producer/capture TCB, and most bug-detection evidence consists of faults constructed against the audit’s existing predicate vocabulary on a single bespoke stack. These limitations constrain the demonstrated generality and significance.

## Strongest verified contribution

The strongest contribution verifiable from the PDF itself is the concrete, phase-indexed ownership contract and pointer-free storage-witness schema: the separation of mandatory evidence coverage from replay verdicts, explicit setup/transition/final obligations, conservative byte-interval reconstruction, and fail-closed treatment of absent records (Section 3, pp. 3–4; Appendix D, pp. 15–16; Tables 6 and 17).

This is more substantive than merely comparing outputs and gives implementers a precise checklist for hybrid-state ownership regressions. The empirical receipts themselves cannot be independently checked in this PDF-only review.

## Strengths

- The threat model and assurance boundary are unusually honest and precise. Section 3, pp. 3–4 explicitly excludes malicious or coherently faulty producers, transient writes between observation points, OS/driver faults, and unbound compiled dispatch.
- The lifecycle decomposition is technically sensible. ForkAudit checks setup, a registered transition, and final state, thereby addressing failures that terminal output comparisons can miss (Figures 1–2, pp. 2 and 5).
- The pointer-free range construction in Equations 3–4 and Appendix D, p. 16 is clear, conservative, and compatible with detached replay.
- Controls are reported per frozen fault rather than misleadingly pooled into a detection rate. Tables 1–2 and 7–9 clearly distinguish first-gate localization, gate-suppressed behavior, constructed faults, and the single historical defect.
- Memory denominators are carefully separated. The paper consistently distinguishes retained tensor payload, allocator delta, logical state, process memory, and capacity (Section 4.4, p. 6; Table 3, p. 9; Table 18, p. 22).
- The appendices provide strong traceability: cohort-to-claim authorization, limitations, environment details, and a claim-to-artifact map (Tables 4–6 and 24).
- The rendered PDF has no clipping, overlap, corrupted glyphs, or illegible figures. Main-text presentation is polished.

## Weaknesses

### Critical issues

None identified. I found no contradiction that invalidates the paper’s narrowly stated, trace-relative claim.

### Major issues

1. **The assurance remains largely producer-relative.**
   Locations: Section 3, pp. 3–4, especially the threat model and Equation 2; Section 5.2, “Out-of-process GDN observation,” p. 8; Table 6, p. 15; Section H, pp. 24–25.

   The conditional statement assumes honest, mandatory-event-complete capture and correct manifest binding, digests, and replay. The producer still enumerates and semantically labels the relevant slots. The out-of-process observer reconstructs relations only from producer-supplied CUDA-IPC handles and does not independently determine what should have been observed. Consequently, a coherent omission, misenumeration, or common-mode capture error can escape the audit.

   This is properly disclosed, so it is not a fatal soundness flaw. It nevertheless substantially limits the practical strength of “validation”: the current result establishes trace self-consistency at registered points, not independently established execution ownership.

   A material fix requires independently deriving expected slots from the frozen model geometry and schedule, redundant capture from a separate implementation layer, and omission/relabeling controls demonstrating that producer-side enumeration errors cannot silently pass.

2. **Effectiveness is demonstrated mainly with faults designed for the existing gates.**
   Locations: Section 4.5, p. 6; Table 1 and Figure 3, p. 7; Table 2, p. 8; Tables 7–9, pp. 16–17; Section H, p. 25.

   The nine primary mutations directly target predeclared predicates. The five designer–executor-separated faults are fresh to the executor but explicitly remain within ForkAudit’s predicate vocabulary. The historical study contains one organically encountered mechanism, and a simpler persistent-base content invariant also catches it.

   Thus, the experiments demonstrate reachability and localization of known obligations, but not broad bug-finding utility, unseen-fault recall, false-positive behavior, or advantage over a disciplined suite of conventional invariants. This is a material evidence limitation for a validation-system contribution.

   Evidence that would address it includes an externally designed blind mutation set, multiple historical/natural regressions, clean-workload false-positive testing, and direct comparison with output checks, persistent-state invariants, allocation assertions, and relevant sanitizer/testing baselines.

3. **Transfer and adoption evidence is too narrow for the title’s broader hybrid-LLM framing.**
   Locations: Section 4.1, p. 5; Tables 4–6, pp. 13–15; Section H, pp. 24–25.

   The primary evidence uses one Qwen3.5 architecture, one H20 family, one KV page size, one Transformers GDN implementation, one vLLM attention path, `N ≤ 32`, and primarily sequential round-major execution. The two-stream experiment establishes call-interval overlap but neither kernel overlap nor in-flight cancellation. Compiled dispatch remains partial. There is no native continuous/ragged batching, eviction, multi-document workload, second hybrid architecture, or second production backend.

   The paper also does not quantify instrumentation effort, trace volume, capture/replay latency, synchronization overhead, or the engineering work needed to adopt the contract elsewhere. A second genuinely different hybrid model/runtime, a native scheduling workload, and adoption-cost measurements are needed to establish reusable systems value.

4. **The retained-memory/CoMem result is only loosely coupled to ForkAudit and receives disproportionate prominence.**
   Locations: Abstract, p. 1; contribution list, p. 2; Section 5.3 and Table 3, pp. 8–9; Tables 14–18, pp. 21–22.

   The paper repeatedly states that the CoMem Store–F1 panel is not ForkAudit evidence. It uses an eight-item Qasper/2WikiMQA slice, a Store-only denominator excluding process memory and capacity, unpooled harnesses, and no uncertainty analysis for quality. The 54.5% allocator reduction against full-copy KV is also an expected consequence of prefix sharing rather than a new policy.

   This creates a two-paper narrative: an ownership validator plus a small deployment-memory study. Presentation can be corrected by demoting the latter to supporting context and simplifying the abstract. If retained as a headline contribution, it needs broader paired quality evaluation, per-item outcomes or uncertainty, and process/capacity measurements under a common harness.

### Minor issues

- “Independent oracle” and “out-of-process observation” can sound stronger than their actual boundaries. Figure 3, p. 7 and Table 21, p. 23 should consistently say that replay begins from producer-captured boundaries and that slot enumeration remains producer-controlled.
- Timing tables provide point summaries without dispersion or per-item values (Table 3, p. 9; Table 15, p. 21).
- The main text is dense and qualifier-heavy. Figure 3 and Table 1 on p. 7, Tables 5–6 on pp. 13–15, and Table 21 on p. 23 require substantial zoom.
- Appendix p. 19 has excessive whitespace caused by float placement, while adjacent pages contain dense tables.
- The artifact map on pp. 26–27 is thorough but organized around internal round-specific paths. A single reviewer-facing entry point, dependency list, runtime estimate, and exact replay command would improve usability.
- Appendix p. 24 describes a postexecution correction of the target-gate-suppression run identifier. The disclosure is welcome, but the relationship between that correction and the paper’s fail-closed governance promise should be explained more directly.

## Questions for the authors

1. How is the mandatory slot set \(M_i(\Sigma)\) derived independently of producer enumeration? Can the audit detect a layer, request, tensor family, or lifecycle event that the producer omits coherently?
2. What clean, valid executions were used to measure false positives, especially for conservative interval overlap and unusual tensor views?
3. How many person-hours and code changes are required to instrument a new hybrid model/runtime? What are trace size, capture overhead, replay time, and synchronization cost?
4. Can the accompanying package replay the central results without H20 hardware, and which conclusions require re-executing the model rather than checking archived receipts?
5. Why should the CoMem Store–F1 panel be a headline contribution of this ownership-audit paper?
6. In the p. 24 postexecution correction, was the canonical execution identifier already present in a pre-execution, hash-bound receipt? Could the same correction procedure ever alter a classification?
7. Beyond the historical alias, which faults are uniquely detected or materially better localized by ForkAudit than by persistent-base invariants and ordinary allocation assertions?
8. What changes would be required to cover compiled dispatch, autotuning choice, native continuous batching, and in-flight cancellation?

## Ethical concerns

No substantive ethical concern requiring a flag. The paper uses public PG-19 data, introduces no human-subject collection, and does not make safety or security-attestation claims. A residual risk is that users may overinterpret a passing audit as runtime or security assurance, but the manuscript repeatedly and appropriately disclaims that interpretation.

## Scores

- **Soundness: 3/4 — Good.** The narrowly scoped trace-relative claims are internally coherent, definitions are precise, and the evaluation contains extensive controls. The broad capture TCB and designed-fault evidence prevent an excellent score.
- **Presentation: 3/4 — Good.** The paper is readable, polished, and unusually disciplined about limitations. Density, narrative fragmentation, small tables, and the tangential deployment panel keep it below excellent.
- **Contribution: 2/4 — Fair.** The ownership-contract integration is useful, but its novelty and significance remain uncertain without cross-stack adoption, independent capture, comparative bug-finding evidence, or a broader natural-defect evaluation.

## Overall Rating

**4 — marginally below the acceptance threshold.**

The reasons to reject narrowly outweigh the reasons to accept. The contract is thoughtful and potentially useful, but current evidence does not yet establish that ForkAudit is a reusable, independently trustworthy validation capability rather than a carefully engineered consistency checker for one custom stack and its predeclared obligations.

## Confidence

**4/5.** I read and visually inspected the complete PDF and checked the central definitions, evidence boundaries, tables, and quantitative claims. Confidence is not 5 because this was deliberately a PDF-only review: I did not inspect the artifact, rerun receipts, or independently verify the broader literature.

## Score ceiling under current evidence

**6.** With the existing evidence, substantial restructuring—centering the ownership contract, demoting the CoMem panel, clarifying the TCB, and improving artifact entry points—could make the paper marginally above threshold. An 8 would require new evidence rather than presentation changes alone.

## Evidence that would raise or lower the score

Evidence that would raise the score:

- A blind or natural-defect corpus with clean controls, false-positive reporting, and comparisons to simpler invariant/testing baselines.
- Independent expected-slot enumeration or redundant capture that detects coherent producer omissions and relabeling.
- Successful deployment on a second hybrid architecture and runtime with a materially different state layout and scheduler.
- Native continuous-batching, eviction, and in-flight cancellation coverage.
- Quantified adoption and runtime costs.
- Independent replay confirming all headline receipts and the postexecution correction boundary.

Evidence that would lower the score:

- Failure of the accompanying package to reproduce reported counts or digests.
- Demonstration that producer-side omission can yield “complete” coverage.
- Valid tensor layouts producing frequent false ownership violations.
- Evidence that the postexecution correction changes classification rather than only reading an already bound identifier.
- Per-item deployment results showing that the reported mean-F1 equality hides material quality regressions.

## Final recommendation

**Marginal reject in the present form.** The paper is technically careful and merits revision, but the decisive missing evidence concerns independent assurance, unseen-fault utility, and cross-stack transfer—not wording or polish.

This review was produced by an isolated AI subagent for internal pre-submission quality control.
