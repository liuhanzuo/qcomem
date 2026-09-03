Reviewed source: :codex-file-citation{path="/Users/liuhanzuo/MacLLM-Bench/paper_autonomous_multifork_iteration/review/round_38/pdf_only_input/forkaudit.pdf" purpose="source"}.

## Summary

This paper introduces ForkAudit, an offline, fail-closed trace-validation protocol for shared KV and mutable GDN state in hybrid LLM inference. It separates missing-evidence coverage from replay verdicts, checks lifecycle/storage/call/semantic receipts, and evaluates four KV-by-GDN ownership cells across 96 fixed-stack configurations. It also reports targeted fault tests, one historical alias regression, selected captured-boundary numerical checks, and a clearly separated retained-Store/F1 panel.

## Strongest verified contribution

The strongest demonstrated contribution is the explicit conditional ownership-audit contract: mandatory, phase-indexed receipts prevent a target from passing when required trace records are missing, while Table 6 keeps partial dispatch provenance visibly separate from complete targets. This is a useful and unusually careful methodology for diagnosing state-sharing regressions within its stated trusted capture/replay boundary.

## Strengths

- The threat model and conditional guarantee are unusually explicit (Sec. 3, pp. 3–4), including coherent omission, transient writes, and untrusted compiled dispatch as exclusions.
- The protocol is concrete rather than aspirational: the pointer-free witness schema and mandatory-record dependencies are precise (Appendix D, pp. 15–16).
- The experiments align with the stated contract: 96 ownership configurations, phase captures, 288 cross-N relations, and the historical alias case all directly target failures that output equality can miss (Sec. 5.2, pp. 6–8; Table 2, p. 8).
- The paper responsibly separates allocator deltas, retained tensor “Store,” process memory, and capacity; Table 3’s narrow Store–F1 claim is well labeled (Sec. 4.4, p. 5; Sec. 5.3/Table 3, pp. 8–9).
- Presentation is generally polished: Figures 1–3 (pp. 2, 5, 7) make the lifecycle and evidence flow easy to follow, and tables are legible.

## Weaknesses

### Critical issues

1. The core assurance is necessarily self-referential with respect to producer enumeration. The producer supplies the object-to-role mapping, capture IDs, storage IDs, and mandatory slots; the replay validates consistency of that representation but cannot establish that all relevant live state was faithfully observed. This is acknowledged in Sec. 3 (pp. 3–4), Table 6 (p. 15), and Appendix H (pp. 24–25), but it materially limits the practical strength of the central “ownership validation” claim. The process-separated observer still trusts producer slot selection and CUDA-IPC semantics (Sec. 5.2, p. 8).

2. Fault sensitivity is demonstrated only for constructed, predicate-vocabulary-aligned faults. The nine live mutations, five designer–executor-separated faults, and one retrospective defect are valuable positive controls, but they do not establish sensitivity to naturally occurring or out-of-vocabulary failures (Table 1, p. 7; Table 8, p. 17; Appendix H, pp. 24–25). The paper states this boundary correctly; however, it means the evidence supports bounded localization, not robust detector effectiveness.

### Major issues

1. The primary result is very narrow: one Qwen3.5/H20 configuration, one page size, BF16 KV, \(N\leq32\), a single partial-tail geometry, and batch-one sequential calls on one CUDA stream (Sec. 4.1, p. 5; Table 4, p. 13; Appendix H, p. 24). The two-stream result measures call-interval overlap rather than kernel overlap or in-flight cancellation (Sec. 5.2, p. 8). This prevents generalizing to native continuous/ragged batching or production scheduling.

2. Dispatch provenance, one of the seven targets, remains partial because neither compiled-binary selection nor autotuning choice is bound per call (Sec. 3, p. 4; Sec. 5.1, p. 6; Table 6, p. 15). This is not merely a missing presentation detail: compilation choices are part of the system whose ownership behavior is being audited.

3. The numerical checks are useful but do not independently validate end-to-end execution. Attention begins after producer-captured RoPE inputs and GDN after native normalization; even the expanded sweep covers two inputs and 12 of 30 recurrent layers (Sec. 5.2, p. 7; Tables 21–23, pp. 23–24). Thus, “exact” cross-arm agreement should not be conflated with independent model correctness.

4. The retained-Store/F1 result is based on only eight workloads and reports an intentionally narrow payload metric that excludes metadata, pools, process memory, and capacity (Sec. 5.3/Table 3, pp. 8–9). The authors label this correctly, but its prominence risks distracting from the audit contribution without providing strong deployment evidence.

### Minor issues

1. The appendix repeatedly restates limitations and non-comparability boundaries. A shorter claim-to-evidence map in the main paper would improve readability (Appendix C–I, pp. 13–27).

2. Page 19 is visually sparse, and the paper’s long sequence of contextual tables dilutes the central audit story. This is correctable presentation rather than an evidence issue.

3. “Exact” is carefully defined (Sec. 3, p. 4), but “exact semantics” in the results can still be misread as end-to-end or device-tensor equivalence rather than equality of registered serializations.

## Questions for the authors

1. How can an independent deployment auditor validate faithful producer slot enumeration rather than only replaying producer-declared slots and roles?

2. What additional interface or capture support would be required to bind compiled dispatch and autotuning choices per call?

3. What is the runtime/storage overhead of full synchronous capture, and which subset of receipts would be viable in routine CI?

4. Which kinds of real integration failures would ForkAudit detect beyond the current handcrafted predicate vocabulary, and how would false positives be assessed?

5. Can the authors demonstrate the contract on a native continuous/ragged-batching runtime with in-flight cancellation, rather than a sequential or post-synchronization schedule?

## Ethical concerns

No material human-subject concern is apparent. The paper appropriately states that it does not assess model safety, privacy, bias, or deployment effects. Because more reliable cache sharing can facilitate scaling, deployment implications and the licensing/privacy status of benchmark text should remain part of any broader release discussion.

## Scores

- **Soundness: 2/4.** The conditional trace argument is internally coherent and carefully scoped, but the central evidence depends on trusted producer capture and remains incomplete for compiled dispatch and realistic schedules.
- **Presentation: 3/4.** Clear figures, readable tables, and disciplined caveats; nevertheless, the appendix is overly dense and repetitive.
- **Contribution: 2/4.** A useful systems-testing integration and reporting discipline, but not yet evidence of a broadly effective ownership validator across runtimes or failure modes.
- **Overall Rating: 4 — Weak Reject.** The paper is careful and potentially useful, but current evidence supports a bounded engineering case study rather than the broader methodological impact expected for acceptance.
- **Confidence: 4/5.** The paper is unusually explicit about scope, making its strengths and limitations assessable from the manuscript.

**Score ceiling under current evidence:** 6 (Weak Accept). The careful claims, trace schema, and controls could justify that ceiling, but the current single-stack and trusted-capture evidence do not support a stronger score.

Evidence that would raise the score: independently selected/enumerated capture witnesses; per-call compiled-dispatch binding; artifact-backed re-execution; broader runtime/model/schedule coverage including native batching and in-flight cancellation; and a preregistered set of realistic or unseen defects.

Evidence that would lower the score: inability to reproduce the reported traces, evidence that producer-side enumeration can omit relevant state without detection under ordinary failures, or any clean production schedule that passes while exhibiting an ownership violation.

**Final recommendation: Weak Reject.**

This review was produced by an isolated AI subagent for internal pre-submission quality control.
