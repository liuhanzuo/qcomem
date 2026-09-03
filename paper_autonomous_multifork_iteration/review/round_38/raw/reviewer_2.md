I reviewed only the frozen 27-page PDF (SHA-256 matched): :codex-file-citation{path="/Users/liuhanzuo/MacLLM-Bench/paper_autonomous_multifork_iteration/review/round_38/pdf_only_input/forkaudit.pdf" purpose="source"}

## Summary

ForkAudit is a fail-closed, trace-based audit protocol for forked hybrid LLM inference state. It records lifecycle, storage-range, call, and semantic receipts for shared document KV and mutable recurrent GDN state, then replays phase-specific ownership predicates. On one Qwen3.5-35B-A3B/H20 configuration, it reports six complete-and-passing targets, with dispatch provenance explicitly partial. It further provides constructed-fault sensitivity studies, a retrospective alias case, selected captured-boundary numerical oracles, and separate narrowly scoped retained-Store/F1 measurements.

## Strongest verified contribution

The strongest contribution is the unusually explicit separation of trace coverage from predicate verdicts. In particular, the paper does not convert missing evidence into a pass, and Table 6 (p. 15) clearly labels compiled-dispatch provenance as partial rather than overclaiming completeness. Within its stated trusted-capture boundary, the 4-cell × 3-fan-out × 8-book ownership study is a coherent demonstration that semantic equality can coexist with a storage-alias bug.

## Strengths

- The threat model and conditional implication are unusually candid and precise (§3, pp. 3–4). The paper clearly distinguishes an offline debugging/CI contract from security attestation.
- The ownership model is concrete: it covers partial-tail COW, document/request and request/request disjointness, transition-time rebinding, and semantic relations (§3; Figure 2, p. 5).
- The experimental protocol is carefully separated by claim domain rather than pooling unrelated cohorts (Tables 4–6, pp. 13–15).
- The constructed faults are reported responsibly as per-fault sensitivity rather than estimated recall (Table 1, p. 7; Table 8, p. 17).
- The authors are commendably careful about memory denominators. The paper explicitly avoids equating retained tensor Store with process memory, capacity, or general serving efficiency (§4.4, p. 5; §5.3, pp. 8–9).
- The PDF is visually clean and figures/tables render legibly. Figure 3 and the main tables are particularly readable.

## Weaknesses

### Critical issues

1. The central assurance remains conditional on a broad, unvalidated capture TCB. The formal result effectively says that, if the producer faithfully enumerates every mandatory event and correctly captures it, replay proves recorded predicates at registered observation points (§3, pp. 3–4). The out-of-process observer still trusts producer-side slot enumeration and CUDA-IPC semantics (§5.2, p. 8; Table 5, pp. 13–14), while Appendix H explicitly excludes coherent omission, transient writes restored before capture, OS/driver truth, and independent execution (pp. 24–25). This is honestly disclosed, but materially limits the substantive meaning of “validation” beyond the instrumented producer’s records.

2. The numerical independence is boundary-local, not end-to-end. The FP32 oracles begin from producer-captured post-RoPE or post-normalization inputs, cover selected rows, and share the capture boundary with the candidate (§5.2, p. 7; Tables 21–23, pp. 23–24). Thus they cannot rule out common-mode errors in activation construction, capture, or unobserved execution. The paper states this limitation, but the abstract and headline framing should make the restricted evidentiary status still more prominent.

### Major issues

1. External validity is very limited: one model, H20 family, one page size, BF16 KV, \(N \leq 32\), a fixed partial-tail geometry, and primarily sequential one-stream execution (§4.1, p. 5; Table 4, p. 13; Appendix H, p. 24). The two-stream study establishes only host-call interval overlap and quiescent cancellation, not kernel overlap, continuous/ragged batching, or in-flight reclamation (§5.2, p. 8). This is insufficient evidence for broad hybrid-serving applicability.

2. The fault evidence is compelling for specified gates but not for robustness. The nine primary mutants, five PDF-designed faults, and one historical bug are deliberately constructed and largely aligned with the protocol’s predicate vocabulary (§4.5, p. 5; Table 1, p. 7; Table 8, p. 17). The historical case is also caught by a conventional persistent-base invariant (§5.2, p. 8). These results establish targeted sensitivity, not evidence that ForkAudit is materially better than a well-chosen conventional invariant suite for unseen defects.

3. The retained-memory/quality claims are too small and under-characterized for their prominence. Table 3 (p. 9) uses an eight-item slice, averages F1 while reporting median Store, and gives no per-item outcomes or uncertainty. Timing repeats are three per item, while the HYPIC Store and timing/F1 results come from distinct cohorts. The narrowly worded claim is defensible, but it should be demoted or strengthened with paired per-workload results and variation.

4. Reproducibility is asserted through paths and hashes rather than independently demonstrated in the paper. The artifact map (Tables 24, pp. 26–27) is useful, but most tensors are not archived and recomputation requires the pinned environment. On the submitted evidence alone, the central replay claims cannot be independently checked.

### Minor issues

1. The manuscript is exceptionally dense and appendix-heavy. Tables 5, 13, and 24 (pp. 13–14, 20, 26–27) are useful but impose substantial cognitive load, while several appendix pages have large unused areas. A shorter claim-to-evidence table and a more selective appendix would improve reviewability.

2. The paper introduces many closely related terms—target, gate, coverage, predicate, receipt, witness, relation, and control—before a compact running example. Figure 1 helps, but a small end-to-end trace example in §3 would make the central mechanism easier to audit.

3. The source-visible correction to the target-gate-suppression aggregation identifier is disclosed (p. 24), but the paper should place a concise statement of its exact effect next to Table 1 rather than only in the appendix.

## Questions for the authors

1. What evidence would make producer slot enumeration and mandatory-event completeness independently checkable, rather than TCB assumptions?
2. Can the authors evaluate at least one substantially different hybrid architecture/runtime, compiled recurrent kernel, and native continuous/ragged batching regime?
3. How many faults outside the pre-existing predicate vocabulary, preferably naturally occurring or independently generated, can be localized relative to conventional invariants?
4. For Table 3, can the authors provide paired per-item Store/F1 values, variability, and the exact reason for combining median Store with mean F1?
5. Can compiled binary identity and autotuning choice be bound per call, allowing dispatch provenance to be complete?

## Ethical concerns

No direct human-subject concern is apparent. The paper appropriately states that its system audit is not security attestation. A residual risk is that users could nevertheless treat a pass as a runtime safety/security guarantee despite the trusted-capture boundary; stronger user-facing wording and fail-safe deployment guidance would help.

## Scores

- **Soundness: 3/4.** Internally coherent and carefully scoped, but the core evidence depends on a trusted producer/capture path and selected boundaries.
- **Presentation: 3/4.** Clear figures, strong caveats, and good table labeling; however, the document is overly dense and difficult to synthesize.
- **Contribution: 2/4.** A useful systems-audit integration with disciplined reporting, but its novelty and demonstrated value are narrow and not yet shown across realistic serving conditions.

**Overall Rating: 4 — Marginal Reject.** The paper is unusually rigorous about its limitations and contains a promising, useful audit artifact. However, the current evidence supports a narrowly instrumented fixed-stack debugging protocol, not a broadly compelling ICLR contribution about hybrid LLM serving correctness.

**Confidence: 4/5.** I am confident in the assessment of the reported evidence and stated boundaries; specialized runtime details limit confidence in judging implementation novelty.

**Score ceiling under current evidence: 6.** With the present PDF evidence, the strongest defensible outcome is a weak accept for a narrowly scoped systems-validation contribution; it does not support an 8.

Evidence that would raise the score: independent producer/capture validation; multiple hybrid stacks and compiled kernels; realistic continuous batching and in-flight cancellation; end-to-end independent numerical validation; and externally generated or natural fault evidence. Evidence that would lower it: failed artifact replay, unbound/omitted mandatory records, discrepancies in the corrected aggregation, or materially different results under expanded configurations.

**Final recommendation: Weak reject; encourage resubmission after broadening independent validation and tightening the contribution around the genuinely supported audit boundary.**

This review was produced by an isolated AI subagent for internal pre-submission quality control.
