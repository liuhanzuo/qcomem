## Summary and claimed contributions

ForkAudit proposes a receipt-bound audit protocol for hybrid LLM prefix sharing: it records and replays ownership, lifecycle, tail copy-on-write, recurrent-state rebinding, and Python-call contracts, supplementing output equality with storage/lifecycle evidence. The paper evaluates one Qwen3.5-35B-A3B/H20 configuration with a 2×2 KV-by-GDN ownership factorial, selected attention/GDN numerical sidecars, nine designed faults, and several explicitly unpooled support cohorts.

## Strengths

- The assurance boundary is unusually explicit. The paper repeatedly distinguishes trusted-capture replay from independent recapture, and states that compiled dispatch, native continuous batching, capacity, and end-to-end correctness remain open.
- The protocol is concrete: typed record classes, pointer-free range schema, phase-specific invariants, and a clear target-to-record matrix are valuable engineering artifacts.
- Internal numerical/count consistency is strong. Examples: 8×3×4=96 primary configurations; 96×3 comparisons=288; the N=32 allocation reductions (4.901−2.229=2.672 GiB and 4.890−2.229=2.661 GiB); and the 3,063,111-byte artifact total is 2.921 MiB.
- The authors avoid turning positive controls into a detection-rate claim. The target-gate-suppression experiment is candidly described as per-fault evidence, and M8’s sentinel is correctly excluded as a production detector.
- Presentation is polished overall: the ownership diagrams and Figure 3 communicate the paper’s structure well, and limitations are clearly surfaced rather than hidden.

## Weaknesses

### Fatal / acceptance-critical

- The main “audit” guarantee is conditional on an honest capture producer, which is also the party creating the evidence. Replay can validate trace consistency but cannot detect coherent omission, fabrication, transient mutation, or compiled-dispatch divergence. The same-process “source-distinct” observer shares candidate-created objects, labels, and the PyTorch storage API, so it does not materially resolve this central common-mode problem. This makes the central contribution closer to a self-consistency logging protocol than a strong ownership audit.
- Evidence is only a single tightly controlled stack and schedule: one model, one hardware family, BF16 KV, N≤32, one partial-tail shape, an exact 32-token transition, eight decode steps, and batch-one sequential calls. The paper calls this feasibility, but the empirical basis is too narrow to establish that the proposed protocol is broadly useful for hybrid LLM serving.
- The evidence for semantic correctness is weak relative to the ownership claims: attention checks cover eight selected rows and the GDN oracle covers four selected transitions after native normalization. Cross-arm/cross-N equality compares outputs and digests generated under the same candidate stack, so it cannot rule out common-mode semantic defects.

### Fixable / substantial

- The novelty positioning remains somewhat diffuse. Many components—recording invariants, digest replay, metamorphic equality, fault injection, and storage-range checking—are established ideas; the paper needs a sharper formal statement of what integration is new and what threat model it actually solves.
- The nine hand-designed faults are useful demonstrations but do not establish sensitivity to realistic faults, false positives, or robustness to unanticipated implementation mistakes. The gate-suppression study is particularly close to testing the designed checks against their own designed failures.
- Large unpooled context sections (CoMem/HYPIC/operator-transfer/Marconi) substantially increase length and cognitive load without strengthening the main claim. They should be moved to an appendix or sharply reduced.
- “RC” risks sounding stronger than warranted. Although defined carefully, “receipt-complete” may still be read as evidence completeness rather than completeness only within a producer-trusted trace model.

## Questions for authors

1. What is the smallest trusted computing base, and what concrete adversary is prevented if the capture producer is assumed honest? Can the method be framed as trace-consistency verification rather than an ownership audit?
2. Can the authors demonstrate a genuinely independent live observer—e.g., independently produced device/runtime instrumentation or a separately implemented capture path—on even a smaller subset?
3. How often do the selected numerical oracles fail under naturally occurring regressions or blind mutations, and what is their false-positive behavior?
4. Which parts of ForkAudit transfer unchanged across different hybrid architectures, page sizes, transition lengths, and real ragged/continuous batches?
5. Why should the many unpooled deployment and related-work cohorts be part of the main narrative rather than supplementary context?

## Numerical, evidence, and scope audit

The visible numerical claims are internally consistent. The paper appropriately separates deterministic rank/book outcomes from stochastic replication and does not claim confidence intervals. The stated 4.321× overhead is correctly bounded to five paired, one-input, 16-token calls; it is not serving overhead. The 2.23 GiB shared-KV endpoint is an allocator delta under a controlled source-retaining design, not process memory or capacity. The main limitation is not arithmetic but evidentiary independence: nearly every decisive witness is generated by the system under test or by a closely coupled observer.

## Presentation and figure/table readability

Figures 1–3 are clear and helpful. The abstract is exceptionally dense and overburdened with qualifiers/numbers. Several appendix tables—especially Tables 6, 13–15, and 23—are visually cramped at normal reading scale, with small text and long cells. Acronyms such as RR2/R28 and the proliferation of cohorts make the narrative difficult to retain. A shorter main paper centered on the contract, threat model, primary factorial, and one decisive independent validation would be much stronger.

## Recommendation

**Score: 4 / 10 — Weak Reject**  
**Confidence: 4 / 5**

I appreciate the unusually careful limitation language and the engineering rigor of the trace protocol. However, the trusted-producer boundary leaves the central audit claim fundamentally unable to establish independence, while the experimental support is too narrow and self-coupled for an ICLR acceptance. I would encourage resubmission with a sharper claim and independently captured/runtime-level validation.

Reviewed solely from :codex-file-citation{path="/tmp/forkaudit-final-pdf-only.xs72Hx/ForkAudit.pdf" purpose="source"}.
