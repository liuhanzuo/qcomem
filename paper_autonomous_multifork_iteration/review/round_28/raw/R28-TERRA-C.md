# R28-TERRA-C

## Summary

ForkAudit is a receipt-and-replay audit contract for forked decoding in hybrid LLMs, where shared attention KV and mutable recurrent GDN state create ownership risks not exposed by matching outputs. On one Qwen3.5-35B-A3B stack, it records lifecycle/storage/call receipts, checks four KV×GDN ownership configurations across \(N\in\{1,8,32\}\), runs limited independent operator oracles, and uses nine designed faults. The paper carefully limits its claim to trusted-capture, sequential, fixed-stack feasibility.

## Strengths

- Clear motivation: equality of tokens/logits does not establish isolation of mutable state.
- Unusually explicit threat model, evidence hierarchy, denominators, and limitations.
- Strong engineering specification: typed records, pointer-free range schema, replay predicates, and artifact map.
- Sensible factorial isolation of KV and GDN ownership; selected FP32 attention and independent NumPy recurrence checks provide some corroboration beyond self-consistency.
- Positive controls and target-gate suppression distinguish localization from detection rate rather than overstating results.
- The presentation is polished, legible, and internally consistent.

## Weaknesses

### Must-fix

- The central evidence is fundamentally conditional on an “honest capture producer.” Since storage IDs, content digests, lifecycle events, and dispatch receipts all originate from that producer, replay largely verifies consistency of its trace, not actual runtime ownership. This limitation is acknowledged, but it substantially narrows the scientific force of the claimed audit.
- Empirical validation is narrow and largely constructed around the proposed gates: one model/configuration, sequential one-stream execution, one partial-tail geometry in the primary study, selected operator rows, and nine designed faults. There is no independent recapture, unseeded/real bug evaluation, concurrent execution, or comparison against a credible alternative audit/testing workflow.
- The paper’s main experimental conclusion is therefore closer to “this instrumentation can witness its specified predicates in one stack” than a general validation of ForkAudit. The paper should sharpen the contribution and provide materially stronger external validation or empirical evidence of bugs missed by conventional tests and caught by the audit.
- The disclosed post-execution repair to the run-ID wrapper weakens the paper’s emphasis on preregistration/governance. Even if candidate bytes were unchanged, the authors should explain why the original schema omission was not caught by the frozen validation process and make the provenance chain independently auditable.

### Optional improvements

- Substantial unpooled CoMem/HYPIC/related-work context makes the paper long and distracts from its core audit result. Move more of it to the appendix.
- Explain more concretely what deployment users should implement, its runtime/storage overhead, and how the method compares with existing tracing, invariant checking, sanitizers, or differential testing.
- Better distinguish a logical ownership contract from evidence of physical allocator/kernel behavior; “receipt-complete” terminology risks sounding stronger than the actual trust boundary.

## Questions for authors

1. What independent observer, hardware-level mechanism, or adversarial capture evaluation could reduce reliance on the honest-producer assumption?
2. Can you demonstrate a naturally occurring or held-out defect, ideally under concurrency/continuous batching, that ForkAudit detects and existing invariants or differential tests do not?
3. What are capture overheads (latency, memory, trace size, engineering burden) during production-like decoding?
4. Why did the run-ID schema defect survive the frozen/pre-output governance process, and what guarantees show the post-execution wrapper cannot affect interpretation beyond the stated field?
5. Which components are genuinely new versus a careful integration of tracing, ownership invariants, metamorphic relations, and targeted fault injection?

## Reproducibility assessment

Good for offline trace replay: the PDF specifies a manifest-first package, artifact counts/footprint, replay scope, schemas, decision rules, and an artifact map. Moderate for end-to-end scientific reproduction: most tensors are represented by hashes, numerical recomputation needs the pinned model/data environment, and the design explicitly does not permit independent producer recapture or compiled-dispatch attestation.

## Rating

**Overall score: 5 / 10 (Borderline; weak reject).** The work is careful, technically articulate, and responsibly scoped, with a useful systems-auditing perspective. However, the core assurance remains self-reported under a trusted-capture assumption, and validation is too narrow and gate-designed to establish broad practical or scientific impact at ICLR.

**Confidence: 4 / 5.**

