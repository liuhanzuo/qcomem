### (1) Summary and claimed contributions

The paper introduces ForkAudit, a trace-based validation protocol for ownership isolation in hybrid KV-cache and recurrent (GDN) state when decoding branches fork from a shared prefix. It combines phase-indexed capture, fail-closed replay predicates, storage-range witnesses, selected captured-boundary numerical oracles, and mutation-based controls. On one fixed Qwen3.5/H20 configuration, it reports complete passing evidence for six ownership/semantic targets and partial Python-scope dispatch evidence; it explicitly limits assurance to honest, complete producer capture and declared observation points. This review is based solely on the submitted PDF.

### (2) Strengths

- The problem is real and clearly motivated: equality of generated tokens or logits cannot establish non-aliasing of mutable branch state.
- The paper is unusually explicit about its assurance boundary. It separates trace coverage from replay verdicts and repeatedly disclaims malicious-producer, compiled-kernel, transient-write, OS/driver, and broad serving claims.
- The contract is concrete and technically legible: lifecycle phases, conservative storage intervals, tail-COW obligations, and cross-arm/cross-fan-out relations are well specified.
- The evaluation includes meaningful positive controls, a gate-suppression analysis showing several ownership faults preserve outputs, and a separately frozen five-fault campaign.
- Presentation and artifact mapping are strong. Tables clearly authorize narrow inferences, and reproducibility limitations are candid.

### (3) Weaknesses

- The central result remains conditional on a trusted producer that enumerates slots and captures state correctly. Thus the protocol can validate consistency of a producer-generated trace, but does not independently establish the live runtime ownership property it is meant to audit.
- Novelty appears primarily to be a careful integration of tracing, metamorphic relations, hashing, storage overlap checks, and targeted mutations. The paper does not convincingly distinguish this integration from a rigorous engineering validation framework rather than a substantial new ML/systems-method contribution.
- Empirical evidence is highly narrow: one model, one hardware family, sequential batch-one decoding, one main prefix/transition geometry, and no native continuous or ragged batching. Supporting cohorts do not substantially remedy this limitation.
- The fault evidence is informative but not strong evidence of general detection capability: original faults are designed around named gates, and the held-out set has only five deliberately constructed mutations. There is no evaluation on naturally occurring bugs, false positives, or comparison against credible alternative testing/auditing baselines.
- Full capture and replay overhead, storage cost, and operational feasibility for CI or production debugging are asserted to be costly but not quantified. This omission matters for a proposed validation workflow.
- Numerical validation begins from producer-captured intermediate inputs and covers selected boundaries; it cannot independently validate upstream construction or end-to-end execution.

### (4) Questions for the authors

- What concrete threat model makes trusted producer-side slot enumeration acceptable, and how would ForkAudit be deployed where the runtime itself may be faulty in precisely that enumeration?
- Can the authors quantify capture/replay time, memory, artifact size, and engineering overhead relative to ordinary inference and existing testing workflows?
- Can the method be evaluated on known historical cache/state bugs or externally sourced bugs, with comparisons to output equivalence, assertions, differential testing, and conventional tracing?
- What modifications are needed for continuous batching, asynchronous scheduling, and optimized/fused recurrent kernels, and which core invariants remain applicable?
- Why should the five PDF-only mutations be considered sufficiently independent of the proposed predicates, given that their expected first failing predicates are part of the frozen protocol?

### (5) Reproducibility and ethics

The paper provides a detailed artifact and claim map, manifests, replay descriptions, pinned environments, and an honest explanation of which artifacts cannot independently rerun the model. Reproducibility of the reported trace-replay results appears reasonably supported, but independent reproduction of the live-system conclusions remains limited by the required hardware, proprietary/external model stack, and trusted capture path. The ethics discussion is adequate for a systems study using PG-19 data; the paper appropriately notes that lower memory costs can increase deployment scale, though it does not study downstream impacts.

### (6) Overall score—state exactly one allowed integer and concise justification

4 — Weak reject. The paper is careful, technically thoughtful, and valuable as a bounded auditing protocol, but its trusted-capture assumption, limited independent validation, absence of operational overhead evidence, and narrow fixed-stack evaluation leave insufficient evidence of broad significance or methodological novelty for acceptance.

### (7) Confidence—state one integer 1–5 and concise justification

4 — The paper is detailed and its stated scope is clear, enabling a confident assessment; uncertainty remains about how much novelty and systems significance the venue should assign to this validation-framework contribution.

### (8) Verdict—Accept or Reject

Reject
