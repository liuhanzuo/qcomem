1. Summary and claimed contributions

ForkAudit is a fail-closed, phase-indexed trace-validation protocol for shared KV caches and mutable GDN recurrent state after request forking. It records ownership, lifecycle, tail-COW, call-contract, storage-range, and semantic receipts; replay checks seven targets, separating trace coverage from verdicts. The paper evaluates one Qwen3.5/H20 configuration across 96 ownership/fan-out cells, selected captured-boundary numerical oracles, targeted and PDF-only designed fault campaigns, limited IPC observers, and memory endpoints.

2. Strengths

- The assurance boundary is unusually explicit. The paper correctly distinguishes matching outputs from storage ownership, clearly labels conditional assumptions, and does not overclaim OS/driver-level truth or end-to-end correctness.
- The protocol is technically well specified: phase-conditioned invariants, pointer-free storage intervals, mandatory-record semantics, and explicit treatment of partial compiled-dispatch coverage are valuable.
- Empirical reporting is careful: ownership factors, memory denominators, supporting cohorts, and non-pooled contextual results are separated rather than conflated.
- The designed-fault experiments are more informative than output-only checks and show that several injected faults preserve tokens and even logits.
- The paper is generally clear and professionally presented despite its density.

3. Weaknesses

- The central empirical evidence is a single highly controlled stack, sequential batch-one scheduling, one main partial-tail geometry, and \(N \leq 32\). This is insufficient to establish that the methodology is broadly useful for realistic serving systems with ragged/continuous batching, eviction, concurrency, other hybrid models, or optimized recurrent kernels.
- Much of the claimed novelty is integration and engineering discipline rather than a new validation principle. The paper does not experimentally compare ForkAudit’s defect-finding value, overhead, or adoption cost against credible existing testing, tracing, or runtime-invariant baselines.
- The primary evidence remains producer-generated and assumes honest, mandatory-event-complete capture. The IPC observer still trusts producer slot enumeration and CUDA/PyTorch semantics, while numerical checks begin from candidate-captured intermediate inputs. These are reasonable stated boundaries, but materially limit independent validation.
- Fault sensitivity is based on bespoke, deliberately constructed mutations. The held-out PDF-only process is interesting, but five fixed mutants cannot support meaningful evidence about naturally occurring bugs, recall, false positives, or general detection advantage.
- Reproducibility is described in detail, but the PDF provides relative artifact paths rather than an accessible anonymous artifact/repository, and it states that no package independently re-executes the model. Consequently, key empirical claims cannot be independently confirmed from the submission as provided.
- The paper is very dense and uses round labels such as RR2/R28/R30/R33 extensively; this obscures the main contribution and makes the evaluation narrative feel closer to an internal audit log than a focused research paper.

4. Questions for the authors

- What is the measured runtime, memory, and engineering overhead of mandatory capture and replay, and which portions are required for practical CI deployment?
- Can ForkAudit detect real historical bugs or independently sourced bugs, and how does it compare with output metamorphic testing, assertions, or existing tracing approaches under a common fault corpus?
- How would the contract and storage witness handle continuous/ragged batching, eviction, in-flight cancellation, multi-document scheduling, and optimized recurrent kernels?
- What concrete mechanism establishes honest and complete producer-side slot enumeration in a deployment setting, beyond the stated assumption?
- Will the complete artifact, frozen manifests, replay commands, and environment be anonymously available to reviewers, including enough data to reproduce the central tables?

5. Reproducibility and ethics

The protocol specification, fixed geometry, schemas, equations, and replay boundaries are detailed. However, reproducibility is only partial from the visible submission: the actual artifacts are not accessible here, model execution is not independently reproduced, and core observations depend on trusted capture. The ethics statement is appropriate for a systems study using PG-19 training data and acknowledges that memory efficiency can enable greater deployment scale. No major additional ethics issue is apparent.

6. Overall score—state exactly one allowed integer and concise justification

4. The paper presents a thoughtful and carefully scoped audit methodology with strong documentation, but its novelty is largely integrative and its evidence is too narrow, producer-dependent, and baseline-light to justify acceptance at ICLR.

7. Confidence—state one integer 1–5 and concise justification

4. The paper is explicit enough to assess its technical boundaries and empirical scope, though unavailable artifacts prevent direct verification.

8. Verdict—Accept or Reject

Reject
