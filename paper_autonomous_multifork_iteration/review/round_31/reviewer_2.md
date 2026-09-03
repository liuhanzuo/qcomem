## 1. Summary and claimed contributions

ForkAudit is a trace-replay validator for ownership of hybrid LLM inference state: shared read-only document KV, request-private KV append pages, and mutable GDN recurrent state. It records phase-indexed lifecycle, storage-range, call-contract, semantic, and artifact-binding receipts, then replays predicates under an explicit honest, mandatory-event-complete capture-producer assumption. The paper evaluates one Qwen3.5-35B-A3B/H20 implementation with a four-cell KV-by-GDN ownership factorial, targeted faults, selected captured-boundary numerical oracles, and memory/overhead measurements. It carefully limits its claim to trace-relative validation at declared observation points.

## 2. Strengths

- The assurance boundary is unusually clear. The paper repeatedly distinguishes trace replay from independent live recapture, compiled-dispatch attestation, end-to-end correctness, and a fault-detection rate.
- The ownership model usefully addresses a real hybrid-state distinction: sharing immutable KV is not equivalent to safely sharing mutable recurrent state. The phase-aware rebinding and partial-tail COW obligations are concrete and well specified.
- Evaluation reporting is disciplined. The paper separates coverage from replay verdicts, avoids pooling heterogeneous cohorts, names memory denominators, and reports the non-serving 4.321x live-capture cost.
- The targeted fault and gate-suppression experiments are informative within their intended scope: several designed ownership/call faults preserve tokens and, in four cases, exact canonical logits, supporting the argument that output equality is insufficient.
- Reproducibility is thoughtfully documented through manifests, schemas, hashes, artifact paths, and disclosed replay-only corrections.

## 3. Weaknesses

- The central limitation is also central to the contribution: a producer trusted to capture every event faithfully can coherently omit or fabricate the same receipts that replay validates. Hash bindings protect archived artifacts after capture, not the truthfulness or completeness of capture. Thus the main result establishes consistency of a self-attested trace, rather than independently validating the runtime ownership behavior. The paper discloses this honestly, but it substantially limits the scientific strength of “validation.”
- The novelty appears primarily to be an engineering integration of known tracing, COW, hashing, storage introspection, metamorphic relations, and selected differential checks. The paper does not convincingly show that the proposed schema or factorization yields capabilities unavailable from a well-engineered conventional runtime tracer/test suite, beyond the tailored nine-fault suite.
- The sensitivity evidence is narrow and target-aligned: faults are designed around preregistered gates, all-gates-on runs stop before semantics, and suppressed-gate cases are not held-out tests. This appropriately is not called a detection rate, but it leaves no evidence on realistic, unexpected, or adversarially trace-consistent failures, false positives, or false negatives.
- External validity is very limited: one model, one hardware family, one page size and KV precision, one partial-tail geometry, a fixed 32-token transition, short eight-token continuations, batch-one sequential calls, and one stream per rank. The separate two-stream experiment does not show concurrent kernels or in-flight cancellation safety.
- The numerical checks depend on producer-captured intermediate inputs and cover selected boundaries only (20 attention rows and 24 GDN rows, with 12/30 GDN layers). They cannot validate upstream activations, capture correctness, or end-to-end behavior.
- The memory and timing results are useful characterization but not strong systems evidence: the full-copy controls are explicitly not optimized production baselines, allocator values are not process memory/capacity, and the overhead experiment has five paired measurements on one frozen request step.

## 4. Questions for the authors

1. What realistic deployment or debugging threat model makes an honest, mandatory-complete producer assumption acceptable, and what failure modes remain detectable if the candidate implementation and capture instrumentation share bugs?
2. Can the authors provide an independently collected observation path—for example, lower-level allocator/kernel instrumentation or an external capture process—and quantify how often it disagrees with ForkAudit receipts?
3. How does ForkAudit compare against a strong baseline consisting of lifecycle assertions, PyTorch storage checks, and conventional differential tests? An ablation showing which trace obligations discover which non-tailored faults would clarify the incremental value.
4. Why is the registered 32-token transition representative? Does the approach remain practical and correct under varied transition lengths, tail geometries, ragged batching, and longer decoding?
5. What is the expected artifact replay and capture cost for a practical CI workflow beyond the single reported fixed configuration?

## 5. Reproducibility and ethics

The submission provides unusually detailed replay instructions, artifact maps, manifest/hash descriptions, and disclosed corrections. However, based on the PDF alone, reproducibility remains unverified externally and requires a substantial pinned environment and, for primary capture, eight H20 GPUs. Some supporting context lacks raw trials, and most runtime tensors are represented by hashes rather than full archived values. The trusted-capture assumption also limits reproducibility as independent verification of the underlying execution.

The ethical discussion appropriately states that no human-subject study is conducted and acknowledges that efficiency can facilitate larger-scale deployment. It does not evaluate model safety, privacy, bias, or deployment impacts; this is acceptable for the bounded systems study but means the work provides little evidence on those dimensions.

## 6. Overall score (integer 1–10, using ICLR-style meaning; state score and concise justification)

**4/10 — Borderline reject.** This is a careful and technically detailed systems-validation case study with commendable limitation disclosure, but its evidence remains self-attested, highly implementation-specific, and narrowly fault-targeted. The methodological novelty and general scientific evidence seem insufficient for ICLR acceptance.

## 7. Confidence (integer 1–5; state confidence and concise justification)

**4/5.** The PDF clearly defines the protocol, data scope, limitations, and artifacts, enabling a confident assessment of the stated contribution. Confidence is not maximal because the underlying artifacts and execution cannot be independently checked from the submission PDF.

## 8. Verdict (Accept or Reject)

**Reject**
