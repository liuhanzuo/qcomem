1. Summary and claimed contributions

ForkAudit is an offline, fail-closed trace-validation protocol for ownership correctness when LLM requests fork from a shared prefix with both paged KV state and mutable GDN recurrent state. It records phase-indexed storage, lifecycle, call, and semantic receipts; replays pointer-free overlap/disjointness predicates; and evaluates a KV-by-GDN ownership factorial. On one Qwen3.5/H20 configuration, it reports semantic equivalence across configurations, selected captured-boundary numerical-oracle checks, designed fault localization, and limited process-separated observation.

2. Strengths

- The paper is unusually explicit about its assurance boundary, TCB, incomplete compiled-dispatch coverage, and the distinction between coverage and a passing replay verdict.
- The phase-aware ownership specification is clear and technically sensible: it separates prefix immutability, tail COW, request/base and request/peer isolation, and semantic equality.
- The pointer-free storage-witness schema and conservative byte-interval replay are concrete, reproducible ideas.
- Evaluation includes positive controls and gate-suppression experiments demonstrating that tokens, and sometimes full logits, can miss selected ownership violations.
- The appendices are comprehensive and candid about limited scope; presentation and tables are generally polished.

3. Weaknesses

- The empirical evidence is fundamentally self-referential: the producer enumerates the slots, captures the evidence, and supplies the inputs on which replay depends. The process-separated observer does not materially remove this central dependency. Thus the work validates recorded observations under a strong TCB rather than independently validating runtime behavior.
- The main result is restricted to one model, one hardware/software stack, sequential batch-one execution, a narrow range of fan-outs, one principal prefix geometry, and one transition point. This is insufficient to establish broad relevance for real hybrid serving systems.
- The numerical checks begin after producer-captured RoPE or normalization boundaries, cover selected layers/rows, and do not independently validate upstream activations or end-to-end computation.
- Mutation evidence is targeted to the proposed predicates. The five designer–executor-separated faults are a better control, but remain constructed fixed-stack mutations and do not establish detection of realistic, unforeseen regressions or compare against existing testing/debugging practice.
- The methodological novelty is primarily a careful integration of established testing, lifecycle, and storage-accounting concepts. The paper lacks a compelling comparison showing that this protocol finds real bugs or provides practically superior assurance relative to conventional invariant tests, differential testing, or runtime instrumentation.
- Reproducibility is described in detail, but the paper itself provides no artifact-access evaluation; many results rely on hashes rather than retained tensors and substantial pinned hardware/software infrastructure.

4. Questions for the authors

- Can the authors demonstrate ForkAudit finding an organically occurring bug, or compare its detection/localization value against a strong conventional test suite on the same faults?
- What independent mechanism could validate producer-side slot enumeration and capture completeness, rather than treating both as part of the TCB?
- How does the protocol scale under native continuous/ragged batching, arbitrary transition lengths, multiple documents, and optimized recurrent kernels?
- Which portions of the method are intended as generally reusable software versus stack-specific instrumentation for this adapter?

5. Reproducibility and ethics

The manuscript provides unusually detailed manifests, schemas, paths, replay scopes, and limitation statements, so its bounded offline replay appears potentially reproducible for readers with the pinned environment. However, full independent reproduction is hardware-intensive and cannot independently verify the producer capture or unarchived tensor state. The ethics statement appropriately notes the absence of human-subject work and the limited systems focus; no major additional ethical concern is apparent.

6. Overall score—state exactly one allowed integer and concise justification

4 — The paper presents a careful and useful engineering validation protocol with strong transparency, but its central evidence is conditional on a broad shared capture TCB, its novelty is largely integrative, and the narrow self-contained evaluation does not yet justify acceptance as a broadly impactful ICLR contribution.

7. Confidence—state one integer 1–5 and concise justification

4 — The paper’s scope, assumptions, protocol, results, and limitations are described in sufficient detail to assess its contribution, though artifact execution was not part of this review.

8. Verdict—Accept or Reject

Reject
