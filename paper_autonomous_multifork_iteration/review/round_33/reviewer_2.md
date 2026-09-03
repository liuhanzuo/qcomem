(1) Summary and claimed contributions

ForkAudit is a trace-validation protocol for forking hybrid LLM inference state: shared document KV plus mutable GDN recurrent state. It combines phase-indexed ownership/lifecycle receipts, pointer-free storage-range checks, call-contract evidence, semantic equality across a 2×2 KV-by-GDN ownership factorial and fan-outs, selected captured-boundary NumPy oracles, and targeted fault campaigns. On a fixed Qwen3.5/H20 stack, it reports passing conditional trace verdicts for six complete targets and one Python-scope-only dispatch target.

(2) Strengths

- The paper crisply distinguishes output equivalence from physical ownership isolation, and formalizes a useful fail-closed coverage condition.
- The threat model and limitations are unusually explicit: the authors clearly state reliance on an honest, complete producer capture, selected observation points, and the absence of OS/driver or compiled-dispatch guarantees.
- The evaluation is detailed and internally coherent: 96 factorial configurations, cross-N relations, lifecycle extensions, selected independent numerical replays, and both original and separately designed fault controls.
- The storage-witness schema is concrete and potentially useful in practice, especially its handling of strided views, transition-time rebinding, partial-tail COW, and separate memory denominators.
- Presentation is polished, with tables consistently separating primary evidence from contextual or supporting cohorts.

(3) Weaknesses

- The central guarantee is fundamentally conditional on the same producer’s complete and faithful slot enumeration, storage IDs, phase labels, and captured inputs. The external observer still consumes producer-selected GDN slots and trusted CUDA IPC. Thus the paper demonstrates replay consistency of a carefully instrumented system, not independent validation of the underlying runtime ownership behavior.
- Empirical support is narrow: one model family, one H20 configuration, sequential batch-one calls, one primary partial-tail geometry, N≤32, and a Torch GDN implementation. The extensive appendices make these limitations clear, but they substantially limit the claimed method’s demonstrated generality and practical serving relevance.
- The fault evidence is mostly expected-gate testing of deliberately constructed mutations. The “PDF-only held-out” faults improve the protocol, but remain five author-designed fixed-stack mutations; this does not demonstrate detection of natural defects, characterize false positives/negatives, or compare against realistic alternatives.
- Novelty is primarily the integration and engineering of established trace validation, metamorphic relations, storage reasoning, and positive controls. The paper itself disclaims novelty for individual mechanisms, yet does not convincingly show that the integrated framework is readily adoptable, low-overhead, or superior to a simpler instrumentation/test suite.
- The numerical oracle begins after producer-captured post-RoPE or post-normalization inputs, so it cannot validate upstream construction or independently corroborate the end-to-end model. This is appropriately disclosed but weakens its role as an independent check.

(4) Questions for the authors

- What independent mechanism, if any, can validate producer-side slot enumeration and the mapping from live storages to normalized storage IDs? Can the protocol support a threat model stronger than an honest capture producer?
- What is the runtime, memory, synchronization, and engineering overhead of mandatory full capture in a realistic debugging/CI workflow?
- Can the method be evaluated on native continuous/ragged batching and an optimized hybrid serving runtime, where the ownership risks and performance constraints are most consequential?
- How would ForkAudit compare with a substantially simpler invariant-based test suite or existing runtime assertions on naturally occurring regressions?
- Which elements of the protocol are reusable across model architectures and serving engines without bespoke instrumentation?

(5) Reproducibility and ethics

The PDF provides an unusually detailed artifact/claim map, environment versions, schemas, frozen geometry, and replay boundaries, which supports reproducibility in the pinned environment. However, the results remain difficult to independently validate without the referenced package and specialized H20/CUDA environment; most primary evidence is hashes/receipts rather than archived tensors. Ethics discussion is adequate for a systems study using PG-19, though it appropriately does not establish broader model safety, privacy, or deployment effects.

(6) Overall score—state exactly one allowed integer and concise justification

4. The paper is careful, technically thoughtful, and potentially useful as a systems-assurance protocol, but the contribution is mostly an integrated, highly system-specific auditing methodology whose core evidence remains conditional on trusted producer instrumentation. The narrow evaluation and lack of natural-bug or practical-overhead evidence make acceptance premature.

(7) Confidence—state one integer 1–5 and concise justification

4. The paper is exceptionally explicit about its scope, protocol, and limitations, enabling a confident assessment; uncertainty remains about the practical novelty and utility that only external artifacts or broader deployment evidence could resolve.

(8) Verdict—Accept or Reject

Reject
