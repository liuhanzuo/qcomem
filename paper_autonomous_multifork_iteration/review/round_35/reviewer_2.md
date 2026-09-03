(1) Summary and claimed contributions

ForkAudit is a fail-closed, offline trace-validation protocol for ownership isolation when multiple continuations share a prefix in hybrid attention/GDN LLMs. It records phase-indexed lifecycle, storage-range, call-contract, digest, and semantic receipts, then replays predicates for identity, prefix immutability, private ownership, tail COW, dispatch provenance, cross-arm equivalence, and cross-fan-out consistency. On one Qwen3.5/H20 stack, it reports six fully covered passing targets and one dispatch target that is explicitly only partially covered. Supporting evidence includes selected captured-boundary FP32 oracles, constructed fault injections, an out-of-process CUDA-IPC relation check, lifecycle tests, and one retrospective aliasing regression.

(2) Strengths

- The paper identifies a real systems-validation gap: identical tokens/logits do not establish correct mutable-state ownership.
- The contract is unusually explicit about its assurance boundary, mandatory evidence, coverage versus verdict distinction, and trusted computing base. This substantially improves clarity and prevents overclaiming.
- The pointer-free storage witness and phase-conditioned ownership rules are concrete, technically sensible, and useful for CI-oriented debugging.
- Evaluation is carefully structured: 96 ownership cells, selected numerical controls, targeted mutations, and a historical defect that preserves conventional output observables.
- Limitations are candid and extensive; the authors avoid presenting the work as runtime attestation, a general security mechanism, or a serving-performance result.

(3) Weaknesses

- The central conclusion is conditional on a very large trusted producer/capture path, including faithful slot enumeration, framework storage semantics, and absence of omitted/transient observations. The external observer still trusts producer-selected slots and CUDA IPC. Thus the method validates recorded evidence more than it independently validates runtime ownership.
- Empirical scope is narrow: one model/configuration, one hardware family, one page size, sequential batch-one execution, limited fan-out, and a fixed transition geometry. The key problem is broadly motivated, but the evidence does not establish portability to realistic continuous batching, eviction, arbitrary lifecycle transitions, optimized recurrent kernels, or multi-document serving.
- The semantic checks are mainly metamorphic comparisons among implementations/arms sharing substantial system components. The independent numerical checks begin after producer-captured boundaries and cover only selected operator rows, so they cannot rule out common-mode errors in upstream activation construction or capture.
- Fault sensitivity remains difficult to interpret. Most faults are constructed from the audit’s own predicate vocabulary; the held-out campaign withholds them from the executor, not from the method design. The retrospective case is valuable but is a single mechanism and is also detected by a persistent-base invariant.
- The paper is dense and lengthy for its central contribution. The numerous supporting cohorts, context tables, and claim-boundary caveats make the main methodological novelty harder to isolate. A more compact presentation centered on the protocol, a small number of decisive experiments, and a clearer comparison to existing systems-testing approaches would improve impact.
- Reproducibility claims reference an accompanying package and manifests, but the PDF alone cannot establish that these artifacts are available, complete, or runnable.

(4) Questions for the authors

- Can the authors evaluate ForkAudit under native continuous/ragged batching and at least one optimized recurrent-state implementation, where ownership and scheduling failures may differ materially?
- What is the runtime/storage overhead of mandatory capture and replay as a function of fan-out and context length, and what practical CI cadence does that imply?
- Can the TCB be reduced or independently cross-checked—for example, through runtime-level allocation instrumentation, independent slot enumeration, or sampling designed to expose transient writes?
- How does the method compare against a strong baseline composed of conventional persistent-base immutability checks, allocator assertions, and differential testing on the same defect suite?
- Will the authors release the complete raw receipts, frozen manifests, replay environment, and all fault implementations needed to reproduce every table?

(5) Reproducibility and ethics

The paper gives unusually detailed versions, geometry, record schemas, artifact paths, boundaries, and replay descriptions, which is a strong reproducibility foundation. However, without inspecting an accompanying artifact, reproducibility remains unverified; many conclusions depend on hashes/receipts rather than retained tensors and on a pinned environment. Ethics treatment is appropriate for a systems study using PG-19 training data, though broader implications of enabling more scalable deployment are only briefly discussed.

(6) Overall score—4

Weak reject. This is careful, technically thoughtful systems-validation work with strong reporting discipline, but its contribution is largely an integration of known testing/ownership concepts and its evidence remains narrowly trace-relative and dependent on a broad TCB. I would be more positive with stronger independent validation and realistic serving coverage.

(7) Confidence—4

High confidence in the assessment of the paper’s stated scope and evidence because the limitations and protocol boundaries are explicit. Some uncertainty remains because artifact availability and execution cannot be assessed from the PDF alone.

(8) Verdict—Reject
