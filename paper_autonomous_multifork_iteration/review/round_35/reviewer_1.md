(1) Summary and claimed contributions

ForkAudit is an offline, fail-closed trace-validation protocol for prefix-forked hybrid LLM inference with read-only KV and mutable GDN recurrent state. It defines phase-indexed receipts for identity, storage ownership, tail copy-on-write, recurrent rebinding, dispatch calls, and semantic equivalence. On one Qwen3.5-35B-A3B/H20 setup, it reports complete passing coverage for six of seven targets, partial Python-scope dispatch coverage, exact equivalence across 96 ownership configurations, selected numerical-oracle checks, mutation controls, a process-separated GDN observer, and one historical alias-regression reproduction/repair.

(2) Strengths

- The assurance boundary is unusually explicit. The paper clearly distinguishes trace-relative validation from security attestation and acknowledges trusted producer enumeration, capture, framework semantics, and unobserved transient behavior.
- The phase-aware ownership formulation is technically coherent: it separates immutable document state, request-private state, setup/transition/final phases, and semantic versus physical-storage predicates.
- The pointer-free witness schema and conservative byte-range overlap logic are well specified and independently replayable in principle.
- The evaluation is carefully scoped, with useful positive controls showing that token equality—and sometimes full-logit equality—can miss selected ownership violations.
- Presentation, tables, limitations, and ethics discussion are thorough and generally clear.

(3) Weaknesses

- The central evidence remains highly dependent on the trusted producer and its slot/role enumeration. Byte-bound replay validates consistency of supplied evidence, but cannot independently establish that the right tensors/events were captured or that a trace is complete; the paper acknowledges this, which substantially narrows the practical assurance gain.
- Empirical support is narrow: one model stack, H20 hardware family, a fixed partial-tail geometry, N ≤ 32, sequential batch-one execution, and selected captured-boundary numerical rows. The supplementary lifecycle work still does not demonstrate native continuous/ragged batching, arbitrary in-flight cancellation, or per-kernel concurrency.
- The fault evidence is not strong evidence of general detection capability. Most mutations are designed directly against registered predicates; the five designer–executor faults are fresh only relative to the disclosed PDF and use the same predicate vocabulary. The historical defect is one mechanism and is also detected by a conventional persistent-base invariant.
- Novelty appears primarily to be a careful integration and instrumentation protocol rather than a fundamentally new inference method or broadly general validation framework. The paper would benefit from clearer comparisons to existing systems-testing/trace-validation approaches on common defects or multiple runtimes.
- The paper is exceptionally dense and spends substantial space repeatedly restating boundaries. This precision is valuable, but the main methodological insight and the minimal adoption path are harder to extract than necessary.
- Claims of replayability are described through an artifact map, but the PDF alone does not enable assessment of artifact completeness, usability, or independent rerunning.

(4) Questions for the authors

- Can the authors evaluate the protocol on at least one additional hybrid architecture/runtime and a native continuous-batching workload, including a real scheduler lifecycle?
- What concrete mechanism prevents accidental producer-side under-enumeration of relevant state slots, beyond treating faithful enumeration as part of the TCB?
- How much runtime/storage overhead does full phase capture impose, and is there a practical reduced-receipt configuration that preserves useful detection power?
- Can the authors compare ForkAudit against a strong conventional invariant suite and a differential-testing baseline on a more realistic bug corpus, rather than only predicate-aligned injected faults?
- What portions of the artifact package can be rerun without the pinned H20 environment, and are complete raw receipts/tensors available for independent audit?

(5) Reproducibility and ethics

The PDF provides unusually detailed environment, geometry, schemas, artifact paths, and replay boundaries, which supports reproducibility in principle. However, reproducibility of the main GPU experiments still depends on a pinned large-model/H20 environment, and the paper’s key producer-capture assumptions cannot be independently checked from replayed digests alone. The ethics statement is appropriate for a systems study using PG-19 training data and appropriately notes that it does not assess broader model-safety or deployment effects.

(6) Overall score—4

Weak reject. This is a careful and technically thoughtful validation protocol with honest limitations, but its evidence is too fixed-stack, producer-TCB-dependent, and mutation-aligned to establish a broadly significant systems-validation contribution at this stage.

(7) Confidence—4

The paper is detailed enough to assess its contract and stated evidence boundaries, though confidence is limited by the inability to inspect the referenced artifact package and reproduce the large-scale experiments from the PDF.

(8) Verdict—Reject
