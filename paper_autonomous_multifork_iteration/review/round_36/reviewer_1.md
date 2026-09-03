(1) Summary and claimed contributions

This paper presents ForkAudit, a fail-closed, phase-aware trace-validation protocol for forked hybrid LLM inference with shared KV cache and mutable GDN recurrent state. It records lifecycle, storage-range, COW, transition, call-contract, and semantic receipts, distinguishes coverage from verdicts, and validates them offline. On one Qwen3.5-35B-A3B/H20 setup, it reports complete passes for six targets, partial dispatch provenance, exact cross-cell outputs/state across 96 configurations, constructed-fault gate rejections, selected operator-level numerical checks, five designer/executor-separated faults, and a retrospective alias-bug reproduction/repair.

(2) Strengths

- The assurance boundary is unusually explicit. The paper clearly states that results are conditional on capture completeness, producer enumeration, framework storage semantics, and registered observation points.
- The distinction between output equality and ownership correctness is important and well motivated, especially for hybrid mutable state.
- The protocol is carefully specified: phase-dependent obligations, a pointer-free interval schema, fail-closed treatment of missing records, and separate coverage/verdict reporting are all clear.
- The evaluation is extensive within its declared fixed-stack scope. The paper includes matched controls, targeted mutations, a historical regression, selected independent NumPy replays, and useful lifecycle extensions.
- Limitations are candid and substantive rather than buried; the paper does not improperly present the designed faults as a general detection-rate estimate.

(3) Weaknesses

- The main technical novelty appears primarily to be a rigorous integration of established testing, tracing, COW, differential/metamorphic, and storage-witness ideas. The paper does not provide a strong conceptual result showing why this particular contract is sufficient beyond its explicit TCB, nor comparisons to realistic alternative auditing workflows.
- Empirical support is narrow: one model, one hardware family, a sequential batch-one schedule, one primary partial-tail geometry, and only selected operator boundaries. The paper responsibly discloses this, but it substantially limits significance and practical transferability.
- Most evidence is producer-captured hashes and receipts. The process-separated observer still trusts producer-side slot selection and CUDA/PyTorch IPC semantics, while no experiment independently re-executes the model or validates the critical capture pipeline. This makes the core claim closer to internal CI trace consistency than robust validation of deployed serving behavior.
- The fault evidence is mostly constructed and designed around the framework’s predicate vocabulary. The historical case is valuable but only one defect, and a conventional persistent-base invariant also detects it. Thus the incremental empirical case for ForkAudit over a carefully engineered invariant suite remains limited.
- The paper is very dense and artifact-oriented. Numerous secondary contextual tables and supporting cohorts dilute the central contribution, while the principal protocol and its essential evidence could be made more accessible.
- Reproducibility is asserted through relative artifact paths and replay claims, but the PDF itself does not provide an accessible artifact or enough implementation detail to independently reproduce the full capture and execution pipeline.

(4) Questions for the authors

- What practically realistic baseline audit suite—e.g., persistent-state immutability checks, allocator instrumentation, and targeted metamorphic tests—would fail to catch the historical bug or the held-out faults that ForkAudit catches? Please quantify this comparison.
- Can the contract be instantiated for native continuous/ragged batching and optimized recurrent kernels without intrusive synchronized capture? What is its runtime/storage overhead in CI?
- How are storage IDs guaranteed to faithfully represent backing storage across allocator reuse, views, IPC import, and framework-version changes, given that absolute addresses are intentionally excluded?
- Can the authors release a minimal runnable artifact or an independently replayable trace that lets reviewers validate the central 96-cell result without the full pinned H20 environment?
- Which portions of the methodology are expected to generalize unchanged to non-GDN hybrid architectures, and which rely specifically on the observed 60-tensor GDN state layout?

(5) Reproducibility and ethics

The paper gives an unusually detailed reproducibility statement, frozen geometries, software versions, schemas, and an artifact map. However, based on the submission alone, the claimed packages are not directly available, and independent re-execution of the model/capture path is explicitly absent. Reproducibility is therefore good in design but only partially substantiated for an external reviewer. The ethics statement is appropriate for a systems study using PG-19 training data and clearly notes that broader safety, privacy, and deployment effects are out of scope.

(6) Overall score—4

The paper is careful, technically thoughtful, and likely useful as an engineering methodology. However, its novelty is mainly integration and protocol discipline, and the central evidence remains a narrowly scoped, producer-trusted case study without an accessible independent reproduction or persuasive baseline comparison. I therefore lean weak reject for ICLR.

(7) Confidence—4

The paper is detailed, explicit about boundaries, and internally coherent, enabling a high-confidence assessment. Remaining uncertainty concerns the unavailable underlying artifacts and whether the integrated methodology offers sufficient conceptual and practical advance beyond existing audit practices.

(8) Verdict—Reject
