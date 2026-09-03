(1) Summary and claimed contributions

This paper presents ForkAudit, an offline, fail-closed trace-validation protocol for ensuring ownership isolation when hybrid LLM inference forks continuations sharing KV pages and persistent recurrent/GDN state. It records phase-indexed storage, lifecycle, call, semantic, and allocator receipts, then replays predicates for identity, prefix immutability, ownership, tail copy-on-write, dispatch provenance, cross-arm equality, and cross-fan-out consistency. On a fixed Qwen3.5/H20 stack, it reports six complete passing targets and one dispatch target passing only at Python-call scope; targeted mutations, selected captured-boundary numerical oracles, process-separated GDN observation, and limited lifecycle cohorts support the stated bounded assurance claim.

(2) Strengths

- The central distinction—output equality does not establish state ownership—is important and well motivated.
- The paper is unusually candid about its assurance boundary (Sections 3 and 6): honest producer capture, slot enumeration, PyTorch/CUDA semantics, and declared observation points remain in the TCB.
- The protocol is technically concrete: phase-specific obligations, normalized pointer-free byte intervals, coverage semantics, and explicit separation of complete/partial/open coverage make the claim inspectable.
- Evaluation is thoughtfully designed for the stated fixed-stack goal. The ownership factorial, per-gate faults, gate-suppression comparison, and frozen designer–executor-separated faults give more diagnostic evidence than simple output matching.
- The manuscript distinguishes pooled from unpooled cohorts and avoids converting designed-fault results into a detection-rate claim. Presentation is clear despite the amount of material.

(3) Weaknesses

- The contribution appears primarily to be a careful integration and instrumentation of existing systems-testing, metamorphic-testing, storage-witness, and hybrid-cache ideas. Tables 9–10 position it responsibly, but do not fully establish a strong conceptual novelty beyond this particular contract and implementation.
- Empirical support is narrow: one model/hardware family, one KV precision/page size, N ≤ 32, short fixed continuations, sequential batch-one calls, and selected GDN layers. The expanded numerical checks begin after producer-captured post-RoPE/post-normalization inputs, so they cannot validate upstream construction or provide an independent end-to-end oracle.
- The fault evidence establishes sensitivity to deliberately constructed, largely predicate-aligned faults, rather than natural-defect coverage, false-positive behavior, or unseen-fault recall. The paper states this limitation, but it materially limits the practical evidence ceiling.
- The central ownership conclusion depends on trusted producer-side enumeration and paused snapshots. The CUDA-IPC cohort only moves descriptor/relation reconstruction; it does not independently validate capture, KV state, compiled dispatch, or transient writes.
- Several deployment and related-work tables are carefully labeled as unpooled context, but their volume distracts from the core validation contribution and makes the main empirical story feel broader than the actual authorized claims.

(4) Questions for the authors

1. Can you evaluate the protocol on a naturally occurring historical regression or an independently discovered bug, rather than only designed mutations, while retaining the frozen-predicate protocol?
2. What implementation burden and runtime/storage overhead does full trace capture impose as prefix length, fan-out, layers, and concurrent scheduling increase?
3. Can the contract be extended to bind compiled kernel binaries/autotuning selections or otherwise make dispatch provenance complete?
4. Which parts of ForkAudit are model- and GDN-specific versus directly portable to other recurrent/hybrid architectures and serving runtimes?
5. Can an independent process or tool perform producer-slot enumeration/capture, reducing the principal TCB assumption?

(5) Reproducibility and ethics

The PDF provides unusually detailed environment, geometry, record-schema, replay, and artifact-path descriptions, plus a clear reproducibility statement. However, from the PDF alone, the referenced code, ledgers, sidecars, manifests, and one-command replays cannot be inspected or executed; reproducibility of the reported results therefore remains unverified here. The ethics statement appropriately scopes the work to systems behavior and notes that it does not assess downstream safety, bias, privacy, or deployment impacts. No acute ethics issue is apparent.

(6) Overall score—6

The paper offers a useful, carefully scoped validation methodology with concrete and unusually transparent evidence. I lean weakly positive because its conditional claims are largely aligned with the evidence. The limited fixed-stack scope, dependence on the capture TCB, and lack of natural/held-out defect evidence prevent a stronger recommendation.

(7) Confidence—4

I am confident in the assessment based on the complete PDF and appendices; the remaining uncertainty concerns the unavailable implementation/artifacts and the broader novelty landscape.

(8) Verdict—Accept
