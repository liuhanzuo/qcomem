## Summary and claimed contributions

ForkAudit proposes a receipt-bound audit protocol for shared-prefix hybrid LLM inference, combining ownership/lifecycle receipts, phase-aware storage checks, call-contract records, cross-arm/cross-fan-out equivalence, selected numerical sidecars, and designed faults. The empirical core is one Qwen3.5-35B-A3B/H20 implementation with four KV-by-GDN ownership cells, \(N\in\{1,8,32\}\), and eight books. The paper claims bounded feasibility and replayability under an honest capture producer—not general serving correctness, independent recapture, or production throughput.

## Strengths

- The paper is unusually explicit about its assurance boundary. It repeatedly distinguishes trace replay from independent observation, and does not overstate the two-stream, mutation, memory, or overhead results.
- The contract is concrete: required record classes, phase semantics, pointer-free interval logic, and target-to-record dependencies are all well specified.
- The factorial design directly targets a real hybrid-state concern: sharing immutable KV while correctly rebinding mutable recurrent state.
- The paper goes beyond token equality: it checks canonical logits, logical KV, GDN-state receipts, selected dense-FP32 attention rows, and a candidate-import-free GDN recurrence at a stated boundary.
- The target-gate suppression matrix is a useful diagnostic experiment. It clearly shows that equality of tokens, and sometimes logits, does not expose several constructed violations.
- Memory denominators are carefully separated; the paper avoids equating allocator deltas with service capacity.

## Weaknesses

### Fatal / central

- The principal “audit” guarantee is conditional on an honest capture producer that records every mandatory event faithfully. The paper itself concedes coherent omission/fabrication, transient writes restored between snapshots, compiled-dispatch choices, and unrecorded semantics. Consequently, the primary result is a replayable consistency check over producer-supplied evidence, rather than a convincing audit of the implementation being audited. This is a fundamental limitation for the claimed ownership-assurance contribution.
- There is no independent end-to-end recapture or differential implementation. The source-distinct observer still reads candidate-created tensors in the same process and trusts the same PyTorch storage API and phase labels. The selected numerical oracles begin from captured inputs; hence they cannot validate upstream activation construction or full-model behavior.
- External validity is very limited: one model, one adapter/runtime stack, one H20 family, one BF16 KV/page geometry, one partial tail, a fixed 32-token transition, \(N\leq32\), eight greedy steps, and sequential batch-one calls on one stream. The paper provides no evidence that its approach applies to native continuous/ragged batching—the natural target setting for prefix-sharing serving.
- The experimental campaign is almost entirely designed around the proposed checks. Nine hand-crafted positive controls reaching predeclared gates, including a sentinel that is explicitly not a detector, do not establish robustness against realistic, held-out, or adversarial faults. There is no false-positive/false-negative characterization.

### Fixable but important

- The novelty story is mostly integration. Tables 11–12 acknowledge that the constituent ingredients are established, but the paper lacks a crisp formal result or empirical comparison showing why this integration is substantively more effective than a simpler trace validator plus standard differential/metamorphic tests.
- Much of the manuscript is devoted to “unpooled context” (CoMem, HYPIC, Hydragen/Palu, Marconi, lifecycle cohorts). The authors label it carefully, but it dilutes the central message and makes the paper read as a large audit dossier rather than a focused research contribution.
- Practicality is not compelling: the full audit adds a median 512 ms and 4.321× wall-time for a single frozen 16-token request call, and the reviewer replay package is about 851 MiB. These measurements are useful but underscore that deployment feasibility is unestablished.
- Reproducibility disclosures describe postexecution corrections to a raw-versus-derived GDN schema and a detached-manifest run-ID issue. Disclosure is good, but a paper centered on audit integrity needs a clearer, independently checkable account of which artifacts/results predate each correction and why the corrections cannot affect conclusions.
- “Exact” means equality after specified CPU-FP32 canonicalization or digest comparison, not bitwise equality of original device tensors. This is appropriately disclosed but should be much more prominent in claims and captions.

## Questions for the authors

1. What threat model makes the honest-producer assumption operationally meaningful? How does ForkAudit provide materially stronger assurance than a conventional trusted trace validator when the candidate controls capture completeness?
2. Can the authors provide an independent capture path that does not share candidate-created tensor objects, labels, storage APIs, or runtime-side hooks?
3. Why should the nine selected mutations be considered representative of likely hybrid-serving defects? Please include held-out or naturally induced bugs, plus a negative-control/false-positive study.
4. Can the authors test native ragged/continuous batching, real scheduler interleavings, more than one document, and cancellation before synchronization? These seem central to the motivating serving setting.
5. Is there a formal soundness/completeness statement for the receipt schema relative to a precise runtime model, rather than only a list of trace predicates?
6. Please clarify the full provenance and independence of all postexecution replay/schema amendments, ideally with an immutable timeline and independent re-verification.
7. Why is the costly live-capture mode a plausible deployment option rather than an offline debugging mode? What reduced audit configuration offers useful assurance at acceptable overhead?

## Numerical/evidence/scope consistency audit

- The main count is internally consistent: \(8\) books × \(3\) fan-outs × \(4\) ownership cells = \(96\) configurations. The stated \(288\) adjacent fan-out comparisons is consistent with four cells, eight books, and \(1+8\) repeated request identities across the \(1\to8\) and \(8\to32\) transitions.
- The paper correctly distinguishes eight books from stochastic repetitions; they are distinct workload sources, not independent replicated runs. Thus “all pass” claims provide no uncertainty estimate.
- The selected numerical evidence is genuinely small: eight attention rows and four GDN layer transitions. Since the GDN oracle starts post-normalization and the attention oracle starts from captured inputs, neither validates end-to-end computation.
- The \(N=32\) reported allocator deltas are numerically consistent within their stated estimands: Table 18 gives 4.901 vs. 2.229 GiB for the primary final-allocation contrast. Other Table 8 figures use a different source-retaining/full-lifecycle denominator and should not be conflated.
- The source-distinct observer’s 1,080 descriptor denominator versus 2,160 serialized before/after rows is explained consistently.
- The paper is commendably cautious about its 4.321× cost, interval-overlap observations, and context tables. Nonetheless, the evidence supports only frozen-stack trace consistency, not practical production auditability or implementation correctness.

## Presentation and figures/tables

The PDF is professionally typeset, with clear section hierarchy and strong conceptual diagrams (especially Figures 1–3). Figures make the audit pipeline and assurance levels understandable. However, the paper is extremely dense; several appendix tables and labels are small and difficult to absorb at ordinary reading scale. The repeated limitation statements are admirable but overextended, and the large number of context tables obscures the core evidence. A shorter main narrative centered on the threat model, formal contract, primary experiment, and one decisive independent validation would be substantially stronger.

## Recommendation

**Score: 4 / 10 — Below acceptance threshold**  
**Confidence: 4 / 5**  
**Recommendation: Reject**

The submission is careful, technically detailed, and unusually honest about its limitations. However, its central evidence remains producer-trusted trace consistency on a single highly constrained stack. Without independent capture/differential validation, realistic held-out fault evidence, or evaluation in the continuous-batching serving regime that motivates the work, the contribution does not yet establish a sufficiently strong or general ownership-audit result for acceptance.

Reviewed :codex-file-citation{path="/tmp/forkaudit-final-pdf-only.xs72Hx/ForkAudit.pdf" purpose="source"}
