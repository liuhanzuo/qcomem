## Summary

ForkAudit proposes a receipt-bound audit protocol for ownership isolation when forking hybrid LLM state: shared immutable KV prefix pages plus request-private append pages, and mutable GDN recurrent state that must rebind privately at a registered transition. On one Qwen3.5-35B-A3B/H20 stack, it reports exact relational equality across 96 KV×GDN×fan-out configurations, selected FP32 operator checks, and nine designed faults. The authors explicitly limit claims to trusted capture and a fixed sequential schedule.

## Strengths

- Clear motivation: output equality alone does not establish state isolation.
- Strongly scoped and unusually candid limitations; it does not pretend to prove independent recapture, compiled dispatch identity, production scheduling, or capacity.
- The protocol separates ownership, semantic, dispatch, and memory-denominator claims well.
- The factorial design is coherent: 8 books × 3 fan-outs × 4 ownership cells = 96 configurations, with 288 adjacent-fan-out relations.
- Positive controls and the target-gate-suppression matrix are useful diagnostic evidence, especially the cases preserving exact tokens/logits after a target gate is suppressed.
- Presentation is polished overall; Figures 1–2 communicate the state/lifecycle model effectively.

## Weaknesses

**Fatal / central**

- The assurance claim rests on an “honest capture producer.” The proposed replay can validate only events faithfully recorded by the same producer; it cannot detect coherent omission, fabricated receipts, transient mutation restored before capture, or compiled-kernel behavior. The same-process “source-distinct observer” still shares tensors, labels, process, and PyTorch storage API. This makes the central ownership assurance substantially weaker than an independent audit, despite careful disclosure.
- Empirical support is extremely narrow: one model, one hardware family, one page size/precision, one partial-tail geometry, N≤32, batch-one sequential round-major calls, and one registered 32-token transition. This is a compelling case study, not evidence that the audit is broadly usable for hybrid serving systems.
- There is no formal soundness/completeness result for the receipt contract under its stated threat model, nor a comparison against an independently implemented runtime or live observer. Consequently, it is difficult to distinguish a useful instrumentation framework from a bespoke validation harness.

**Fixable / important**

- Much of the paper is occupied by explicitly unpooled contextual cohorts (CoMem, HYPIC, Hydragen, Palu, Marconi). Even with careful caveats, these tables dilute the core contribution and make the paper feel like a broad systems report without broad systems evidence.
- No artifact is available in this PDF-only review, so the many replayability and hash-bound claims cannot be independently assessed here.
- The selected numerical oracles cover only eight attention rows and four post-normalization GDN transitions. They do not validate upstream activation construction or end-to-end behavior.
- The protocol’s practical cost is high: 4.321× median wall-time in a very narrow 16-token request-step measurement. This may be acceptable for audit mode, but needs a clearer deployment story.
- The paper should better explain why “eight argmax steps” correspond to “39 appended tokens”; on first reading, 32 query tokens plus eight generation steps suggests 40 state-appended tokens.

## Questions for authors

1. What concrete threat model makes an honest producer an acceptable trust anchor, and who operates that producer?
2. Can the authors supply an independent live recapture implementation that does not read candidate-created objects or share the candidate’s storage-view API?
3. Which parts of ForkAudit are reusable protocol abstractions versus stack-specific instrumentation for Qwen/vLLM/Transformers-torch?
4. Why is the transition fixed at 32 tokens? How would the contract cover arbitrary transition lengths, ragged decoding, eviction, or native continuous batching?
5. Please reconcile the stated eight decoding steps with the reported 39 appended tokens.
6. What is the false-positive behavior of gates on non-adversarial runtime variation, and how often do realistic faults evade all recorded predicates?
7. Could the authors provide a concise ablation showing the marginal audit value of each receipt family and the cost of lighter-weight audit configurations?

## Numerical / evidence / scope audit

- The principal counts are internally consistent: 8×3×4 = 96 primary configurations; nested N={1,8,32} gives 72 non-vacuous rank-query relations per cell and 288 across four cells.
- The N=32 final-allocation result is consistent with Table 18: 4.901−2.229 = 2.672 GiB for materialized GDN, matching the stated reduction.
- The 2.921 MiB artifact figure agrees with 3,063,111 bytes.
- The two-stream 993,280 scalar-pair count is compatible with four full-vocabulary samples, although only three unique logit digests reduces diversity.
- The manuscript correctly avoids treating books/ranks or deterministic cells as stochastic replications, and correctly separates most supporting cohorts. That restraint is a strength.
- Nonetheless, “receipt-complete” risks sounding stronger than warranted: it is complete only relative to a trusted, potentially common-mode capture path.

## Presentation / readability

No obvious rendering defects, clipped content, or broken tables. Figure 2 is the clearest figure. Figure 1 and particularly Figure 3 use small labels and are hard to parse at normal viewing size. Several appendix tables are extremely dense; Tables 6, 13, and 23 would benefit from simplification or moving operational detail to the artifact. The main narrative is readable but overburdened by qualifications and auxiliary results.

## Recommendation

**Score: 4 / 10 — Borderline reject**

The paper is careful, technically thoughtful, and commendably honest about limitations. However, its central assurance is conditional on the very capture path it seeks to audit, while the evaluation is a single highly customized stack and schedule. I would support acceptance only with a stronger independent-observation story or a more modest framing as a narrowly validated audit harness/case study.

**Confidence: 4 / 5**

Reviewed solely from :codex-file-citation{path="/tmp/forkaudit-final-pdf-only.xs72Hx/ForkAudit.pdf" purpose="source"}.
