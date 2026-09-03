# Reviewer 3

## Summary

ForkAudit proposes a fail-closed, phase-indexed trace-validation contract for hybrid LLM prefix forks, combining KV ownership, recurrent GDN-state isolation, tail copy-on-write, lifecycle/dispatch receipts, and cross-arm semantic checks. On one Qwen3.5/H20 stack, it reports complete passing evidence for six of seven targets, with compiled dispatch explicitly remaining partial. Its evaluation includes 96 controlled configurations, selected independent numerical replays, targeted mutations, five designer–executor-separated faults, and one historical alias regression.

## Strengths

- Clear core insight: output/logit equality cannot establish mutable-state ownership.
- Careful threat model and unusually candid claim boundaries; coverage is separated from verdicts.
- Strong engineering rigor: frozen manifests, fail-closed record schema, pointer-free storage intervals, phase-specific checks, and replayable artifacts.
- Fault experiments meaningfully illustrate failures that token equality and even exact logits can miss.
- Good separation of the core audit claim from contextual Store/F1 and serving measurements.

## Weaknesses

- Material evidence limitation: all primary evidence remains conditional on an author-side capture/producer TCB. The process-separated observer does not independently enumerate slots, execute the model, validate compiled dispatch, or rule out transient state corruption.
- Material external-validity limitation: the core result covers one model/backend/hardware family, a highly constrained sequential batch-one schedule, one primary tail geometry, N <= 32, and a Torch GDN implementation rather than an optimized production recurrent kernel.
- The mutation studies are targeted constructed controls, not evidence of recall, false-positive rates, or performance on naturally occurring defect populations. The historical bug is compelling but only one case and is also detected by a persistent-base invariant.
- Novelty is mainly a careful integration and formalization of systems-testing practices rather than a fundamentally new inference method; the paper should better establish why this particular contract is the right minimal abstraction across hybrid architectures.
- Presentation is polished but dense and acronym-heavy. The distinction among “target,” “gate,” “coverage,” and “supporting cohort” requires substantial effort to follow. This is correctable, unlike the evidence limitations above.

## Questions for the authors

1. Can the artifact support a genuinely independent end-to-end re-execution from frozen inputs, rather than replay of producer-captured receipts? If not, what is the practical trust model for CI users?
2. What additional instrumentation would be needed to make compiled-binary selection and autotuning coverage complete?
3. How does audit overhead scale with context length, fan-out, concurrent/ragged batching, and continuous serving?
4. Can the contract be validated on at least one additional hybrid architecture/backend and a naturally occurring regression corpus?
5. Why should the selected ownership predicates be considered sufficient or minimally necessary for other recurrent-state families?

## Ethical concerns

No major direct concern. The work uses no human subjects and appropriately limits its claims. Large-scale GPU experimentation and improved serving efficiency can respectively increase compute use and deployment scale; these implications deserve a brief discussion beyond the current acknowledgement.

## Overall score

**5/10 — Borderline / weak accept.**

## Confidence

**4/5.**

## Final recommendation

**Borderline.** I find the contribution technically careful, useful for hybrid-serving CI, and exceptionally transparent about scope. However, the narrow fixed-stack evidence and reliance on the producer-side capture TCB limit its broader scientific significance and soundness claims for ICLR. I would lean accept if the venue values rigorous systems validation artifacts, but would favor rejection if broad generality or independent validation is expected.

