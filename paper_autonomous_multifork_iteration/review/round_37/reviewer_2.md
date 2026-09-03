# Reviewer 2

## Summary

ForkAudit is an offline, fail-closed trace-validation protocol for hybrid LLM inference that shares immutable KV-prefix state while managing mutable GDN recurrent state. It records lifecycle, storage-range, COW, transition, call, and semantic receipts, then replays ownership predicates separately from evidence coverage. On a single Qwen3.5-35B-A3B/H20 stack, it evaluates a 2×2 KV/GDN ownership design across 96 configurations, selected numerical sidecars, targeted mutations, a small process-separated observer cohort, and one historical alias regression. The authors carefully delimit claims to captured observation points and their stated TCB.

## Strengths

- The problem is real and under-addressed: output/logit agreement does not establish safe ownership of mutable hybrid state after branching.
- The paper's core conceptual distinction—trace coverage versus replay verdict—is clear and useful for CI/debugging systems.
- The trace contract is unusually explicit: phases, mandatory records, normalized pointer-free storage intervals, aliases/disjointness, tail COW, and dispatch receipts are operationalized rather than left informal.
- The evaluation is transparent about scope. The authors explicitly acknowledge the producer/capture TCB, unbound compiled dispatch, selected-boundary numerical checks, and lack of population-level fault-detection estimates.
- Positive controls are thoughtfully structured, including gate-suppression outcomes and five designer–executor-separated faults. The historical case compellingly illustrates an alias that preserves reported tokens, logits, and terminal state.
- Presentation is polished: diagrams, tables, and appendices are readable and the memory denominators are appropriately separated.

## Weaknesses

- The contribution is primarily a careful integration of established testing, provenance, storage-witness, and metamorphic ideas. The novelty over an engineering-quality audit harness is limited; the paper does not clearly isolate a new general principle or demonstrate adoption beyond its custom stack.
- The central assurance claim is materially limited by the TCB: producer-side slot enumeration and semantic labeling are trusted, while replay validates only the captured trace. The process-separated observer still relies on producer selection and CUDA-IPC/PyTorch semantics. Thus this is not strong independent validation of the runtime's ownership behavior.
- Evidence is extremely narrow: one model, one hardware family, one page size, one partial-tail geometry, batch-one sequential scheduling, and fan-out at most 32. The added scheduler and two-stream cohorts do not establish native continuous/ragged batching, kernel overlap, or in-flight cancellation.
- The numerical "independent" checks begin from candidate-produced post-RoPE or post-normalization inputs. They provide useful local operator checks, but cannot validate upstream state construction or end-to-end correctness; only 12/30 GDN layers are included in the expanded sweep.
- The mutation evidence is sensitivity to designed faults, not a meaningful estimate of recall, false positives, or robustness to naturally occurring defects. The authors state this correctly, but it substantially constrains the significance of the empirical result.
- Dispatch provenance is explicitly partial because the selected compiled binary and autotuning choice are not bound. This is a material evidence limitation for a systems paper, not merely a presentation issue.
- The Store/F1 and related-system panels are peripheral, use tiny eight-item cohorts and different harnesses, and add substantial length without materially validating ForkAudit. They would be clearer as a compact appendix or omitted.

## Questions for the authors

1. Can the protocol be integrated into an unmodified or lightly modified production hybrid serving stack, and what are the runtime/storage overheads of capture on realistic continuous-batching workloads?
2. What prevents a benign but incorrect producer from omitting or misclassifying a tensor slot while still emitting a syntactically complete trace? Can an independent enumerator or lower-level allocator witness reduce this TCB?
3. Why is the transition fixed at 32 tokens, and how robust are results to other transition points, page alignments, longer generations, multiple documents, and concurrent cancellation?
4. Can you demonstrate the protocol on a second hybrid model/runtime and on at least one native continuous/ragged batching execution?
5. Which parts of the artifact are sufficient for an outside reviewer to reproduce all headline results without access to the original GPU environment or unredacted tensors?

## Ethical concerns

No major concerns. The work uses PG-19 training data and does not involve human subjects. As the authors note, lower inference cost could facilitate increased deployment; this is a standard indirect systems concern rather than a paper-specific ethical issue.

## Overall score

**5/10 — Marginally below the acceptance threshold.**

The paper is careful, technically competent, and unusually honest about its scope. However, the principal evidence remains a narrow, self-instrumented case study whose strongest conclusions depend on a broad capture TCB; novelty and external significance are not yet demonstrated at the level expected for ICLR.

## Confidence

**4/5 — High confidence.**

## Final recommendation

**Weak Reject.** I would encourage resubmission after showing cross-stack adoption, realistic scheduler/concurrency coverage, and stronger independence of observation/enumeration from the instrumented producer.

