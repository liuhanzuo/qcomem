(1) Panel consensus

All three reviewers judge ForkAudit careful, coherent, and unusually candid about its conditional assurance boundary. They agree its core insight—output equality cannot establish mutable-state ownership—is valuable, and that the fail-closed, phase-aware protocol is well specified. The frozen PDF is the shared primary source :codex-file-citation{path="/Users/liuhanzuo/MacLLM-Bench/paper_autonomous_multifork_iteration/review/round_36/pdf_only_input/forkaudit.pdf" purpose="source"}.

(2) Meaningful disagreements

There are no material disagreements on outcome or central reasoning: all three recommend Reject with the same score. Reviewer 1 especially emphasizes missing comparisons with realistic conventional audit suites and manuscript density; reviewer 2 foregrounds portability, fresh capture, and CI overhead; reviewer 3 most strongly requests a less-coupled capture-completeness check, treatment of legitimate views/false positives, and a formal bridge from instrumentation to ownership.

(3) Decisive strengths

- Crisp problem framing and phase-aware ownership decomposition for hybrid KV/GDN state.
- Explicit TCB, observation-point boundary, and separation of evidence coverage from replay verdicts.
- Strong within-stack experimental discipline: factorial cells, frozen gates, selected numerical replays, and a historical alias case where outputs can remain unchanged.
- Honest limitations; the paper does not misrepresent designed faults as a detection-rate estimate.

(4) Decisive blockers

- The main guarantee rests on trusted producer-side enumeration, capture, storage semantics, and semantic binding; the separated observer does not independently validate those dependencies.
- Evidence is too narrow for broad practical significance: one fixed stack, sequential batch-one execution, limited geometry, selected operator boundaries, and no native continuous/ragged serving or optimized recurrent kernel.
- Fault studies are predicate-aligned constructed mutations. One historical defect is useful but insufficient, and is detectable by a conventional persistent-base invariant.
- Incremental value over a strong invariant/metamorphic/storage-tracing baseline is unquantified; external reproducibility of fresh capture is not established from the submission.

(5) Highest-value next evidence

1. A minimally coupled, independently checkable capture-completeness/slot-enumeration mechanism, plus a runnable artifact for fresh capture and the historical defect.
2. A quantitative comparison against a realistic conventional audit suite on held-out or naturally occurring failures, including false-positive/diagnostic burden.
3. Portability on a materially different hybrid runtime and native batched scheduler, with capture/replay runtime and memory overhead.

(6) Editorial issues versus evidence issues

Editorial: streamline artifact/contextual cohorts, foreground the core contract and decisive evidence, and clarify the adapter interface and storage-ID/view behavior.

Evidence: TCB independence, capture completeness, comparative detection value, portability to realistic scheduling/kernels, overhead, and externally reproducible fresh execution. These require new evidence, not presentation changes.

(7) Unscored panel disposition

The panel disposition is Reject: all three reviewers gave score 4 with confidence 4. This summarizes the three submitted scores and is not a fourth score.
