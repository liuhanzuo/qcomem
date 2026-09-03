# Round 00 — imported initial screen

This is a Phase 0/1 audit record, **not** a fresh independent five-reviewer panel and not an ICLR acceptance prediction.

Source: `01_p5/research_portfolio_36/review_initial_20260822/INITIAL_ICLR_PORTFOLIO_SCREEN_ZH.md` (sections 3.1, 3.2, 4, and 8).

The imported screen identifies A11 as the strongest portfolio entry at that time, reporting cross-review ratings `6/6/8` and meta score `6` under its local ICLR 2026 semantics.  It names the clear count-exchangeable posterior stopper for full-N vote flips as the strongest point and identifies one decision-driving barrier: the public carrier provides pass counts, not real ordered rollouts; its endpoint is full-vote agreement rather than correctness.  The screen's recommended highest-information next step is a new model/task collection with randomized/prefixed order, gold accuracy, completion tokens, cancellation, and latency.

Phase 1 audit agrees with the evidence distinction and adds no new score:

- Existing replay and drift artifacts support a narrow mechanism/boundary story.
- They do not make full-vote agreement a correctness surrogate.
- `1 - mean(k)/N` is not actual billing or latency evidence.
- The local snapshot's check script validates many printed artifact values but cannot validate inaccessible upstream provenance, external citations, or live deployment behavior.

Ledgered actions are `A11-I001` through `A11-I004`.  Do not use this screen to satisfy any five-reviewer, meta-review, or completion gate.
