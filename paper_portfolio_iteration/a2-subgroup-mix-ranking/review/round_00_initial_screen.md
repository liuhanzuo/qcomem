# Round 00 — inherited initial screen

This is a provenance record, not a new blind full review. No reviewer subagent was invoked in
Phase 0/1, and no ICLR 2027 form was verified. The internal local ICLR 2026 rubric remains a
proxy only.

## Source and result

Source: `01_p5/research_portfolio_36/review_initial_20260822/INITIAL_ICLR_PORTFOLIO_SCREEN_ZH.md`.
Its portfolio table gives this paper cross-review ratings **4/6/6** and an overall internal
meta score **6**. The source explicitly says these are internal AI screening signals rather
than acceptance probabilities. The rating distribution cannot be converted into official
dimension medians because those values are not supplied.

## Decision-driving finding

The screen judges the continuous-simplex simultaneous paired-regret certificate as the
substantive contribution, while finding the allocation statement majorly defective: Eq. `(mm)`
lacks candidate-indexed uncertainty and its engine restores only a width surrogate rather than
the complete candidate/mix regret UCB. It directs the paper to downgrade or re-prove Prop. 5
and to validate real shift invariance on temporal/geographic mixtures.

## Audit disposition

- Accepted as supported for planning: protect the paired certificate and prioritize scope
  repair of the allocation section.
- Open, not adjudicated: all detailed mathematical proof obligations; this audit did not
  independently re-prove them.
- Added from Phase 1 inventory: quantify abstention/fallback cost and distinguish exact from
  normal/CLT evidence throughout the paper.

No review gate is passed; this record must not be represented as a five-reviewer round.
