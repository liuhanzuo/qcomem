# A14 Structure Audit

## Paper identity

This paper audits how multiple-choice scoring protocols change LLaDA-8B measurements on a bounded GSM8K-MC bed, including a draw-count-matched non-injected reference and structured reveal-order comparisons.

## Claim thread and closure

- Problem → option visibility, reveal order, and token feedback are part of the scoring instrument.
- Evidence 1 → confidence scoring exceeds the non-injected reference across the tested budget curve, including the exact 2L comparison.
- Evidence 2 → a fixed schedule that reads no model output can beat confidence ordering on a disjoint fresh pool, with construction-dependent reversals.
- Boundary → protocols are non-interchangeable on this bed; compute-only, confidence-necessary, uniquely-optimal, causal, dose-response, and transfer claims are not supported.

## High-impact findings

- Major: the introduction's three claims occupy one very dense paragraph with nested endpoint qualifications. Split them into explicit contributions while preserving every pool and multiplicity boundary.
- Minor: the abstract is appropriately bounded but can foreground the matched-budget and fixed-order results more directly.
- Major: the available exact main source calls an appendix file that was not preserved with the exact snapshot. The nearest conservative pre-snapshot appendix is used; this must remain an explicit reconstruction limitation.
- Minor: the conclusion is too compressed to restate the protocol-level contribution clearly.

## Priority edits

1. Convert the long contribution paragraph into three readable, scoped items.
2. Expand the conclusion only enough to close the same three claims.
3. Avoid any causal or cross-bed strengthening.
4. Preserve every exact budget, count, p-value, multiplicity status, and endpoint-registration label.
