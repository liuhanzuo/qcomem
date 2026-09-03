# A1 Structure Audit

## Paper identity

This paper audits fallback-aware answer parsing on two frozen free-form GSM8K beds, decomposes canonical accuracy into format coverage and conditional correctness, and shows which level, order, and pattern claims remain identifiable under explicitly scoped rules.

## Claim thread and closure

- Problem → parser fallback changes the measurement instrument.
- Gap → raw canonical accuracy hides format coverage and fallback prevalence.
- Method → report `P(fmt)`, `P(correct | fmt)`, and their strict joint product; align support for paired claims.
- Evidence → frozen Bed 1/Bed 2 counts, the 5.8–9.5 pp format-channel contrast, 35–67× rescue-ratio contrast, raw 3/28 and favorable gated 0/23 order summaries, and common-support J3 analysis.
- Bounded implication → fallback removal is necessary but does not create cross-bed capability comparability.

## High-impact findings

- Major: the abstract carries too many denominator and MDA qualifications before stating the paper's main result. Repair by retaining every inferential boundary while restoring a problem → method → evidence → implication sequence.
- Critical for blind readability: a main-text subsection reports an earlier external review score and argues against that review. This is revision history rather than scientific evidence and contaminates a fresh blind reading. Remove it; retain the actual limitations and parser-lineage boundary elsewhere.
- Major: repeated disclaimers obscure the asymmetric result (level inadmissible, order rule-dependent, pattern not identified). Consolidate them in the abstract and conclusion.
- Minor: several headings state interpretation before evidence. Prefer observational wording where possible.

## Priority edits

1. Rewrite the abstract around the three audited questions.
2. Remove the old-review positioning subsection.
3. Tighten conclusion repetition while keeping all scope qualifiers.
4. Preserve appendix provenance, denominator distinctions, and preregistration deviations unchanged.
