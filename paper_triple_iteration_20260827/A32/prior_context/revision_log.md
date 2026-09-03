# A32 revision log

## Scope

Evidence-preserving manuscript polish. No experiments, independent citation verification, or reviewer scoring were performed.

## Source changes

- Replaced the status-heavy title with a descriptive title centered on the observational and within-item evidence.
- Rebuilt the abstract around the paper's inferential hierarchy: observational association, randomized length intervention, cross-bed transport limit, and judge-audit contribution.
- Tightened the introduction and contribution list so that observational, causal, and transport claims are not conflated.
- Clarified the second-bed `POWER_LIMITED` result and the role of the outcome-defined accuracy-regime analysis.
- Rewrote the conclusion to preserve the reported directions and statuses while explicitly rejecting a length-to-gap causal inference from the observational contrast.
- Moved the bibliography before the appendix. Main text now ends on PDF page 8; references and the start of Appendix A occupy page 9. No evidence was deleted to meet the page boundary.

## Semantic safeguards

- Preserved all reported numbers, intervals, sample sizes, equations, table/figure contents, citation keys, named artifacts, and registered-versus-post-hoc distinctions.
- Added no experiment, literature claim, or mechanism claim.
- Narrowed only unsupported causal or transport wording; no negative or inconclusive status was upgraded.

## Verification

- Built from the isolated source with shell escape disabled, using BibTeX and two final LaTeX passes.
- Checked the extracted PDF text, cross-reference/citation warnings, page boundary, and rendered pages.
- Remaining source-dependent items are listed in `needs_verification.md`.
