# C3 revision log

## Scope

Evidence-preserving manuscript polish. No experiments, independent citation verification, or reviewer scoring were performed.

## Source changes

- Replaced the promotional title framing with a descriptive title tied to the paper's evidence object.
- Rewrote the abstract and introduction to foreground the executable evidence bundle, its measured checks, and its explicit limits.
- Softened unnecessarily adversarial related-work phrasing while retaining the paper's technical distinction.
- Compressed the conclusion and kept the D1 and 16k attribution boundaries visible.
- Moved the reproducibility and ethics statement, unchanged in substance, to Appendix A so the conclusion ends on PDF page 9; references begin on page 10.

## Semantic safeguards

- Preserved every reported count, digest, table/figure value, equation, citation key, case label, and experiment status.
- Did not imply that scanner/red-team evidence was rerun, that SparseForge transfer was independently reproduced, or that the 16k causal attribution was upgraded beyond inspection grade.
- Added no new system, experiment, or literature claim.

## Verification

- Built from the isolated source with shell escape disabled, using BibTeX and two final LaTeX passes.
- Checked extracted text, current citation/reference warnings, main-text boundary, and rendered pages.
- Remaining source-dependent items are listed in `needs_verification.md`.
