# P7 Revision Log

## Editorial scope

Evidence-preserving polish of the exact manuscript source corresponding to the supplied PDF. All reported block measurements, uncertainty records, equations, tables, figures, citations, labels, and provenance strings were locked before editing.

## Structural changes

- Narrowed the title to identify the work as an exploratory two-checkpoint study.
- Shortened the abstract, introduction, and conclusion while retaining both checkpoint outcomes and the full evidence boundary.
- Reduced duplication between the main Results narrative and the appendix uncertainty ledger.
- Added the missing local table inputs required for a self-contained build; their contents are unchanged.

## Language and claim changes

- Reconciled an internal inconsistency in the depth discussion. The revised manuscript states that activation-tail turnover and block depth are strongly entangled and that independent predictive value is unresolved.
- Described the 4B result as an exploratory positive association and the 1.7B estimate as weak/uninformative, not as a proven cross-model difference.
- Replaced causal or promotional readings of activation-outlier work with observational, compatibility-based language.
- Removed an uncertain provenance qualifier attached to the unchanged 0.401 value; the remaining label ambiguity is recorded for author verification.

## Preserved scientific content

- The two checkpoints, probed-block counts, HQQ configuration, turnover definition, primary correlations, depth correlation, partial estimates, intervals, robustness records, AUROC values, hashes, and two-of-eight-cell boundary are unchanged.
- No new experiment, citation, universal trigger claim, or mechanism claim was added.

## Build and QA

- Built twice with `pdflatex` through `latexmk` and shell escape disabled.
- Citation-key, label, and displayed-equation sets match the untouched source exactly.
- The final log contains no LaTeX errors, unresolved citations/references, fatal errors, or overfull boxes.
- Extracted the final text and rendered all twelve pages; no clipping, overlap, or missing figure/table was observed.
