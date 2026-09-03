# A32 repository inventory -- post-meta evidence audit

## Located

- The mutable authoring directory contains `main.tex`, `refs.bib`, the ICLR style
  files, and compiled manuscript/auxiliary files. It contains no figure asset, and the
  manuscript includes no figure.
- The immutable round-00 and round-01 submissions, reviews, panel summaries, and
  meta-reviews are present under `review/` and were used read-only.
- `literature/citation_lock.json` contains primary-source support records for the
  signed-summary boundary and the Xia 2025, Zhao 2024, and Kamoda 2023 neighboring
  prompt-variation studies.

## Not located

- Experimental records A32-E1 and A32-E2 or a stable ID-to-file crosswalk.
- The exact natural-strata and 400-item manifests, overlap map, exclusions, literal
  filler string, insertion location, serialized prompts, tokenizer dose fixtures,
  arm construction, or gate implementation.
- Per-item predictions, commit-score traces, final-answer labels, preregistration,
  model/tokenizer revision, decode launcher/configuration, environment lock, analysis
  code, or table reproducer.

## Consequence

Estimand definitions, printed arithmetic, citation-lock scope, manuscript consistency,
build status, and layout can be audited. Empirical measurements, treatment fidelity,
sample construction, score semantics, uncertainty, and decoder replay cannot be
independently recomputed. Unvalidated uncertainty brackets, source-labelled Brier/NLL
measurements, and MATH500 numbers are excluded from the manuscript. No experiment was
run as part of this audit.
