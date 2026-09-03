# A32 reviewer-safe repository inventory

## Located

- The submitted manuscript PDF and a hash manifest for its TeX source,
  bibliography, and compiled PDF are available.
- Reviewer-safe claim--evidence, experiment-registry, and method-provenance
  records are available. The manuscript includes no pre-rendered figure.
- `literature/citation_lock.json` contains primary-source support records for the
  signed-summary boundary and the Xia 2025, Zhao 2024, and Kamoda 2023 neighboring
  prompt-variation studies.

## Not located

- Experimental records A32-E1--A32-E4 or a stable ID-to-file crosswalk.
- The exact natural-strata and 400-item manifests, overlap map, exclusions, literal
  filler string, insertion location, serialized prompts, tokenizer dose fixtures,
  arm construction, or gate implementation.
- Per-item predictions, commit-score traces, final-answer labels, judge decisions,
  oracle fixtures, human-anchor records, preregistration, model/tokenizer revision,
  decode launcher/configuration, environment lock, analysis code, or table reproducer.

## Consequence

Arithmetic, citation-lock scope, manuscript consistency, build status, and layout can
be audited. Experimental measurements, treatment fidelity, score semantics,
intervals, decoder replay, and judge scenarios cannot be independently recomputed.
No executable experimental bundle was available for recomputation.
