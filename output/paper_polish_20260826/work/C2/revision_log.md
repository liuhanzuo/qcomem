# C2 revision log

## Scope

Evidence-preserving manuscript polish. No experiments, independent citation verification, or reviewer scoring were performed.

## Source changes

- Reorganized the abstract around the retrieval operating point, the measured depth-cost trade-off, and the limits of generalization beyond a frozen training-free backbone.
- Tightened the introduction and closest-work paragraph while retaining the explicit absence of a same-bed prior-system comparison.
- Replaced “pure-depth” and “depth only” language with “depth-varying” where the lower-band context set remains different.
- Propagated that attribution boundary through the claim map, protocol, experiment narrative, table header/caption, and conclusion.
- Distinguished the scored-endpoint relocation null from non-identical raw generations.

## Semantic safeguards

- Preserved all reported numbers, equations, intervals, multiplicity status, sample sizes, table/figure contents, citation keys, and preregistration qualifiers.
- Did not upgrade the E--A comparison to a fully deconfounded causal effect.
- Added no claim of constant system memory, cache-aware-training performance, or superiority over unimplemented prior systems.

## Verification

- Built from the isolated source with shell escape disabled, using BibTeX and two final LaTeX passes.
- Checked extracted text, current citation/reference warnings, main-text boundary, and rendered pages.
- Remaining source-dependent items are listed in `needs_verification.md`.
