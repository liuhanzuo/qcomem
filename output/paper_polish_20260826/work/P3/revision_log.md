# P3 Revision Log

## Editorial scope

This is a conservative LaTeX reconstruction and evidence-preserving polish of the supplied PDF. The historical source was unavailable; no later manuscript was treated as a substitute. The rendered PDF is therefore the authority for scientific content.

## Structural changes

- Rebuilt a compilable ICLR manuscript with the same central progression: estimand, three entry constructions, paired same-target test, set-ordered ridge counter-arm, and carrier-limited conclusion.
- Consolidated repeated explanations of the cross-sectional/paired sign distinction, the sealed-contract erratum, and the B2 routing confound.
- Moved provenance and implementation detail to the end matter so that the main argument remains readable.
- Recreated all ten tables and four displayed equations, and retained both supplied figures as crops from the original PDF.

## Language and claim changes

- Shortened the abstract, introduction, contribution framing, limitations, and conclusion.
- Replaced broad or causal phrasing with carrier-specific, design-specific language.
- Made explicit that negative Kendall tau and negative paired slope `g` encode different comparisons.
- Kept the ridge result as a sign-only counter-arm; the revision does not transport ridge magnitude across carriers.
- Described B2 as a reference-conditioned routing stress test rather than decisive path evidence.

## Preserved scientific content

- All sample sizes, schedules, seeds, thresholds, effect estimates, confidence intervals, p-values, detector/agreement results, and table cells were transcribed from the supplied PDF.
- The undecided general identification verdict and the synthetic-carrier boundary were preserved.
- No new experiment, citation, scientific result, or mechanistic interpretation was added.

## Build and QA

- Built twice with `pdflatex` through `latexmk` and shell escape disabled.
- Checked the final log for LaTeX errors, unresolved citations/references, fatal errors, and overfull boxes; none were found.
- Extracted final PDF text and checked all locked key values.
- Rendered and visually inspected all eight pages for clipping, overlap, missing figures, and unreadable tables.

## Reconstruction caveat

The revision is semantically faithful to the rendered source, but exact source-level fidelity, original float placement, and original citation keys cannot be guaranteed without the missing historical TeX project.
