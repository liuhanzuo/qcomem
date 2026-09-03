# S1 Revision Log

## Editorial scope

Evidence-preserving polish of the exact frozen manuscript source corresponding to the supplied PDF. Numerical endpoints, equations, tables, figures, citations, labels, code identifiers, hashes, and archived evidence status were locked before editing.

## Structural changes

- Reframed the title and opening around the directly observed 0.118 pp post-recovery spread.
- Compressed the abstract, introduction, contribution list, and conclusion.
- Split the limitations into evidence, mechanism, export/provenance, and external-validity boundaries.
- Kept detailed archived analyses in the appendix rather than expanding the main claim.

## Language and claim changes

- Removed “registered negative result” wording because the manuscript lacks paired predictions, a joint uncertainty estimate, and an equivalence margin.
- Replaced an unsupported literature-completeness claim with a narrower statement about what the cited and archived materials establish.
- Kept the 0.52 pp comparison explicitly marginal and per-checkpoint; it is not a pairwise resolution threshold.
- Presented curvature scoring, annealing, and SLoRB as pipeline background, not isolated causal contributions.
- Clarified that no folded/reprojected exact-2:4 deployment export was produced.

## Preserved scientific content

- The ALPS, ELSA-4096, and ProxSparse supports; 624,951,296-token recovery; 2.675 pp and 0.118 pp ranges; 0.52 pp reference; archived 58.47 Avg-9 endpoint; +0.528 pp margin; p=0.47; and all appendix records are unchanged.
- No new experiment, equivalence conclusion, mechanism claim, citation, or deployment result was added.

## Build and QA

- Built twice with `pdflatex` through `latexmk` and shell escape disabled.
- Citation-key, label, and displayed-equation sets match the untouched frozen source exactly.
- The final log contains no LaTeX errors, unresolved citations/references, fatal errors, or overfull boxes.
- Extracted the final text and rendered all sixteen pages; no clipping, overlap, or missing figure/table was observed.
