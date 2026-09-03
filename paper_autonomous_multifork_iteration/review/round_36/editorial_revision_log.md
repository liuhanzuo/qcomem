# Round 36 editorial revision log

## Editing contract

- Source: `main.tex` at SHA-256 `87998cb3dfc83f9cd164ec6d1fbeac17d3bdbd29bab27878cf1fec79a784f7db`.
- Candidate: `main_polished.tex`; the source manuscript was not overwritten.
- Evidence policy: preserve quantitative values, equations, citations, labels, cohort boundaries, and the fixed-stack scope. Do not add claims unsupported by the frozen evidence chain.
- Required QA: paper-wide reread after the final edit, semantic-drift audit, clean compilation, and page-by-page inspection of the final PDF.

## Material editorial changes

- Reframed the title around ownership-trace validation for hybrid LLMs, rather than treating the method as a cache policy or runtime.
- Compressed the abstract and introduction so the fixed-stack contract, fail-closed coverage rule, falsification evidence, historical defect, and capture TCB form the primary narrative.
- Separated a complete target pass from a Python-call predicate pass: dispatch coverage remains partial because per-call compiled-binary and autotuning-choice evidence is absent.
- Clarified that GDN oracle coordinates 0, 10, 20, and 38 are global model-layer indices.
- Added concise audit-target and selected-numerical-support signposts, reduced repeated caveats, and demoted deployment measurements to explicitly contextual evidence.
- Reorganized discussion into implication and limitations, and compressed the conclusion to avoid repeating results already visible in the abstract and tables.
- Changed Figure 3 from forced placement to a top float, eliminating a large blank page and restoring the nine-page main-text boundary.

## Semantic disclosure

- No quantitative value, equation, citation key, label, or unique numeric literal was added or removed.
- The only claim-status change is conservative: wording that could be read as a passing dispatch target is narrowed to a passing Python-call predicate within a still-partial target.
- “Retrospective reproduction under a preregistered protocol” replaces potentially ambiguous historical-study wording without changing chronology or evidence.
- The historical evidence remains one defect mechanism over three archived-coordinate and five additional frozen-input cells; it is not presented as eight natural defects or as a recall estimate.

## Verification

- Editorial semantic-drift review: resolved.
- Original evidence validator: passed against the unchanged source and frozen evidence chain.
- Candidate static audit: citation-key set, label/reference set, and unique numeric-literal set are identical to the source; LaTeX environments are balanced.
- Candidate compilation: 27-page letter PDF, no undefined references/citations, no LaTeX errors, and no overfull boxes.
- Final visual QA: all 27 pages inspected; main text ends on page 9 and references start on page 10.
- Frozen review PDF SHA-256: `bca29bca5065b3367939498d03776382d6251ab12bbddb92b0ea3f18bb5fafb4`.

## Unresolved evidence ceilings

- Assurance remains conditional on faithful producer enumeration and the capture/replay TCB.
- Empirical validation remains one fixed Qwen3.5-35B-A3B/H20-3e stack with bounded schedules.
- The fault evidence does not estimate natural-defect recall or false-positive/negative rates.
