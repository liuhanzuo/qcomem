# P5 Revision Log

## Editorial scope

Evidence-preserving polish of the exact manuscript source corresponding to the supplied PDF. The intervention, evaluation protocol, frozen gate, numerical results, figures, tables, equations, citations, labels, and artifact identifiers were locked before editing.

## Structural changes

- Tightened the abstract and introduction around the single causal sequence: endpoint shock, alarm, frozen settlement, and checkpoint rollback.
- Removed forced page breaks that fragmented the narrative.
- Moved the separate Route-3 diagnostic from the main argument to the appendix.
- Reduced repeated treatments of the same result across Results, Discussion, Limitations, and Conclusion.

## Language and claim changes

- Defined the E1 synthetic endpoint-shock setting in plain language before relying on the internal label.
- Kept the core positive result narrow: the gate detects the registered shock and improves over deployment of the last checkpoint.
- Repeatedly but compactly preserved the negative boundary: the gate is outcome-identical to best-validation selection and early stopping on all 64 shocked hold-out records.
- Distinguished the observed 0/64 healthy alarms in a structurally monotone synthetic carrier from realistic deployment false-alarm calibration.

## Preserved scientific content

- The 12-epoch schedule, 150/600 intervention size, epochs 9--11, deterministic label map, threshold 0.1354038956, FIT/hold-out split, 64/64 and 0/64 counts, confidence bounds, paired effects, and epoch-8 selector identity are unchanged.
- No new experiment, citation, result, or deployment claim was added.

## Build and QA

- Built twice with `pdflatex` through `latexmk` and shell escape disabled.
- Citation-key, label, and displayed-equation sets match the untouched source exactly.
- The final log contains no LaTeX errors, unresolved citations/references, fatal errors, or overfull boxes.
- Extracted the final text and rendered all ten pages; no clipping, overlap, or missing figure/table was observed.
