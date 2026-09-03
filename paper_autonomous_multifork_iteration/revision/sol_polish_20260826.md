# Sol review and manuscript-editor pass — 2026-08-26

## Review result

- Frozen input: `review/round_38/pdf_only_input/forkaudit.pdf`
- Frozen SHA-256: `064a6fd55eda24a58c082d4ed8893a187df22a1adc9983d7472edf77b03facf3`
- Three independent Terra reviewers: `4 / 4 / 4`, confidence `4 / 4 / 4`
- Independent Sol reviewer: `4`, confidence `4`
- Dimension consensus: Soundness `3/4`, Presentation `3/4`, Contribution `2/4`
- Evidence ceiling: `6`

All reviewers found the bounded numerical and memory claims internally correct.
The decision-driving limits are material evidence gaps: trusted producer
enumeration, one fixed model/runtime stack, predicate-aligned constructed
faults, partial compiled dispatch, and no independent end-to-end oracle.

## Editorial intervention

The `academic-manuscript-editor` pass produced a non-overwriting candidate:

- `main_sol_polished.tex`
- `tables/h20_deployment_table_sol_polished.tex`
- `output/pdf/ForkAudit_sol_polished_20260826.pdf`

Representative changes:

- announced the primary ForkAudit track and separate Store--F1 track before the
  contribution list;
- defined target, receipt, coverage, verdict, gate, registered observation, and
  canonical equality once;
- made the six-complete/one-partial target status consistent across Abstract,
  Results, Table 6, Discussion, and Conclusion;
- stabilized the 44-control, nine-mutation, five-separated-mutation, and one
  historical-defect taxonomy;
- retained the Store headline: `88.68%`/`93.06%` reductions with
  `0.000`/`-0.022` mean-F1 deltas, while keeping the denominator and speed
  boundary explicit;
- compressed repeated caveats while retaining the capture TCB, fixed-stack,
  compiled-dispatch, constructed-fault, full-capture-cost, and cohort limits.

No experimental row, citation key, equation, or comparator changed.  The only
omitted numeric text consists of redundant `8.83x` and `14.41x` ratios directly
recoverable from the retained table values and percentage reductions.

## Verification

- Safe no-shell-escape build passes.
- 27 pages total; Conclusion ends on page 9 and References starts on page 10.
- Citation-key diff: none.
- LaTeX environments: `37 / 37`.
- Critical LaTeX warnings: none.
- All 27 pages were rendered and inspected after the substantive edit.
- After restoring the explicit full-capture-cost qualifier, all 27 pages were
  rerendered: page 9 was re-inspected and the other 26 rendered pages were
  pixel-identical to the already inspected final pass.
- Independent read-only semantic and visual verifiers found no overclaim,
  number drift, clipping, overlap, or malformed token.  The only cosmetic note
  is appendix page 19's large lower-page whitespace before Table 13.
- Final SHA-256: `f55c0c2dca7201904ff82897af75e6f7fc6a31cbf52a1ee76624280d4cdcb72c`.

The final polished PDF has not been blind re-scored; the reported Sol/Terra
ratings apply to the frozen pre-polish Round 38 PDF.
