# Primary retained-memory editorial revision — 2026-08-26

## Scope

This is a non-destructive editorial revision derived from
`main_memory_highlight.tex`.  It promotes the bounded deployment Store–F1
result to a primary narrative position without changing experimental evidence,
the paper's fixed-stack framing, or the ForkAudit contribution boundary.

## New candidate

- Source: `main_memory_primary.tex`
- Deployment table: `tables/h20_deployment_table_memory_primary.tex`
- Built artifact: `main_memory_primary.pdf`
- Source SHA-256: `d985be11ba15f27f79d78458d3a8430af04ab2359a958a20d2a21dc8043be6a1`
- Table SHA-256: `3dca9e0fd8c673d0b3122706f744174faf2a3d536a2c0a119a3085201e2d505f`
- PDF SHA-256: `064a6fd55eda24a58c082d4ed8893a187df22a1adc9983d7472edf77b03facf3`

## Preserved source candidate

- `main_memory_highlight.tex`:
  `223efbd66913978eefa03f90cfa91ed716f1f7080ec459bce307ab688f30e94e`
- `tables/h20_deployment_table_memory_highlight.tex`:
  `f9f690a238e42b451de12abe6f8f68e0d7b5ad411effd681a17976310edc9a8f`

The prior candidate was not overwritten.

## Editorial changes

- Reordered the abstract so the 54.5% allocator result is followed immediately
  by the separate 88.68%/93.06% retained-Store result.
- Replaced the Introduction's secondary “context” paragraph with a standalone
  bold contribution bullet.  It reports that per-layer mixed retains 6.94% of
  full-prefix Q16 Store (9.74 vs. 140.34 MiB/document; 93.06% less) with a
  -0.022-point mean-F1 change, while Q8 retains 11.32% (15.89 MiB; 88.68% less)
  with a 0.000 mean-F1 delta.
- Renamed the Results subsection to “Deployment retained-memory–quality
  measurements” and replaced the ambiguous “baseline is larger” wording with
  the explicit direction: full-prefix Q16 retains 8.83× and 14.41× as much
  Store as Q8 and per-layer mixed.
- Promoted Table 3 from deployment “context” to same-checkpoint retained-memory–
  quality measurements and marked its two bold Store cells as headline Store–F1
  points.
- Updated the related-work and cohort-authorization wording to use the same
  retained-Store estimand.  No citation, label, numerical row, or experiment was
  added or removed.
- Corrected the HYPIC comparison direction: Prefix Cache and HYPIC retain 8.78×
  and 20.39× as much Store as CoMem Q8, respectively.

## Claim boundary

- Store is median retained-document tensor payload, not total GPU/process
  memory, NVML memory, admission capacity, or peak allocator memory.
- The deployment result is directly comparable only within the same
  checkpoint/runtime and the frozen eight-item Qasper/2WikiMQA slice, with one
  H20-3e per item, a 4,096-token cap, greedy decoding, at most 32 generated
  tokens, and three timing repeats per item (24 measurements per configuration).
- Q8's measured mean-F1 delta is 0.000; per-layer mixed's is -0.022 points.  The
  text does not claim identical predictions, no quality loss, or broad quality.
- Full-prefix Q16 remains faster in the Transformers cohort.  No overall speedup
  or cross-block timing comparison is claimed.
- The 54.5% post-priming allocator result and the 88.68%/93.06% Store results
  retain separate cohorts and denominators and are not additive.

## Verification

- LaTeX build succeeds with 27 pages.  The main paper, including Conclusion,
  ends on page 9; references begin on page 10.
- The log contains no overfull boxes, undefined citations, undefined references,
  multiply defined labels, or rerun warnings.
- All 27 pages were rendered and visually inspected after the final manuscript
  edit; there is no clipping, overlap, or displaced figure/table content.
- The final compiled PDF was reread in full from page 1 through page 27 after
  the final manuscript edit.  Abstract, Introduction, Results, Table 3,
  Discussion, Conclusion, and appendix claim boundaries are mutually
  consistent.

## Fresh semantic-drift audit

A fresh read-only verifier compared both source candidates and both deployment
tables after the final edit.  It found no blocking issue: all percentages,
ratios, F1 deltas, comparison directions, fixed-cohort qualifiers, citations,
labels, section counts, and speed/memory caveats remain consistent.  The numeric
rows of the old and new deployment tables are byte-identical; only the caption
and explanatory note change their narrative hierarchy.
