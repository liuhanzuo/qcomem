# Memory-highlight editorial revision — 2026-08-26

## Scope

This is a non-destructive editorial revision derived from `main_polished.tex`.
It uses `academic-manuscript-editor` for paper-wide emphasis, claim-boundary,
and consistency editing. No experimental result was changed or invented.

## New candidate

- Source: `main_memory_highlight.tex`
- Context table: `tables/h20_deployment_table_memory_highlight.tex`
- Built artifact: `main_memory_highlight.pdf`
- Frozen review PDF: `review/round_37/pdf_only_input/forkaudit.pdf`
- Candidate/frozen PDF SHA-256: `0a0081d15b51a83b9749631b0b75cb0f62a4d7cc396745d0b979dd70cf27106f`

## Preserved originals

- `main_polished.tex`: `a6f42cff02a9774745ec3326597d524dc6b3704cbaf2560ed70ef5233003a1ea`
- `tables/h20_deployment_table.tex`: `b9bf280c69d07088682e66f7748551af53f17f96797655474b407301d17501c9`

Both hashes match their pre-revision values.

## Editorial changes

- Foregrounded the primary allocator result in the abstract, contribution list,
  teaser caption, Results, Discussion, and Conclusion: at N=32 in the
  materialized-GDN arm, shared-document KV reduces final allocated delta above
  the frozen post-priming baseline from 4.901 to 2.229 GiB (54.5%) versus
  full-copy KV, while all registered observables remain exact.
- Foregrounded the separate deployment Store–F1 result: relative to full-prefix
  Q16 at 140.34 MiB/document and mean F1 39.137, CoMem Q8 retains 15.89 MiB
  (88.68% less) with an unrounded mean-F1 delta of 0.000; per-layer mixed retains
  9.74 MiB (93.06% less) with a mean-F1 delta of -0.022 points.
- Reordered Results so the allocator payoff follows the 96-configuration
  exact-semantics result directly.
- Highlighted only the two Store cells in the deployment table; the F1 cells are
  not bolded as best values.
- Defined the per-layer mixed configuration in the table note.
- Compressed noncentral negative context in the main paper while retaining the
  exhaustive limitations and supporting evidence in the appendix.
- Kept the fixed-runtime framing and made the two memory denominators explicitly
  non-additive.

## Claim-boundary audit

- `Store` means median retained-document tensor payload. It excludes metadata,
  pools, process/NVML deltas, admission capacity, and total process GPU memory.
- The 54.5% allocator result and 88.68%/93.06% Store results use different
  cohorts and denominators and are not combined.
- The same measured mean F1 statement is limited to CoMem Q8 on the frozen
  eight-item cohort. The mixed result is reported as a -0.022-point delta, not
  equivalence or a no-quality-loss claim.
- HYPIC/SGLang timings remain in a separate block; no cross-block speedup or
  overall-superiority claim is made.

## Verification

- LaTeX build succeeds with 27 pages; the main paper, including Conclusion,
  ends on page 9 and references begin on page 10.
- No overfull boxes, undefined citations, undefined references, multiply defined
  labels, or rerun warnings occur. Existing underfull-box warnings are confined
  mainly to narrow appendix tables.
- Labels are unchanged (29); begin/end environments remain balanced (37/37).
- The final PDF was reread from page 1 through page 27 after the last manuscript
  edit.
- All 27 rendered pages were visually inspected. The abstract, teaser mappings,
  architecture evidence row, Figure 3 panel labels, deployment table, conclusion,
  references, and appendices are legible and unclipped.

## Blind PDF-only review

Exactly three fresh `gpt-5.6-terra` reviewers received the same prompt and only
the frozen PDF. Scores are 5/5/5 with confidence 4/4/4. Two recommend weak
reject; one gives a borderline disposition with a conditional lean accept. All
three regard the manuscript as technically careful and the memory denominators
as correctly separated. The recurring score limit remains independent capture,
single-stack/production-scheduler breadth, and natural-fault or strong-baseline
evidence—not a numerical, figure-layout, or paper-wide consistency defect.
