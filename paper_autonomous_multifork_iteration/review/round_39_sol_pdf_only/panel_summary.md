# Round 39 — three independent Sol PDF-only reviews

## Frozen input

- PDF: \`build/r39_layout_post_fullread_v11/main_r39_revised.pdf\`
- PDF SHA-256: \`d8b96dc83a6cf79bd48be23f2b3da5dde0a500cbbbd74f89dcafe35a23a13516\`
- Scope: all three reviewers read only the same 28-page PDF. They did not read
  TeX, repository evidence, experiment logs, prior reviews, author plans, or
  one another's reviews, and they did not browse the web.

## Scores

| Reviewer | Overall | Soundness | Presentation | Contribution | Confidence |
|---|---:|---:|---:|---:|---:|
| A | 6 | 3 | 3 | 3 | 4 |
| B | 6 | 3 | 3 | 3 | 4 |
| C | 6 | 3 | 2 | 3 | 4 |
| Meta-review | 6 | — | — | — | — |

Panel median and minimum are both 6. Median dimension scores are 3/3/3.
These are internal ICLR-form simulations, not acceptance predictions.

## Consensus

All three reviewers judge the paper marginally above the ICLR acceptance
threshold. They agree that the strongest contribution is the integration of a
phase-aware ownership contract, mandatory-evidence coverage, pointer-free
storage witnesses, semantic relations, and fail-closed replay. They also agree
that the claims and memory denominators are unusually well bounded.

All three identify the same two evidence ceilings:

1. slot-ID-to-live-tensor binding and capture honesty remain trusted; and
2. compiled dispatch provenance remains partial because the selected binary
   and autotuning choice are not yet bound per call.

The next shared priority is evidence of incremental practical value beyond
output-only or weak conventional checks. Reviewer A emphasizes a native
scheduler case, Reviewer B a strong baseline detector matrix, and Reviewer C
an outcome-blind fault set. The meta-review recommends combining these when
possible.

Lower-priority recurring comments are CI capture/replay cost, narrative
density, and the limited eight-item Store–F1 validation cohort.

## Machine-readable records

- \`reviews/r39_sol_pdf_review_a.json\` —
  \`e09eb7f38c975eba990b15c6bcb02e97ce2894126a1af06cd83b834ec73c05d2\`
- \`reviews/r39_sol_pdf_review_b.json\` —
  \`49221090602c35b706579a8146a9d443776f9edb1cd52883b4eed82d806fc875\`
- \`reviews/r39_sol_pdf_review_c.json\` —
  \`b87a7832562142c265a6e3ef8fc7e212791f9864bc4f02b260717de8d5a3333c\`
- \`meta_review.json\` —
  \`1885362c8d103d90bc3af508c549becaf7a7f49dae22b3c83b9d3d0a6e9c4cc9\`

All four JSON records pass their frozen JSON Schemas. Reviewer B's
\`R39B-I5.dimensions\` value was changed from the non-schema label
\`contribution\` to the schema-equivalent \`significance\`; no judgment or
score changed.
