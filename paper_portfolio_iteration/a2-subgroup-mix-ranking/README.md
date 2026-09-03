# A2: finite-sample subgroup-mixture ranking

The active manuscript is Revision 04 of *Finite-Sample Safe Model Ranking
under Subgroup-Mix Turnover*.

## Current checkpoint

- Paper: [`manuscript/paper.pdf`](manuscript/paper.pdf)
- TeX SHA-256: `269e95c75e5c03b02a277617435f84126a0e7674f811fa2a04c3e114776ad0c8`
- PDF SHA-256: `e7a47b1f0f02e40fb55dbc79fb98a2b1cb0f3fa2eb3894a2fe1c06047b804fc9`
- 14 pages; frozen Round-4 root
  `4fb48a187264fe400fcfda23f8024741f45ea269c740530b4964705c86d08e2c`.
- Revision-04 targeted verification: `partially_resolved` because the formal
  scope repair passed but the E03 application contract remains unavailable.

Revision 04 is the best verified artifact by integrity/build evidence, not a
score increase. The latest full panel is Round 02:
`[4,4,4,4,4]`, median/meta 4, evidence ceiling 4. Round 03 contains only one
technical score of 4; Round 04 is unscored targeted verification.

## Supported claim boundary

The paper studies safe ranking/abstention under subgroup-mixture turnover,
using simultaneous weighted risk bands and a same-CAL-point paired-difference
MPB regret gate. Strict finite-sample claims belong to the exact
Clopper-Pearson, paired-Hoeffding, and Maurer-Pontil variants. The normal/CLT
frontier and M6 are asymptotic/descriptive; M9/M10 are fixed-statistic width
counterfactuals, not actual safe-allocation theorems.

See [`review/revision_04_independent_verification.json`](review/revision_04_independent_verification.json),
[`review/best_checkpoint.json`](review/best_checkpoint.json), and
[`state/score_trajectory.json`](state/score_trajectory.json).

## Evidence required before another review cycle

1. Reviewer-safe E03 split/collection/candidate-isolation/paired-data/UCB/env
   contract and clean recomputation where necessary.
2. Natural temporal or geographic mixture shift plus within-group invariance audit.
3. Precommitted abstention/fallback cost or utility endpoint.
4. Observable M3 sampling frame and per-seed allocation uncertainty if M3 is retained.

Further prose-only iteration should remain stopped until one of these evidence
gaps changes.
