# Assumptions and boundaries

## Frozen source and audit boundary

This record is based on the local baseline, manuscript, remote snapshot, and the portfolio
screen dated 2026-08-22. Revision 01 edits only `manuscript/` and audit-state files; it compiles
the self-contained local package but executes no experiment or remote operation. Statements
attributed to snapshot logs/verifiers remain provenance claims until independently replayed.

## Working venue assumption

The target is **ICLR 2027**. Official AuthorGuidelines and ReviewerGuidelines plus official
style parity are locked in `state/venue_compliance.md`. The local
`autonomous-paper-agent` ICLR 2026 rubric remains an internal discrete scoring proxy only; it
is not a claim about ICLR 2027 review outcomes or acceptance.

## Scientific assumptions required by the certificate

1. Conditional on FIT, candidates, loss, subgroup definition, confidence allocation, and any
   CAL-count rule are fixed without CAL-error access; FIT and CAL are independent. Candidates are
   not selected or refit using CAL.
2. For every group, CAL examples are conditionally i.i.d. from the unchanged group distribution.
   Individual CAL errors are then Bernoulli, and same-point transformed paired differences
   `X=(d+1)/2` are conditionally i.i.d. and bounded in `[0,1]`. Cross-model dependence on a shared
   point is allowed. Sampling without replacement, adaptive CAL choices using CAL errors, or
   CAL-trained/refit candidates require a separately valid finite-population or sequential bound.
3. CP absolute bands use exactly `kG` cells at per-cell failure `delta/(kG)`. Paired Hoeffding and
   MPB UCBs use exactly `k(k-1)G` ordered-pair/group cells at failure
   `delta/[k(k-1)G]`; the MPB variance formula also needs `n_g >= 2`. The one simultaneous event
   then covers all simplex mixtures and CAL-data-dependent selections; it does not require a
   mixture-grid union bound.
4. Deployment changes only subgroup/label mixture weights; within-group conditional risks stay
   fixed. Covariate shift inside a subgroup is outside the certificate's stated scope.
5. The operator supplies tolerance `tau`; an abstention is not automatically beneficial unless
   the fallback and its cost are measured.

## Audit findings that constrain all future claims

- The simultaneous paired-regret certificate (P3/Thm. 1) remains a separate, potentially
  valid central claim; do not discard or relabel it as the allocation theorem.
- `Eq.(mm)` substitutes `beta_g / sqrt(n_g)` width costs for the actual
  `max_j sum_g w_g UCB_{i*j,g}` candidate/mix regret UCB.  The resulting allocation and
  uniformity proposition therefore support a **width-surrogate** statement only unless a new
  proof closes the equivalence gap.
- Exact/non-asymptotic and normal/CLT/asymptotic results must stay visibly separated in every
  title, table, caption, and conclusion.
- Existing public-carrier experiments use classes as subgroups and constructed mixture grids.
  They do not establish natural temporal/geographic mixture turnover or test the invariance
  assumption in those environments.
- E03's aggregate snapshot rows do not prove the FIT/CAL split, conditional-i.i.d. sampling law,
  paired sufficient statistics, or executed CP/Hoeffding/MPB implementation. They are historical
  diagnostics, not evidence that this certificate contract held in execution.
- M3/M3.5 use historical arrays stratified by ground-truth class. Class is unavailable before its
  costly label unless an independent external stratified frame exists; none is supplied in E04/E08.
  Their count rules and width surrogate are oracle-stratified constructed-mixture diagnostics only.
- The reported abstained-row hard-pick/anchor regrets are useful diagnostics, but there is no
  registered operator cost model, latency, labeling cost, or realized fallback utility.

Unknown or missing provenance is marked as such in the maps rather than inferred.
