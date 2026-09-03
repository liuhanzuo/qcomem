# P7 Structure Audit

## Paper identity

This paper evaluates activation-tail turnover as an exploratory full-precision predictor of single-block 4-bit HQQ degradation on two Qwen3 checkpoints and finds a positive association on 4B, a weak/uninformative association on 1.7B, and unresolved separation from block depth.

## Claim thread

`runtime trigger motivation -> quantization-free turnover statistic -> single-block damage outcome -> two-checkpoint association -> depth confounding and uncertainty -> no trigger certificate`

The empirical thread is coherent, but the abstract and conclusion repeat nearly the full uncertainty ledger. Several phrases call the 4B relation “largely a depth proxy” even though the manuscript's own corrected masked partial analysis is explicitly unresolved. That inconsistency is the highest-priority editorial repair.

## Priority repairs

- Critical: replace affirmative “largely/not fully a depth proxy” language with the supported statement that turnover and depth are strongly entangled and independent value is unresolved.
- Major: shorten the abstract to problem, measurement, two carrier outcomes, depth ambiguity, and evidence boundary.
- Major: shorten the introduction and conclusion so robustness diagnostics do not obscure the unit-of-analysis and two-cell limitations.
- Minor: distinguish observed point-estimate heterogeneity from an established cross-carrier difference consistently.
- Minor: remove promotional or causal interpretations of the activation-outlier literature and use “consistent with” language.

## Promise-evidence closure

- Quantization-free predictor -> turnover is computed on the full-precision model: supported.
- 4B positive direction -> rho about 0.559 plus reported robustness analyses: supported as exploratory, not independently established by a single CI family.
- 1.7B relation -> rho about +0.12 with wide interval: supports “weak/uninformative,” not “no relation.”
- Cross-carrier heterogeneity -> observed at two checkpoints; not established as a population effect.
- Independent value beyond depth -> not resolved because turnover and depth are nearly collinear and partial intervals span zero.
- Deployable trigger -> not evaluated; registered AUROC/damage-gain gate not passed/evaluable at achieved resolution.

## Main-text constraint

Keep the bibliography boundary within nine ICLR main-text pages; appendix ledgers remain after the bibliography.
