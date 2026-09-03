# Formal comparison audit report

Status: **complete, integrity-valid, ready for an independent promotion
audit**.

## Primary pre-registered result (`alpha=0.05`)

All three frozen rules passed the common 12-rule CAL empirical-Bernstein
screen.

| Rule | CAL EB UCB | TEST replay flip | TEST mean k | TEST count reduction |
| --- | ---: | ---: | ---: | ---: |
| BAYES-H | 0.033658 | 0.024888 | 6.128 | 80.85% |
| BAYES-UNIF | 0.025481 | 0.017612 | 7.191 | 77.53% |
| WOR-PPR-CS | 0.004839 | 0.000636 | 12.702 | 60.31% |

The paired BAYES-H minus WOR-PPR-CS count-reduction gap is
`+0.20542 +/- 0.05125`; its pre-registered simultaneous Hoeffding interval
excludes zero.  Thus, on this frozen derived-manifest replay and at the same
nominal alpha, BAYES-H uses substantially fewer expected prefix samples than
this source-faithful uniform-prior running-intersection PPR-CS instantiation.
WOR-PPR-CS also has much smaller replay flip, so this is not a Pareto or
theorem-level dominance claim.

The paired BAYES-H minus BAYES-UNIF count-reduction gap is
`+0.03320 +/- 0.05125`; its interval does not exclude zero. The experiment
therefore does not support a reliable efficiency separation from BAYES-UNIF at
the primary operating point.

The same-nominal-alpha BAYES-H efficiency interval versus WOR-PPR-CS excludes
zero at 0.10, 0.05, and 0.02. BAYES-H did not pass the new 12-rule CAL screen at
0.01, so that row must not be used as a certified BAYES-H operating point.

## Integrity and source-faithfulness checks

- The preflight hashed the manifest without parsing its rows and froze the
  protocol, config, output schema, and runner before FIT/CAL/TEST aggregation.
- FIT froze all per-`K` policy tables. BAYES-H matched the prior frozen runner
  exactly over all 33 counts and four alphas (`max_abs_difference=0` for both
  flip and expected `k`).
- The PPR ratio definition and its uniform-prior hypergeometric threshold were
  checked by exact rational arithmetic. Every terminal set was the observed
  singleton.
- For all 33 fixed counts and four alphas, exact DP verified that WOR-PPR-CS
  decision error is no larger than running-CS miscoverage and that
  miscoverage is no larger than alpha. At alpha 0.05, the worst fixed-count
  decision error was 0.013211 and worst running-CS miscoverage was 0.025892.
- Problem hashes and retained source rows were unique; the seeded
  FIT/CAL/TEST split reconstructed exactly.
- TEST was not used to select a method, alpha, prior, confidence-set variant,
  or output field. The formal TEST artifact was written once and refuses
  overwrite. Final verification did not aggregate TEST again.
- An older, differently split BAYES-UNIF result was inspected only while
  checking whether a runnable implementation already existed. It was not an
  input to this analysis and did not change the required methods, primary
  alpha, working prior, CS variant, or reporting contract.

## Promotion boundary

This result is sufficient to enter a targeted promotion audit for the prior
review issue requiring a direct same-endpoint WoR-CS and BAYES-UNIF
comparison. A manuscript revision may claim only the measured
same-nominal-alpha replay tradeoff for this exact comparator and carrier. It
must retain that WOR-PPR-CS is more conservative in realized flip, that the
BAYES-UNIF efficiency interval overlaps zero, and that no matched-realized-risk,
ordered-online, correctness, or service-cost conclusion was tested.

