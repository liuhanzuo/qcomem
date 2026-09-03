# Frozen protocol: same-endpoint WoR-CS and BAYES-UNIF comparison

Status before execution: **pre-registered CPU analysis; TEST unread for this
new comparison**.

## Question and endpoint

On the exact same `N=32` count-exchangeable random-prefix replay endpoint as
A11, compare the FIT-frozen `BAYES-H` rule with two pre-specified comparators:
`BAYES-UNIF` and `WOR-PPR-CS`.  The terminal target is the binary full-count
decision `1{K > 16}`, with `K=16` assigned to the negative side.  The two
outcomes are exact per-task replay disagreement probability and exact expected
prefix count. They are not answer correctness, observed chronology, token
cost, latency, or deployment safety.

The primary operating point is `alpha=0.05`.  The complete pre-specified grid
`{0.10, 0.05, 0.02, 0.01}` is reported without selecting an alpha from TEST.

## Frozen methods

All methods observe a uniformly random without-replacement prefix.  They may
stop first at `k=3`; if they do not stop earlier, they observe all `N=32`
outcomes.

### BAYES-H

Estimate the distribution of final counts from FIT only.  At prefix state
`(k,x)`, compute under that frozen distribution the posterior probability that
the full-count side differs from the current prefix-majority side. Stop when
this fitted score is at most the named alpha and return the prefix-majority
side. This is a plug-in score, not a per-task conditional certificate.

### BAYES-UNIF

Use exactly the same score and decision rule with mass `1/33` on every final
count `K in {0,...,32}`. No data are used to choose this prior. Its marginal
replay loss is eligible for the same finite-family CAL screen as any other
frozen rule, but its posterior score is not asserted to be the true
conditional replay error.

### WOR-PPR-CS

Instantiate Theorem 2.1 of Waudby-Smith and Ramdas (NeurIPS 2020),
*Confidence sequences for sampling without replacement*, for binary
hypergeometric observations.  Use the paper's recommended no-information
choice, the beta-binomial working prior with `a=b=1`, which is uniform over the
unknown final success count. At state `(k,x)`, the current confidence set is

`C_k = {K: pi_0(K) / pi_k(K) < 1/alpha}`.

Equivalently under the uniform working prior,

`C_k = {K: HypergeomPMF(N,K,k,x) > alpha/(k+1)}`.

The strict inequality matches the source theorem. Use the source theorem's
valid running intersection `I_k = intersection_{s<=k} C_s`. Starting at
`k=3`, stop when the nonempty `I_k` is wholly in `K<=16` or wholly in `K>=17`
and return that implied full-count side. An empty intersection does not imply
a side and therefore continues. At `k=N`, return the observed full-count side.
The rule is data-independent apart from its observed prefix, and its
frequentist guarantee is per fixed `K` over the random order, not across-task
Bayesian credibility.

Primary source checked before implementation:
https://proceedings.neurips.cc/paper_files/paper/2020/file/e96c7de8f6390b1e6c71556e4e0a4959-Paper.pdf
(Proposition 2.1, Theorem 2.1, and Section 2.3).

## Frozen data accounting

- Input: the hash-pinned anonymous count manifest named in `config.json`.
- Manifest split: FIT/CAL/TEST = 4000/4000/3607, seed 20260815.
- FIT estimates only the `BAYES-H` count distribution and freezes every rule
  table before CAL aggregation.
- CAL evaluates the 12 fixed method-by-alpha rules. Each rule receives
  empirical-Bernstein failure allocation `0.05/12`; CAL status is reported for
  every rule. No method or alpha is selected from TEST.
- TEST is aggregated exactly once after both FIT and CAL locks exist. Every one
  of the 12 pre-registered rows is reported descriptively, including a row
  that does not pass the CAL screen.

The input is a monolithic JSON file. FIT and CAL stages necessarily parse and
validate its structure, including stored split labels, but they neither
aggregate TEST counts nor use any TEST value to alter a rule, alpha, status,
or output field.

## Exact computation and uncertainty

For each fixed true count `K`, dynamic programming integrates over every
uniform random-prefix path. It returns exact-to-floating-evaluation replay
flip probability and expected stopping count; no Monte Carlo or seed is used.
For `WOR-PPR-CS`, a separate full-path DP computes the probability that the
running CS ever excludes the fixed true `K`. The preflight requires, for every
configured alpha and all 33 values of `K`, decision error no larger than CS
miscoverage and CS miscoverage no larger than alpha (within `1e-12` numerical
tolerance).

CAL uses the same empirical-Bernstein UCB formula as the frozen A11 runner.
TEST mean flip and normalized mean count receive deterministic two-sided
Hoeffding radii with Bonferroni allocation over 12 rows.  The eight paired
`BAYES-H` versus comparator count-reduction and flip gaps receive a separate
Bonferroni allocation over `2 comparators x 4 alphas`; bounded ranges are
declared in `config.json`. Positive `BAYES-H` count-reduction gap means
`BAYES-H` uses fewer expected replay samples.

## Falsification and interpretation

- A source-faithfulness failure occurs if the exact PPR ratio and equivalent
  hypergeometric threshold disagree, the terminal set is not the observed
  singleton, or fixed-`K` CS miscoverage exceeds alpha.
- A comparison is scientifically negative for `BAYES-H` if the frozen TEST
  result shows that a comparator has no worse replay flip and a materially
  smaller expected prefix at the pre-registered primary alpha.
- An efficiency advantage is supported only by the actual paired interval and
  only for this derived-manifest replay population. No strict theorem-level
  dominance, ordered-online claim, or practical cost claim follows.
- Infrastructure or integrity failures are logged separately and are not
  scientific evidence.

