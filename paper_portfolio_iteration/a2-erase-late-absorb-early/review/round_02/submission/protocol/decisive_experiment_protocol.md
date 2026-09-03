# PLAN-ENABLE-001 — minimal falsifiable deployment experiment

Status: draft planned only. It is not an execution-ready preregistration until
the open data/config fields below are frozen in a versioned manifest. No code
was run, no job was submitted, and no result below is an observed result.

## Decision and claim under test

The practical claim is not that high-loss examples are harmful. It is narrower:
an enable policy using only information observable during training can decide
whether to temporarily quarantine a high-risk cohort during the high-LR window
without systematically damaging a benign-hard/valueful cohort.

This is the minimum decision-changing experiment because the portfolio screen
identified the absence of such a signal as the central barrier. It must not
reuse annotation multiplicity, clean labels, validation/test outcomes, or an
oracle harmful-cohort membership in the enable decision.

## Draft policy skeleton to freeze before any run

For a run with the standard training schedule, collect each example's mean loss
over two fixed warm-up checkpoints. Fit a two-component mixture to those losses
and form posterior `p_high(i)`. Let `Q` be the highest-posterior cohort at the
estimated high-loss mass. Enable temporary quarantine only when all of these
training-visible predicates hold:

1. a two-component mixture beats one component by BIC at both checkpoints;
2. the estimated high-loss mass is in the pre-specified operational range
   `[0.02, 0.20]`; and
3. the two top-cohort sets have Jaccard overlap at least `0.80`.

When enabled, quarantine `Q` only from the first checkpoint through the frozen
LR-decay epoch, then re-admit it. When disabled, continue ordinary training.
The precise mixture family, fitting seed, tie-break, checkpoint epochs,
numerical-failure fallback, output schema, A1 fixed comparator, H dataset, and
H hard-example endpoint remain open and must be frozen in code/config before
this becomes a formal preregistration. A mixture-fit failure or unmet predicate must fail
closed to ordinary training and be reported as `disabled`, not deleted.

The stated thresholds are proposed pre-run operational values, not values to be
learned from test results. If a prior development set is used to select them, it must be named,
disjoint from both evaluation scenarios, and its selection evidence registered.

## Two real matched-path scenarios

### N — natural harmful/noisy candidate: CIFAR-10N human annotation disagreement

Train on the real CIFAR-10N aggregate labels. The enable policy sees only
training losses and schedule state. Per-annotator disagreement and any clean
reference labels are withheld from the policy; they may be used only after the
run to audit whether the policy targeted the natural noisy cohort. This is not
a synthetic flip injection.

### H — benign-hard/valueful candidate: clean-label real-image cohort

Use a real clean-label image training corpus with no label modifications. The
candidate cohort is the policy's own early high-loss cohort, whose labels are
the dataset's clean labels. It is a benign-hard candidate, not a fabricated
noise block. Retention/value is assessed from general test accuracy plus an
independently frozen hard-example endpoint (for example a pre-defined
clean-label hard validation/test subset); the endpoint definition must be
recorded before runs. The policy must not access that endpoint.

The source/version of both datasets, split hashes, licenses, preprocessing, and
any clean-label/hardness audit labels must be recorded in the run manifest.

## Matched-path contract

For each locked seed (at least six), all arms share initialization, data split,
data order or an explicitly shared randomization stream, augmentation draws,
optimizer, LR schedule, total updates, and evaluation checkpoint. Branching is
allowed only at the frozen intervention point. If exact batch matching becomes
impossible after quarantine, record the deterministic sampling algorithm and
use a common seed schedule; do not call paths matched without this record.

Required arms per scenario:

| arm | action | role |
|---|---|---|
| A0 ordinary | no quarantine | safe fallback and policy-disabled target |
| A1 fixed quarantine | fixed fraction/cohort through decay | comparator for timing alone |
| A2 enabled policy | frozen observable signal and re-admission | primary intervention |
| A3 oracle/no-quarantine diagnostic | no intervention or an explicitly separated oracle diagnostic | upper-bound/context only; never used by the policy |

No arm may be dropped after seeing results. Record every seed, including failed
or disabled-policy seeds.

## Primary falsifiers and analysis

The registered paired unit is a seed. Report all seed-level paired differences,
median, mean, an interval appropriate to the small sample, and count of
positive pairs. Do not present a six-seed result as a population guarantee.

1. **N usefulness:** A2 must improve the frozen primary test metric over A0 and
   A1 by the predeclared practical margin (`0.005` accuracy unless a justified
   metric-specific margin is frozen before running). If it does not, the
   practical enable claim fails for N.
2. **H safety:** The observable predicates should disable. If they enable, A2
   must not reduce either the primary metric or the frozen hard-example metric
   by more than the same practical margin versus A0. Otherwise the
   harmful-versus-benign discrimination claim fails.
3. **Signal discrimination:** Report enable rate, estimated mass, BIC deltas,
   cohort stability, and post-hoc enrichment for natural annotation disagreement
   in N. A signal that enables equally on H and N, or cannot stably fit in N,
   fails the intended deployment decision even if one endpoint is favorable.

Success in N alone is insufficient. Success in H alone is insufficient. A
negative result is scientific evidence about the enable rule, not evidence that
the fixed-quadratic/strong-convex theorems are false.

## Theory boundary and allowed interpretation

The experiment may test whether a path-control heuristic is useful. It cannot
upgrade T1, T2, or T3 to a theorem for CNNs, nonconvex training, shuffled SGD,
or these datasets. Any resulting paper language must say that T1 concerns a
fixed quadratic; T2's general guarantee is its stated strong-convex full-norm
condition (with a separate directional precondition); T3 is the stated affine
scalar recursion. The deep-learning outcomes are empirical transfer results.

## Preflight and required outputs (before authorized formal run)

1. Lock code revision, dataset/split hashes, configs, locked seeds, exact
   command, environment, output directory, and duplicate-run guard.
2. Verify the policy cannot read test outcomes, per-annotator labels, clean
   audit labels, or post-hoc cohort membership.
3. Run focused unit tests for policy determinism, disabled fallback, matched
   path/randomization recording, and metric extraction. Do not replace a
   formal run with an unregistered exploratory sweep.
4. Emit per-run configuration, event log, enable decision inputs, cohort IDs
   (hashed or safely stored), seed metrics, failure status, and SHA-256 manifest.
5. Register the formal result in `evidence/experiment_registry.json` only after
   it scientifically executes. Infrastructure/preflight failure is recorded
   separately and is not an experimental negative result.
