# Minimal falsifiable real-shift protocol (planned; not executed)

## Purpose and decision

Test the paper's real deployment premise rather than optimize the width-surrogate allocation
story. The study is successful only if all predeclared isolation, invariance, certificate, and
cost records are available; it may support a **conditional** real-shift claim, never prove
invariance from a nonsignificant diagnostic. A detected invariance failure falsifies the
certificate's use for that target and triggers claim narrowing, not post-hoc regrouping.

## Data lock required before execution

Use **Functional Map of the World (fMoW), Christie et al., CVPR 2018, DOI
10.1109/CVPR.2018.00646**, through a versioned WILDS/fMoW acquisition recorded before any
outcome is read. The original paper documents image timestamp and UTM geographic-zone metadata
(including location/time reasoning); it is the locked public candidate. The precise downloadable
release identifier, license text, acquisition checksum, and WILDS package version remain a
preflight field and must be recorded before execution---do not silently substitute another
dataset after inspecting outcomes. Before touching outcomes, record the metadata schema, exact
timestamp and region/UTM fields, label/group definition, documented filtering, and split hashes.

## Splits and actions to lock in the formal preregistration

1. Choose a source time interval and geographic region set before model training. Train a fixed
   pool of at least three candidate models on source-time/source-region FIT only.
2. Create disjoint CAL and OUTER blocks. CAL is drawn only from the source environment; OUTER
   comprises two natural targets: a later-time target and a held-out-region target. No target
   labels enter selection, allocation, threshold selection, or candidate choice.
3. Define subgroups from the frozen label taxonomy (not outcome-tuned time/region bins). For
   each target, its observed label/subgroup mixture is the natural `w_target`; also evaluate a
   predeclared fixed set of observed target mixtures if multiple target blocks exist.
4. Use the exact paired MPB gate as the primary method and the absolute CP gate as a secondary
   exact comparator. The normal/CLT variant may be reported in a separately labelled
   asymptotic diagnostic panel only.
5. Fix `delta`, `tau`, candidate pool, seeds, training budget, and all aggregation before any
   OUTER labels are read. Report per-target and pooled results without replacing failed blocks.

## Assumption audit (a required endpoint)

For every candidate and subgroup, compare source-CAL and each target-OUTER conditional loss.
Report signed difference, sample sizes, a predeclared uncertainty interval/test, and multiplicity
handling. Also report target versus source subgroup-mixture distance. This audit is descriptive:

- If a material conditional-loss change is detected under the predeclared rule, label the
  turnover assumption failed for that target. Do **not** claim finite-sample target coverage.
- If it is not detected, report uncertainty and “not rejected at available power,” not
  invariance proved.

## Primary endpoints

| Endpoint | Required definition |
|---|---|
| Natural shift | source/target time and region, target mixture, mixture distance |
| Certificate | commit rate; committed true regret; fraction committed with regret `<= tau`; maximum committed regret |
| Abstention | abstain rate and forced point-estimate regret on abstained cases |
| Fallback | predeclared status-quo anchor regret/loss on every abstention; switch rate |
| Cost | CAL labels used, any recorded latency/compute, and an explicit `missing` marker for every unavailable operator cost; if unavailable, a predeclared sensitivity curve over abstention penalties (not a utility estimate) |
| Invariance | subgroup/candidate conditional-loss audit with uncertainty and pass/fail/indeterminate status |

The fallback must be one frozen status-quo model selected from FIT/source policy before target
outcomes. If latency or monetary cost cannot be observed, do not impute it; report it as
`missing` and present a sensitivity curve for total loss $L_c=\text{classification loss}+
c\,\mathbb{1}\{\text{abstain}\}$ over a preregistered grid of abstract penalties $c$.
This is a decision sensitivity analysis, not a measured utility conclusion.

## Falsification and interpretation

- **Target invariance fails:** restrict the theorem/evaluation claim to constructed label-prior
  shift; retain no natural-shift safety statement for that target.
- **Invariant-looking target but a committed violation:** investigate split/provenance/code;
  do not claim coverage pending diagnosis.
- **Exact gate mostly abstains:** report it as the certificate price and fallback cost; do not
  rescue it with normal/CLT results.
- **Fallback dominates or cost is unfavorable:** report the operational limitation even if
  coverage holds.

## Minimal execution record

Before execution create a registry entry with exact command, immutable code/config hash,
dataset/split hashes, seeds, environment, timestamp, log path, output path, and validation
checks. After execution, register negative/infrastructure outcomes separately. This protocol
does not authorize an experiment; it is ready for an explicit execution request.
