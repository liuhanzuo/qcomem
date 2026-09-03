# Phase 2 story and claim architecture — revision 01

Date: 2026-08-22. The three alternatives below were evaluated against the
supplied evidence only. In the current replay paper, `full-N` means the
corresponding binary full-budget pass-count decision, not a delivered-answer
majority or a gold answer.

## Option A — Deployment-safe online stopping

- Problem: reduce deployed majority-vote inference cost without hurting users.
- Structural gap: common stoppers lack a certificate under correlated,
  chronologically generated rollouts.
- Insight/method: BAYES-H's conditional flip certificate.
- Contribution boundary: would require ordered traces, gold correctness, and
  measured token/latency/cancellation telemetry.
- Strongest current evidence: none for its primary deployment claim.
- Likely objection: the count-only carrier cannot establish online validity or
  user-facing correctness.
- Must not claim: online safety, task accuracy preservation, token saving, or
  latency saving.
- Assessment: scientifically important but rejected because its central claim
  is not evidenced.

## Option B — Exact certificate for count-exchangeable replay (selected)

- Problem: quantify when a prefix majority can stand in for the corresponding
  full-$N$ majority under a specified count-exchangeable replay model.
- Structural gap: a fixed-budget or heuristic rule does not use the
  per-prefix posterior uncertainty induced by a calibrated count mixture.
- Insight/method: condition on the observed prefix, compute the posterior
  full-vote flip probability, and calibrate a finite rule family on held-out
  problems.
- Contribution boundary: the endpoint is binary prefix/full-N pass-count
  agreement and the cost proxy is $1-\bar k/N$; neither is a delivered-answer
  majority, gold correctness, or operational cost. Chronological deployment
  order is outside the certificate.
- Strongest current evidence: supplied exact-DP FIT/CAL/TEST replay artifacts,
  a disjoint shard, two additional count/replay carriers, and explicitly
  synthetic ordered-drift stress.
- Likely objection: replay validity is not deployment validity.
- Must not claim: arbitrary online validity, actual token/latency/cancellation
  savings, or superiority at every level and endpoint.
- Assessment: strongest evidence support and lowest overclaiming risk; selected.

## Option C — Robustness geometry of discrete stopping states

- Problem: characterize how a frozen replay certificate changes under prior
  shift and restricted stopping-state subsets.
- Structural gap: scalar mixture-shift bounds can conceal the discrete geometry
  of the induced per-count flip profile.
- Insight/method: exact total-variation radius and certificate-ordered
  state-subset repair.
- Contribution boundary: all analyses reuse the supplied count/replay
  artifacts and do not add an online result.
- Strongest current evidence: supplied TV, cap-repair, and discrete-geometry
  artifacts.
- Likely objection: a dense appendix mechanism risks obscuring the paper's
  primary result and remains replay-scoped.
- Must not claim: that the repair makes an online system deployable.
- Assessment: technically rich, but should remain a secondary appendix story
  because it does not close the decision-driving empirical gap.

## Selected identity, claims, and falsification paths

**Identity.** A finite-sample, mixture-posterior certificate for agreement
between an adaptive prefix majority and its full-$N$ majority under
count-exchangeable replay, with explicit evidence boundaries for ordered
deployment and operational cost.

| Headline claim | Evidence and evaluation | Falsification path |
|---|---|---|
| Conditional certificate controls replay full-vote flip under its assumptions. | Theorem and exact-DP artifacts; C01. | Exhibit a legal replay stopping state with certificate $\le\alpha$ whose population flip exceeds $\alpha$. |
| BAYES-H reduces mean replay rollout count at named replay flip levels on supplied carriers. | FIT/CAL/TEST artifacts; C02. | Recompute the registered exact DP and disagree at printed precision. |
| Synthetic drift analyses map a failure boundary, not real order. | E1--E3 artifacts; C03. | Re-run the specified mechanism and obtain a contradictory table. |
| Online correctness and operational-cost claims are open. | Missing-evidence registry; C04. | Execute the registered ordered, gold-correctness, telemetry protocol. |

## Section architecture

| Section | Function | Main-text budget | Evidence/asset |
|---|---|---:|---|
| Abstract and introduction | State replay identity and exclusions up front. | 1.2 pp | Figure 1 (conceptual, non-evidence) |
| Related work | Position against self-consistency, ASC, ESC, and MARS. | 0.6 pp | Closest-work matrix |
| Model and method | Define replay, flip, certificate, and calibration. | 2.4 pp | Equations/theorems |
| Protocol and replay results | Name endpoint, count proxy, and FIT/CAL/TEST boundary. | 1.5 pp | Main replay table |
| Synthetic drift and limitations | State non-transfer to natural order/correctness/cost. | 1.0 pp | Drift tables in appendix |
| Conclusion/statements | Preserve the evidence boundary and reproducibility/AI disclosures. | 0.4 pp | Reproduction package |

The title/abstract/main table must keep the endpoint distinction visible. Any
future online experiment belongs to `EXP-ORDERED-CORRECTNESS-COST`; it may
support a new, carrier-scoped deployment story only after its preregistered
correctness and operational endpoints pass.
