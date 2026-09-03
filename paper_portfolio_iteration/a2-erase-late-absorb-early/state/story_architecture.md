# Story architecture — revision 01

Date: 2026-08-22. Authority: the frozen public source `d53d4e...825d72`.

## Candidate stories

| Story | Problem and insight | Evidence fit | Main objection | Decision |
|---|---|---|---|---|
| S1: controlled-dynamics theory | Exposure products and clean-tail contraction distinguish three formal regimes. | Directly supported by stated T1--T3 assumptions and proofs. | Narrow practical significance. | Selected. |
| S2: general noisy-label/quarantine method | A training-visible signal can safely quarantine harmful data. | No public executed two-scenario evidence; remote candidate is a different revision. | Unsafe deployment and false positives. | Rejected. |
| S3: remote-result recovery paper | Reuse remote scripts and JSON to support the baseline's real-image narrative. | Remote package is internally checkable but has `e9d...` source hash, not the public `d53d...` hash. | Invalid provenance transfer. | Rejected. |

## Selected architecture

**Identity.** A bounded theoretical paper: path-dependent block interventions admit exact product or contraction statements in three specified regimes; whether a training-visible policy helps on real data remains an open, falsifiable question.

Headline claims and falsifiers:

1. **T1 (fixed quadratic):** the exact product and first-order exposure relation hold under its stability condition. Falsifier: an algebraic counterexample satisfying those conditions.
2. **T2 (strong-convex clean tail):** full-norm error contracts under its smoothness, strong-convexity, step-size, and tail-budget conditions; a directional acceleration needs the additional invariant-subspace premise. Falsifier: a counterexample satisfying all stated premises.
3. **T3 (affine scalar conditional mean):** the conditional-mean product and exponential upper bound hold. Falsifier: a scalar process satisfying the stated recursion and conditional-zero-mean condition that violates the identity.
4. **Planned empirical claim:** no current claim. PLAN-ENABLE-001 can falsify a future conditional policy claim through N usefulness, H safety, or signal-discrimination failure.

Section budget: introduction and setup (1.5 pages); formal results and active proofs (4.5); empirical-status/falsification path and limitations (1); related work/reproducibility/ethics/AI-use (1); references excluded. No evidence figure or result table is assigned until a same-version evidence chain is registered.
