# ICLR portfolio iteration workspace

Status: 2026-08-22. This workspace contains the three strongest manuscripts
selected from `01_p5/research_portfolio_36`, together with frozen baselines,
auditable revisions, isolated reviews, and reproducibility records.

## Current decision summary

| Priority | Project | Best verified artifact | Latest formal ICLR-style review | Current evidence ceiling | Decision |
|---:|---|---|---|---:|---|
| 1 | [`a11-correlated-majority-vote`](a11-correlated-majority-vote/README.md) | Revision 05; independently verified and frozen | Round 04 panel `[6,4,4,6,4]`, median 4, meta 4 | 6 | Strongest paper; run the same-endpoint comparator study before another blind panel |
| 2 | [`a2-subgroup-mix-ranking`](a2-subgroup-mix-ranking/README.md) | Revision 04; targeted verification `partially_resolved` | Round 02 panel `[4,4,4,4,4]`, median 4, meta 4 | 4 | Manuscript-only iteration has plateaued; obtain application-contract and shift/cost evidence |
| 3 | [`a2-erase-late-absorb-early`](a2-erase-late-absorb-early/README.md) | Revision 02; final best-integrity checkpoint | Round 02 panel `[4,2,2,2,4]`, median 2, meta 2 | 2 | Stop prose-only iteration; resume only with a genuinely new theorem or claim-linked experiment |

The ratings are internal ICLR-style judgments, not acceptance probabilities.
An unscored targeted verification is never represented as a score increase.

## 1. A11: correlated majority-vote replay

- **Paper:** [`manuscript/paper.pdf`](a11-correlated-majority-vote/manuscript/paper.pdf)
- **Best checkpoint:** [`review/best_checkpoint.json`](a11-correlated-majority-vote/review/best_checkpoint.json)
- **Round-5 verification:** [`review/revision_05_independent_verification.json`](a11-correlated-majority-vote/review/revision_05_independent_verification.json)
- **Frozen Round-5 root:** `b889ed62195b1b38ffe21b5b846eed1b5b60688d19b7cab12cb18b6961a6b9d7` (134 files)
- **Artifact:** TeX `b0e1207a…e54ffa`; PDF `e5d9f783…2b2c89`; 16 pages; Conclusion ends on page 9.

The paper studies binary pass-count majority decisions under a
count-exchangeable random-prefix replay model. With the true count law it has
an exact oracle conditional flip identity and tower result. The implemented
plug-in rule instead freezes a fitted score on FIT and uses a finite
empirical-Bernstein/Bonferroni screen on CAL; TEST is descriptive. On the
11,607-task OMR count artifact, the reported descriptive TEST readout at
`alpha=.05` is replay flip `0.0249` and rollout-count reduction `80.9%`.

Round 05 fixed the even-`N` center proof, empty odd-budget semantics and
separate FULL-`N` fallback, the TV-radius infeasible sentinel, the misleading
demeaned-correlation interpretation, Table 5 labels, and Figure 1 package
locality. The independent verifier reproduced 54/54 claim checks, the formal
boundary audits, conditional byte-exact replay, deterministic Figure 1, two
byte-identical clean builds, and visual QA.

**Still needed to move the score:**

1. A predeclared same-endpoint comparison against a finite-population WoR-CS
   majority stopper and BAYES-UNIF, using the frozen manifest/split and exact
   replay loss. Until then there is no supported efficiency separation.
2. A chronological rollout study with stopped-answer gold correctness and
   generated-token, latency, cancellation, and post-stop-work telemetry.
3. Raw-parquet-to-manifest clean-room provenance and local reruns for retained
   secondary carriers, if strong reproducibility/cross-carrier claims remain.

Score history: R1 `[6,4,6,4,6]` (median 6, meta 4); R2 `[6,6,4,6,4]`
(median 6, meta 4); R3 `[6,4,4,4,4]` (median 4, meta 4); R4
`[6,4,4,6,4]` (median 4, meta 4); R5 is an unscored targeted verification.

## 2. A2: subgroup-mixture ranking

- **Paper:** [`manuscript/paper.pdf`](a2-subgroup-mix-ranking/manuscript/paper.pdf)
- **Best checkpoint:** [`review/best_checkpoint.json`](a2-subgroup-mix-ranking/review/best_checkpoint.json)
- **Revision-04 verification:** [`review/revision_04_independent_verification.json`](a2-subgroup-mix-ranking/review/revision_04_independent_verification.json)
- **Artifact:** TeX `269e95c7…ad0c8`; PDF `e7a47b1f…804fc9`; 14 pages.

This paper studies safe ranking/abstention under subgroup-mixture turnover,
using weighted Clopper-Pearson risk bands and a same-CAL-point paired-difference
MPB regret gate. Revision 04 correctly limits the M9/M10 argument to a
fixed-statistic width counterfactual, preserves the corrected conditional P3,
and labels E03 as snapshot-only evidence.

The remaining blockers are evidentiary: a reviewer-safe E03 application
contract and clean replay; a natural time/geographic mixture shift with
within-group invariance audit; and a precommitted abstention/fallback cost or
utility endpoint. The current evidence ceiling is 4, so more prose-only
iteration is not useful. R1 was `[6,4,4,4,4]` (meta 4); R2 was
`[4,4,4,4,4]` (meta 4); R3 had only one technical score of 4; R4 was targeted
verification, not a score round.

## 3. A2: erase-late / absorb-early theory

- **Paper:** [`manuscript/paper.pdf`](a2-erase-late-absorb-early/manuscript/paper.pdf)
- **Best checkpoint:** [`review/best_checkpoint.json`](a2-erase-late-absorb-early/review/best_checkpoint.json)
- **Artifact:** TeX `b3a06e3b…edb07a`; PDF `5a3f1b78…ad11e`; 5 pages.

The live paper is a theory-only package of three independent scoped results:
a fixed quadratic exposure product, an ambient-Hessian gradient-descent
contraction, and an affine scalar conditional-mean recursion. It deliberately
makes no comparator-path, intervention, unlearning, empirical, deployment, or
novelty claim.

The latest panel `[4,2,2,2,4]` (median/meta 2, contribution median 1) places a
hard novelty/evidence ceiling of 2. Resume only if there is either a new
matched two-trajectory, time-ordered data-block perturbation theorem with a
clear prior-work separation and falsifier, or a new reproducible practical
experiment tied to a defensible claim.

## Recommended next execution order

No new scientific experiment, GPU job, remote task, or additional TEST read
was launched during the current revision cycle.

1. **Smallest high-information run:** A11 WoR-CS + BAYES-UNIF same-endpoint
   comparison on the frozen reviewer-safe manifest.
2. **Decisive but larger run:** A11 ordered/gold/cost protocol on an authorized
   carrier with chronology and cancellation integrity checks.
3. **Then subgroup:** recover the E03 application contract and run the natural
   shift/invariance plus fallback-cost study.
4. **Keep erase stopped** until a new scientific contribution exists.

## Directory contract

- `baseline/`: immutable public manuscript snapshot and checksum manifest.
- `manuscript/`: active buildable paper checkpoint.
- `remote_snapshot/`: read-only evidence input, never trusted automatically.
- `state/`: assumptions, decisions, score history, and next iteration plan.
- `evidence/`: claim maps, provenance, inventories, and run registry.
- `literature/`: citation requests and verified citation lock.
- `review/`: frozen submissions, isolated reviews, ledgers, and checkpoint record.
- `build/`: machine-readable build and validation records.
- `experiments/`: frozen protocols and launchers; execution still requires
  authorization under the repository experiment policy.

Official ICLR 2027 requirements and style-file hashes are recorded in
[`ICLR2027_REQUIREMENTS.md`](ICLR2027_REQUIREMENTS.md).
