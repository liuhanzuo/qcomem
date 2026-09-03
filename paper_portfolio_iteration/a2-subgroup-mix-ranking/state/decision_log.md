# Decision log

## 2026-08-22 — initialize Phase 0/1 audit

**Decision.** Freeze the local baseline and treat the r1915 snapshot as read-only evidence.

**Evidence.** `baseline`, `manuscript`, and `remote_snapshot/paper` have identical paper
source/PDF SHA-256 values.  `remote_snapshot/MANIFEST.md` identifies r1915 and records its
historical verifier/build assertions.

**Rationale.** This preserves an auditable starting point and respects the no-manuscript-edit,
no-experiment, and no-remote-operation scope.

## 2026-08-22 — protect the certificate; narrow the allocation theorem

**Alternatives considered.** (A) defend Prop. 5 as an actual regret-UCB allocation theorem;
(B) remove M3.5; (C) retain M3.5 explicitly as a width-surrogate analysis while preserving the
separate paired-regret certificate.

**Decision.** Select C as the immediate low-cost repair.  The initial portfolio screen records
that Eq. `(mm)`/Prop. 5 lacks candidate-indexed uncertainty and only restores the width
surrogate.  It separately says the paired-regret certificate remains valid.  A direct proof of
the actual UCB objective is a later, higher-risk alternative, not assumed achievable.

**Required prose/proof changes (planned, not made).** Rename the section/objective to a
certificate-width surrogate; replace “minimax”/“safe” allocation implications that refer to
actual regret UCB; state the proposition's exact fixed-surrogate conditions; move empirical
commit-rate outcomes to descriptive evidence.  Keep P1/P2/P3 definitions and their quantities
unmodified except for cross-references that would imply allocation optimality.

## 2026-08-22 — choose a falsifiable real-shift validation before allocation expansion

**Alternatives considered.** (A) add larger synthetic allocation sweeps; (B) seek a proof
equating the surrogate to the true UCB; (C) test the core certificate on natural temporal and
geographic subgroup-mixture shifts while measuring invariance and fallback cost.

**Decision.** Select C for the next authorized empirical action.  It attacks the most
decision-relevant untested premise, falsifies the real deployment story if it fails, and is
smaller than a broad benchmark suite.  The concrete protocol is frozen in
`experiments/decisive_experiment_protocol.md` but has not been run.

## 2026-08-22 — Phase 2 story selection and revision-01 scope repair

**Alternatives.** (S1) center the paper on the conditional simultaneous paired-regret
certificate; (S2) retain allocation as an actual regret-UCB optimization story; (S3) center
the paper on operational abstention under natural shifts.

**Decision.** Select S1. `state/story_architecture.md` records the three stories, their
falsifiers, and the section/evidence architecture. S2 is rejected because Eq. `(mm)` removes
candidate-specific empirical terms, comparator maxima, and actual UCB widths. S3 is rejected
as a present-tense story because no natural-shift run or precommitted fallback-cost measurement
exists.

**Revision.** Relabel Eq. `(mm)` and Prop.~5 as a fixed-mixture certificate-width surrogate;
remove safe-allocation, actual-UCB-optimization, conditional-gate, and dominance language.
Keep the P1/P2/P3 simultaneous paired-regret certificate unchanged. Split exact CP/Hoeffding/
Maurer--Pontil evidence from the normal/CLT asymptotic diagnostic in abstract, tables,
interpretation, limitations, conclusion, and appendix. Record fallback costs as missing and
specify a sensitivity-only protocol for the planned study.

**Readiness / literature.** Lock fMoW (Christie et al., CVPR 2018, DOI
10.1109/CVPR.2018.00646) as the preflight dataset candidate because its original paper
documents timestamp and UTM geography; executable release/version/license/checksum remain
mandatory preflight fields. Verify central primary metadata and create
`literature/closest_work_matrix.md`. Official ICLR 2027 guidance and style parity are verified
separately in the state/build record.

## 2026-08-22 — change-verifier correction: zero coefficients in Prop. 5

**Finding.** A fresh change verifier correctly identified that the phrase ``constant over
active groups'' was vacuous for $W=\{e_0\}$ and contradicted the appendix counterexample.

**Decision.** Use the continuous extended-value width-surrogate convention: a zero coefficient
$c_g=w_g\beta_g$ receives zero allocation, while a positive coefficient cannot receive zero
allocation. For a nondegenerate fixed mixture, all-$G$ uniform allocation is uniquely optimal
iff every $c_g$ is the same strictly positive value. A mixed zero/positive vector has a boundary
optimum and makes all-$G$ uniform strictly suboptimal; the all-zero case is identically zero and
nonunique. The appendix now explains that $(1197,1,1,1)$ is the integer implementation's
one-label-per-inactive-group near-boundary realization of the continuous $(1200,0,0,0)$
solution. This remains a width-surrogate fact only.

## 2026-08-22 — change-verifier correction: one-mixture versus multi-mixture scope

**Finding.** Eq.~(WS) is $\min_n\max_{w\in W}$, but the preceding water-filling formula had
been introduced as though a fixed binding mixture supplied that general optimum. In general the
multi-mixture solution depends on the dual mixture $\mu$.

**Decision.** Restrict Eq.~\eqref{eq:water} and Prop.~5 explicitly to the one-mixture
subproblem $W=\{w\}$. Immediately state that general $W$ does not follow by selecting an
arbitrary binding mixture and is instead handled by the subsequent multi-mixture dual. This
is a scope repair only; it adds no allocation, certificate, or deployment claim.

## 2026-08-22 — final finite-dual and link-layout clarity lock

**Decision.** State the multi-mixture surrogate only for a finite predeclared set
$W=\{w_1,\ldots,w_K\}$ and write both dual maximizations explicitly as
$\mu\in\Delta^K$. Use `hyperref` with `hidelinks` so citation/reference boxes do not obscure
the PDF. These are notation and layout repairs; they do not expand the allocation claim.

## 2026-08-22 — page-limit correction after full-page QA

**Finding.** The prior 14-page PDF was described as having nine main-text pages, but the
Conclusion began on PDF page 10. It therefore did not meet the ICLR nine-page main-text limit.

**Decision.** Compress only redundant prose and table captions in M3.5, related work,
limitations, and the conclusion. Preserve all certificate, exact-versus-asymptotic, and
width-surrogate boundaries. In the rebuilt PDF, the complete Conclusion ends on page 9;
References begins immediately afterward on page 9 and continues on page 10. Pages 8--10 were
rendered for visual QA with `hidelinks`; no clipping or visible link boxes were observed. No
baseline or checkpoint artifact was modified.

## 2026-08-22 — Round-2 technical and evidence boundary repair

**Decision.** Repair P3 with the explicit self-comparator form
`max{0,max_{j != i} sum_g w_g UCB_ijg}`; cap historical displayed certificate values at zero
in the formal definition rather than reinterpret negative values as regret certificates.

**Evidence and verification.** The deterministic CPU audit
`evidence/audit_paired_self_comparator.py` checks the frozen five-seed frontier at
tau in {0, .02, .04, .10} and a strictly-dominant two-model boundary. Capping cannot alter a
nonnegative-tolerance gate; this is an artifact audit, not a replay or coverage experiment.

**Adaptive result.** Remove the claimed adaptive finite-sample confidence-sequence theorem and
its result column. The snapshot does not establish the observable subgroup process, filtration,
sampling law, or n=0/1 treatment required for the actual adaptive implementation.

**Evidence language.** Replace universal paired-tightness language with conditional
shared-error cancellation; call all 1.000 quantities dependent held-out estimated-regret or
violation diagnostics, never nominal coverage. Restrict empirical evidence to constructed
mixtures. M3 only retains aggregate 3-seed rows, so allocation directions are exploratory.

**Provenance.** Separate historical frozen hashes from mutable current-manuscript hashes and
record a canonical reviewer-safe package hash specification. A full external reproducibility
package remains open because raw prediction/loss arrays, configuration/split manifests, and an
environment lock are not present in the delivered scope.

## 2026-08-22 — Round-3 corrective plateau: conditional certificate and oracle-M3 repair

**Trigger.** The Round-2 technical review and meta-review correctly identified an unstated
sampling contract behind CP/Hoeffding/MPB, a false Proposition 3, and non-observable class
stratification in M3. These are correctness issues, not presentation opportunities.

**Decision.** Retain the central paired theorem only as a conditional result. The manuscript now
requires FIT-only fixed candidates/design choices, FIT--CAL independence, conditionally i.i.d.
within-group CAL draws, and same-point bounded paired observations. It states the exact
Bonferroni counts: `kG` CP cells at `delta/(kG)` and `k(k-1)G` ordered-pair/group Hoeffding/MPB
cells at `delta/[k(k-1)G]`, with the MPB `n_g >= 2` condition. The event is still simultaneous
over all simplex weights and CAL-data-dependent choices; this is a conditional algebraic result,
not a claim that E03's historical execution had the required split or sampling law.

**Proposition repair.** Replace the invalid width condition by a valid two-candidate statement:
explicit `U_a<U_b`, point estimate and true risk favoring `b`, then M1 chooses `a` and has
correctly signed regret `R_a-R_b>0`. A numerical interval/risk witness is included. This is the
narrowest mechanism claim consistent with the M1 selector.

**M3/M3.5 scope.** Reclassify all M3/M3.5 prose, tables, evidence maps, and historical M12
language as oracle-stratified constructed-mixture diagnostics. Ground-truth class cannot direct
an unlabeled-pool query unless an external pre-label stratified frame exists. No such frame,
collection trace, or contract evidence is present. The width surrogate remains distinct from the
actual regret UCB and from a label-acquisition or deployment policy.

**Story choice.** The exact paired MPB frontier is the primary empirical story; normal `0.503`
remains an asymptotic diagnostic. The abstract's former ``abstaining exactly'' language is
weakened to the supported aggregate association. Natural-shift, fallback-cost, and realized
coverage claims are not strengthened. No scientific experiment was run.

## 2026-08-22 — Round-4 Appendix-F allocation scope repair

**Trigger.** The isolated Round-3 technical review identified that Appendix F described
finite budget-grid rows produced with new data-derived selections, but then extended a
full-split margin/width calculation to an allocation-wide conclusion. It also correctly noted
that the displayed UCB-to-mean relation was not the deterministic algebra used by the formula.

**Alternatives considered.** (A) retain a conclusion beyond the stored schedules by proving a
uniform result over changed empirical paired means, $F_0$, $i^\star$, and $\Delta$; (B) delete
M9/M10 entirely; (C) retain only the available fixed-statistic Hoeffding-width calculation and
the finite stored-grid observations. A requires new formal and empirical support not present in
the frozen artifacts; B would discard a clearly bounded diagnostic.

**Decision.** Select C. The retained calculation freezes the empirical paired means, $F_0$,
$i^\star$, and $\Delta(w)$ of each full-split row, writes the paired upper formula as empirical
mean plus nonnegative width, and derives the admission condition by deterministic algebra. Its
monotonicity statement is limited to reducing capped counts inside that fixed-statistic
Hoeffding counterfactual. E07/E08 rows are described only as snapshot-reported finite-grid
diagnostics. Figure 1 and M12 are synchronized to that boundary. No claim is retained for a
new CAL draw, a new data-derived selector, arbitrary MPB widths, or a general allocation policy.

**M6 and E03 boundaries.** The first M6 occurrence now identifies it as an
asymptotic/descriptive relative diagnostic; the inspected exact-relative result is
status-quo-only, so conditional finite-sample deployment language is reserved for absolute
M2.5. E03 remains snapshot-reported with no application contract: immutable split identities,
sampling/collection provenance, candidate isolation, paired sufficient statistics, executable
UCB configuration, and environment lock remain external evidence gaps. No scientific run,
download, scheduler submission, or frozen-object modification was authorized or performed.

## 2026-08-22 — checkpoint selection after Round-4 change verification

**Recorded score trajectory.** Round 1 was a full five-reviewer panel with ratings
`[6,4,4,4,4]`, median `4`, and meta-rating `4`. Round 2 was a full five-reviewer panel with
ratings `[4,4,4,4,4]`, median `4`, meta-rating `4`, and conservative evidence ceiling `4`.
Round 3 contains only one isolated technical-soundness reviewer rating of `4` on snapshot
`8e0822d2b79b3e0e8cd5702d40248003900fb554f228bfde1d25f431cebd18f2`; it is not a panel,
panel median, or meta-review result. Round 4 is targeted change verification with verdict
`partially_resolved`, not a scored review. Its recorded snapshot identity is
`4fb48a187264fe400fcfda23f8024741f45ea269c740530b4964705c86d08e2c`.

**Decision.** Select revision_04 as the best verified artifact, not as the best scored review
round. It has source SHA-256 `269e95c75e5c03b02a277617435f84126a0e7674f811fa2a04c3e114776ad0c8`
and PDF SHA-256 `e7a47b1f0f02e40fb55dbc79fb98a2b1cb0f3fa2eb3894a2fe1c06047b804fc9`.
Independent verification confirmed the fixed-statistic M9/M10 scope, M6's
asymptotic/descriptive and exact-relative-status-quo boundary, unchanged P3 proof, clean build,
and visual quality. The decision is an integrity/build checkpoint selection only.

**Checkpoint and stop-rule consequence.** Revision_04 cannot be reported as a score increase,
an acceptance-ready checkpoint, or a completed review loop: the last full panel and meta-review
remain Round 2 at `4`, Round 3 is single-reviewer only, and Round 4 was not scored. The current
evidence ceiling therefore remains `4`. A future full blind re-review must use fresh reviewers
and a fresh snapshot; it must not inherit Round-1--4 scores or outcomes.

**Conditions to reassess the ceiling.** The minimum evidence path is: (1) a reviewer-safe E03
application contract covering immutable split identity, collection/sampling law, candidate/FIT
isolation, paired sufficient statistics or raw replay inputs, executed UCB configuration, and
environment, with a clean replay where required; (2) a natural time/geographic subgroup-mixture
shift study plus a predeclared within-group invariance audit if retaining practical deployment
relevance; and (3) a precommitted fallback/abstention cost or utility endpoint, or measured
sensitivity analysis, if retaining operational abstention claims. These conditions record the
evidence needed for reconsideration; they do not authorize or assert new experiments.
