# Round-4 claim-scope audit

Date: 2026-08-22  
Scope: current working manuscript and evidence records only. The frozen
`review/round_03/submission/`, its five reviews/meta-review, `baseline/`, and
`remote_snapshot/` were not modified.

## Implemented scope corrections

- **MR-01:** Theorem 1 is explicitly an oracle statement under the true
  count law $H_\star$. BAYES-H instead uses the distinct
  $\widehat H_{\mathrm{FIT}}$ plug-in stopping score. Its only operational
  result is Theorem 2's FIT-frozen, CAL-screened marginal replay guarantee.
  The theorem and protocol state independent i.i.d. FIT/CAL tasks, bounded
  per-task loss, $J=64$, per-rule $\delta_r=0.05/64$, Bonferroni, and a
  single descriptive TEST readout.
- **MR-02:** Withdrawn broad robustness, frontier/optimality, and mechanism
  language. The retained E3 counterexample says that at $\alpha=.05$ and
  $\delta=.15$, BAYES-H is invalid at 5.2% while FIXED-EB is valid at 4.7%.
  All margin results are marked synthetic, exploratory, and nonconfirmatory.
- **MR-03:** The carrier text distinguishes the reviewer-safe derivation
  (22,230 source rows, 12,423 valid count rows, 11,607 deduplicated problems,
  $K=32$, seed 20260815, FIT/CAL/TEST 4000/4000/3607) from a raw-parquet
  reconstruction. The main JSON is byte-exactly replayable only conditional
  on that derived manifest; secondary runners are provenance-only. OpenR1 is
  stated as 9,374 raw unique problems, exactly-two-deduplicated-rollout
  predicate, 8,853 analyzed, and 3000/3000/2853 split.
- **MR-07:** Figure 1 is now a code-native Matplotlib schematic whose numeric
  labels are parsed only from hash-pinned frozen JSON. It separates a random
  replay prefix from chronology, oracle $H_\star$ from the fitted score/CAL
  screen, and replay quantities from synthetic exploratory drift. The
  renderer uses `Decimal(..., ROUND_HALF_UP)`, so frozen
  `saving_vs_full=0.8085` displays as 80.9%. The $\alpha=.10$ BAYES-H 84.0%
  cell is not bolded above FIXED-EB's 84.4%.
- **NP-01:** Closest work is explicitly positioned against Waudby-Smith and
  Ramdas (finite-population without-replacement inference), Rossell and
  Müller, and Novikov (Bayesian sequential stopping). The supported
  novelty is only independent FIT/CAL screening of a finite frozen plug-in
  score family for exact task-level replay loss.

## Verification record

| Gate | Result |
| --- | --- |
| Current hash-bound claim audit | 49 PASS / 0 FAIL |
| Derived-manifest replay | byte-identical main JSON, SHA-256 `b114c72d…c8a5f7` |
| Monotonicity boundary audit | PASS: 21,824 reversal cases; 11,904 zero-$F_0$ support cases |
| Frozen remote numeric audit | 426 PASS / 0 FAIL / 3 external-provenance boundaries |
| Frozen Round-3 snapshot | PASS: 113 files, root `683d2d58…dc6ee` |
| Citation lock | active citekeys locked; closest-work request/lock consistency PASS |
| Clean builds | two isolated builds byte-identical, PDF SHA-256 `fefe25a9…5dbb3`, 16 pages |

## Figure-1 page-width QA

The active figure is a compact 7.8 x 4.05 inch two-row canvas. Its smallest
raw panel label is 8.25pt; at the roughly 6.5 inch manuscript text width it
is approximately 6.9pt effective. Page 2 was rendered at 144 dpi and checked
at 100% page width: the stacked oracle/fitted/CAL panel, replay-prefix label,
80.9% label, and synthetic-drift boundary are readable without zoom; there is
no stray square, clipping, or overlap. The caption uses the matching `(b,
top)` and `(b, middle)` directions.

Pages 2, 6, 8, 9, 14, and 16 were visually inspected. No clipping, overlap,
unreadable affected table/figure, or misleading $\alpha=.10$ emphasis was
observed. The Conclusion is on page 9.

## Remaining evidence boundaries

- No real chronological ordered-rollout validation.
- No stopped-answer gold-correctness comparison or generated-token, latency,
  and cancellation telemetry.
- No clean-room raw-parquet reconstruction or local secondary-runner rerun.
- No target-population containment/coverage result for the assumed
  fitted-mixture TV sensitivity, nor a K-independent prefix-observable
  Appendix E(g) repair.
