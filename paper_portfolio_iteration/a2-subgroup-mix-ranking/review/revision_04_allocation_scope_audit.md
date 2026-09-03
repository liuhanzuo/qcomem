# Round-4 allocation-scope audit

Date: 2026-08-22  
Scope: minimum source/evidence repair after the Round-3 technical review. This record does not report a new scientific experiment.

## Scope repair

The live Appendix-F `$M9$/$M10$` passage now fixes one stored full-split row's empirical paired means, selected `$F_0$`, selected `$i^\star$`, and margin `$\Delta(w)$`.  Its calculation uses

`UCB_g(n_g) = empirical_paired_mean_g + bw_g(n_g)` with `bw_g(n_g) >= 0`.

Thus the displayed feasibility equivalence is deterministic algebra for those fixed statistics: the weighted UCB is `-Delta(w) + weighted width`, so its sign is determined by whether the weighted width is at most the fixed margin.  It does **not** invoke a probability event of the form `UCB_g >= mu_g`, and it makes no claim for a fresh CAL draw, a changed selected candidate or mixture, or a data-derived re-selection.  The stored M10 grid is described only as a finite stored-grid observation.

Figure 1's caption and rendered labels carry the same fixed-statistic/finite-grid limitation. The modified vector graphic is a relabeling of the existing frozen M10-derived plot, not a new result.

`$M6$` is now introduced as an asymptotic/descriptive relative-gate diagnostic. Its exact-relative comparison is explicitly status-quo-only/vacuous for a safe upgrade. The conditional finite-sample absolute `$M2.5$` gate remains the main certificate.

E03 remains snapshot-reported only. No split identity, collection/sampling protocol, candidate/FIT isolation evidence, paired sufficient statistics, or executed UCB implementation/configuration supplies an application contract for its historical values.

## Synchronized records

- `evidence/claim_evidence_map.tsv`: C02/C03/C07/C08 distinguish conditional or snapshot-reported scope from an application contract.
- `evidence/method_provenance.tsv`: M6 is labeled asymptotic/descriptive; M9/M10 is labeled fixed-statistic plus stored-grid only.
- `evidence/experiment_registry.json`: E03 documents the absent application-contract inputs; E07/E08 are confined to the stored/fixed-statistic artifacts.
- `review/issue_ledger.json`: TS05 (allocation scope) and TS06 (M6 signposting) are resolved by the source boundary; TS04 (E03 provenance) remains open.
- `state/decision_log.md`: records the chosen scope repair and the decision not to manufacture unsupported application claims.

## Static and frozen checks

`python3 build/revision_04_allocation_scope_static_audit.py` passed 43/43 checks. The guard requires the M6, M9/M10, M12, caption, claim-map, provenance, and rendered-figure boundary language, and rejects retired allocation-wide/reselected/coverage-event wording in the M9/M10 passage.

Frozen-artifact checks were rerun read-only: paired self-comparator (350 rows; all listed tau decisions unchanged), M5 `105/105`, M7 `28/28`, M8 `17/17`, M9 PASS, M10 PASS, M11 PASS, and M12 `245/245`. These attest only to historical artifact consistency. In particular, historical M10/M11 verifier wording does not broaden the current manuscript's fixed-statistic claim.

## Rebuild and visual QA

The final source-copy builds at `build/revision_04_final_build1/` and `build/revision_04_final_build2/`, plus `manuscript/paper.pdf`, are byte-identical:

- `paper.tex` SHA-256: `269e95c75e5c03b02a277617435f84126a0e7674f811fa2a04c3e114776ad0c8`
- `fig_m10_frontier.pdf` SHA-256: `e994b4b7b59edcdb3c4684c3ab97c555e3bc48a36acd80e8f53718588ed674ea`
- all three final `paper.pdf` copies SHA-256: `e7a47b1f0f02e40fb55dbc79fb98a2b1cb0f3fa2eb3894a2fe1c06047b804fc9`

Each PDF has 14 letter-size pages. Final logs contain no LaTeX error, undefined control sequence/reference/citation summary, overfull hbox, or duplicate PDF destination. Rendered pages 1, 6, 9, 12, 13, and 14 were inspected at 160 dpi; no clipping, overlap, missing glyph, broken table, or Figure 1 defect was found.

## Remaining external evidence gaps

The scope repair intentionally leaves the following open rather than papering them over:

1. E03 lacks the application contract needed to establish the conditional FIT/CAL sampling and UCB implementation assumptions for its historical numbers.
2. No natural temporal/geographic subgroup-mixture shift validation or fallback/abstention cost measurement is available.
3. No uniform statement is supplied for changed CAL data or data-derived selectors; the paper deliberately makes no such claim without a new proof and evidence.
