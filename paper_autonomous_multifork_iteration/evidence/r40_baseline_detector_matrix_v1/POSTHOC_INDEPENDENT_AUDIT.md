# Post-hoc independent read-only audit

Date: 2026-08-27

This note was added after the frozen v1 retrospective analysis and is
intentionally not part of its pre-analysis source/input ledger. It narrows
interpretation; it does not change any frozen row or aggregate.

## Verdict

PASS as a transparent retrospective fixed-case matrix. FAIL for any claim of
an independent same-execution baseline head-to-head, defect recall, precision,
or a population detection rate.

The auditor reproduced the deterministic analysis, 3/3 unit tests, 5/5 product
hash checks, all 65 input hashes, and the reported 52 rows: 22
fault/defect-coordinate rows and 30 clean-control rows. The frozen full
conventional suite flags 21/22 rows, ForkAudit flags 22/22, and no evaluated
detector fires on the 30 clean rows.

## Required interpretation boundary

- The M8 fault execution contains zero evaluated conventional detector
  decisions. Its result establishes callable-provenance **coverage absent from
  the frozen conventional rule set**, not a conventional detector that ran and
  missed.
- The full 21/22 count includes seven fault rows with fourteen decisions
  projected from target-gate-suppression events. They are counterfactual
  projections, not independent detector executions.
- The 11/22 subset should be called the **direct-observer subset**. It is not
  eleven independently implemented baselines or captures.
- Eight R35 rows are eight input coordinates of the one
  \`HISTORICAL_GDN_CONV_ALIAS\` mechanism, not eight independent defects.
- Detector definitions are hindsight-aware rather than outcome-blind.
- Zero catches among 30 controls is a fixed-case count over uneven detector
  applicability, not a precision or false-positive-rate estimate.

## Narrow paper-safe statement

> In a retrospective reanalysis of fixed archived cases, ForkAudit produced
> the expected typed rejection for nine primary mutants, five
> designer–executor faults, and eight input coordinates of one historical
> alias defect. A frozen conventional check suite flagged 21 of 22 rows only
> when archived-receipt recomputation and seven counterfactual fault-row
> projections were included. The remaining M8 row exposed
> callable-provenance coverage absent from that frozen suite, but no
> conventional detector was evaluable in its fault execution, so this is a
> coverage difference rather than an independently executed baseline miss.
> Across 30 matched clean-control rows, none of 114 evaluated conventional
> decisions fired.

