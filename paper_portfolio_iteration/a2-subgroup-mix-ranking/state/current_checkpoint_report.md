# Current checkpoint report

Date: 2026-08-22

## Selected artifact

Revision_04 is the best verified artifact on integrity and build grounds. It is not the
best scored review round, a submission-ready checkpoint, or evidence of a score increase.

- Source SHA-256: `269e95c75e5c03b02a277617435f84126a0e7674f811fa2a04c3e114776ad0c8`
- PDF SHA-256: `e7a47b1f0f02e40fb55dbc79fb98a2b1cb0f3fa2eb3894a2fe1c06047b804fc9`
- Round-4 snapshot SHA-256: `4fb48a187264fe400fcfda23f8024741f45ea269c740530b4964705c86d08e2c`

The independent change verifier returned `partially_resolved`: it verified the fixed-statistic
M9/M10 boundary, M6 scope, non-regression of P3, deterministic clean build, and visual quality;
it retained E03 application evidence as unresolved.

## Recorded review trajectory

| Round | Status | Ratings / verdict | Meta / ceiling |
| --- | --- | --- | --- |
| R1 | Full five-reviewer panel | `[6,4,4,4,4]`, median 4 | meta 4 |
| R2 | Full five-reviewer panel | `[4,4,4,4,4]`, median 4 | meta 4; evidence ceiling 4 |
| R3 | One technical reviewer only | rating 4 | no panel median; no meta-review |
| R4 | Targeted change verification | `partially_resolved` | not a scored review |

R3 snapshot SHA-256 is
`8e0822d2b79b3e0e8cd5702d40248003900fb554f228bfde1d25f431cebd18f2`.

No scored full panel exists after R2. Therefore R3/R4 do not demonstrate score improvement, and
the current evidence ceiling remains 4.

## Conditions for reassessment

1. Provide a reviewer-safe E03 application contract: immutable split identity, sampling and
   collection provenance, candidate/FIT isolation, paired sufficient statistics or raw replay
   inputs, executed UCB configuration, and environment; perform a clean replay where necessary.
2. If retaining deployment relevance, run a natural time/geographic subgroup-mixture shift study
   with a predeclared within-group invariance audit.
3. If retaining operational abstention claims, precommit and measure fallback/abstention cost or
   utility, or conduct a declared sensitivity analysis.

A future score reassessment requires a fresh blind full panel and meta-review. It must treat this
report as author state, not as reviewer evidence.
