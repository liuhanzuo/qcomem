# A23 Structure Audit

## Paper identity

This paper compares three in-decoding confidence readouts on one frozen LLaDA-8B-Instruct × GSM8K event stream and separates rank discrimination from incremental selective-risk utility.

## Claim thread and closure

- Problem → several plausible token-confidence summaries need not induce the same decode ranking or risk-at-coverage behavior.
- Method → aligned commit-versus-all AUROC and risk contrasts with paired, decode-clustered inference.
- Evidence → exploratory aligned AUROC difference +0.0278; small risk differences with zero-crossing intervals at coverages 0.2, 0.5, and 0.8; lattice and reliability-bound cautions.
- Boundary → rank separation on this frozen stream does not establish practical equivalence, general null utility, causality, or reproducible decoding.

## High-impact findings

- Critical provenance limitation: the exact PDF source was not preserved. The active same-lineage source is used only as a reconstruction substrate and cannot be represented as the exact source.
- Major: the abstract repeats the rank-separation statement and contains revision-history language (“review-driven”). Remove the history while keeping the post-alignment exploratory status.
- Major: the conclusion opens with a rebuttal-style sentence about “another wording pass”; replace it with an evidence-centered replication requirement.
- Minor: streamline the contribution list so the rank-versus-risk distinction is immediately legible.

## Priority edits

1. Rewrite the abstract without revision-history leakage.
2. Tighten the contribution paragraph.
3. Reframe the conclusion around the supported result and decisive replication.
4. Preserve every estimator, coverage, interval, lattice statement, and recoverability limitation.
