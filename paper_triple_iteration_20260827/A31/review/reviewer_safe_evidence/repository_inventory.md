# A31 repository inventory

## Located

- The submitted manuscript PDF and a hash manifest for its TeX source,
  bibliography, and compiled PDF.
- Reviewer-safe claim--evidence, experiment-registry, and method-provenance
  records.
- Partial source-reported probe cells printed in the manuscript.

## Not located

- Every `results/*.json` and analysis script named in the manuscript's artifact
  table, including the headline period-profile and cross-arm bootstrap records.
- The printed evidence ledgers, experiment registry, build manifest, event logs,
  and private checkpoint.
- Complete L24/L48 candidate-separation vectors, complete candidate-level
  rank-sum scores, the rank-sum implementation, and replicate indicators.
- The event-schema/time-coordinate contract, missing/terminal handling code,
  exact question-pairing manifest, timestamped registration records, and
  registration diffs.
- The data and generating program for the pre-existing data-derived figure, and
  the model/prompt provenance record for the inherited qualitative figure.

## Consequence

The PDF uses opaque paper-local evidence identifiers and does not expose raw
workspace paths, timestamps, or digest/commit prefixes. The supplied package cannot inspect
or recompute the numerical targets, registration chronology, probe argmaxes, or
rank-sum candidate ordering. All empirical numbers remain source-reported and
locally unverified; no analysis was rerun.
