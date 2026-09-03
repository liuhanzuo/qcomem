# A31 repository inventory

## Located

- Frozen baseline PDF, TeX, bibliography, compiled auxiliaries, and two PNG
  figure assets.
- Five round-0 and five round-1 review JSON files, their panel summaries and
  meta-reviews, and author-side revision notes. Frozen review artifacts were read
  but not edited.
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
- The exact candidate lists and gate outputs underlying the source-reported
  `10/25` wrong-candidate audit. Its five-per-arm denominator cannot be mapped to
  the displayed four-candidate canvas-256 selector grids.
- The ordered per-block allocation vector for the separate L32/NFE32 content
  bed; its total NFE is reported, but the vector is not supplied.
- The data and generating program for the pre-existing data-derived figure, and
  the model/prompt provenance record for the inherited qualitative figure.

## Consequence

The manuscript uses one recorder-relative evidence-audit identity and opaque
paper-local evidence identifiers; it does not expose raw workspace paths,
timestamps, or digest/commit prefixes. This audit cannot inspect or recompute the
numerical targets, registration chronology, probe argmaxes, rank-sum candidate
ordering, or original-gate denominator. Probe argmaxes are excluded from the
headline claim and per-arm gate fractions are omitted rather than guessed. All
empirical numbers remain source-reported and locally unverified; no analysis was
rerun.
