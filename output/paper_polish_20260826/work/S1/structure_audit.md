# S1 Structure Audit

## Paper identity

This paper reports an archived descriptive observation: an identical 625M-token recovery recipe compresses the Avg-9 range across three fixed 2:4 supports from 2.675 pp to 0.118 pp, while missing paired predictions, matched SparseForge controls, component ablations, corpus provenance, and a deployable folded export prevent stronger statistical or mechanistic claims.

## Claim thread

`structured-support question -> three fixed supports -> identical recovery -> compressed endpoint spread -> marginal sampling-scale comparison -> missing joint uncertainty -> pipeline/export/corpus boundaries`

The evidence boundary is unusually explicit, but it is repeated many times. More importantly, the title and abstract call the observation a “registered negative result,” whereas the Results section correctly says it is not a statistical negative result or equivalence claim. The title-level wording therefore exceeds the manuscript's own evidence.

## Priority repairs

- Critical: remove “negative result” from the title and abstract; describe the 0.118 pp endpoint spread without implying an identified joint resolution limit.
- Major: compress the abstract and introduction while preserving the single-checkpoint 0.52 pp scale and the missing-paired-data qualification.
- Major: reorganize the one-paragraph Limitations section into distinct evidence, mechanism, export/provenance, and external-validity limits.
- Major: keep SparseForge components explicitly as pipeline background, not causal contributions.
- Minor: narrow unsupported literature-completeness language (“to our knowledge none”) to the materials actually established by the manuscript.
- Minor: make the conclusion state exactly what is unidentified: joint uncertainty and residual support distinguishability.

## Promise-evidence closure

- Range compression 2.675 -> 0.118 pp after identical recovery: supported as an archived point-estimate observation.
- “Below 0.52 pp” -> supported only as comparison to a marginal per-checkpoint binomial SE; not a pairwise/joint threshold.
- Recovery makes supports equivalent -> unsupported and explicitly rejected.
- SparseForge mechanism advantage -> unsupported because components were not isolated and no matched 625M SparseForge endpoint exists.
- Deployable exact-2:4 result -> unsupported because SLoRB fold/reprojection/export was not executed.
- Archived 5B corpus interpretation -> unavailable because the checkpoint and manifest are unreachable.

## Main-text constraint

Preserve the nine-page ICLR main-text boundary before the bibliography. Editorial compression should not move appendix material into the main paper.
