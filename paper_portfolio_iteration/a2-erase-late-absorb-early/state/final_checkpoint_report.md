# Final checkpoint report

Status: **stopped on the current-evidence ICLR rescue path.** The artifact is
retained as an honest scoped exposition, not as an ICLR-submission candidate.

## Selected artifact

- Checkpoint: `revision_02` (the best verified artifact, not a passing score
  checkpoint).
- PDF: `manuscript/paper.pdf`, SHA-256
  `5a3f1b781e38e790660b3a64356c45d3aed27bcd699b414a7735302369cad11e`.
- Source: `manuscript/paper.tex`, SHA-256
  `b3a06e3b40a2262051df308553f6c068b41e95530fe0dff3c32238351cedb07a`.
- Format: 5 letter-sized pages. The PDF equals both controlled clean-build
  outputs and the Round-2 frozen submission PDF.
- Review binding: Round-2 snapshot SHA-256
  `65acc8505a02b1a4002de506890647928ee23c80b1ea14f35696a82b0cd1b56f`.

## Final evidence check

- All 27 Round-2 manifest file hashes match, and its documented
  `snapshot_path` ordering reproduces the frozen snapshot root.
- All 20 Round-1 manifest file hashes match. Its legacy manifest does not
  state an aggregate-root algorithm, so this report makes no independent
  aggregate-root reproduction claim for Round 1.
- All eight declared revision-02 build inputs match the build record. Both
  archived clean builds match the selected PDF hash; final logs have no
  undefined-citation/reference, overfull-box, or TeX-error match.
- Frozen reviews were not edited.

## Why iteration stops

Round 2 is `[4, 2, 2, 2, 4]` (median/lower quartile `2`), with meta-score and
current-evidence ceiling `2`; Contribution median is `1`. The later novelty
feasibility audit agrees: existing T1--T3 offer only a trivial time-ordered
affine-recurrence unification, not a demonstrated research contribution.
Additional wording, scope cleanup, or provenance hygiene cannot change that
evidence ceiling.

## Conditions to resume

Resume theory only for a genuinely new U3 matched two-trajectory,
time-ordered data-block perturbation theorem with an explicit estimand,
noncommutation treatment (or a meaningful proved restriction), a falsifier,
and a primary-source theorem matrix showing material novelty. Alternatively,
resume practice only after a new claim-linked reproducible matched-path study
is frozen and executed with same-version code, data/splits, configuration,
seeds, comparators, outputs, and a decision-time information audit. Generic
tail contraction, restating T1--T3, historical/remote-only material, or prose
revision is not a recovery condition.
