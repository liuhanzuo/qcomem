# R40 protocol amendment: Attempt B

Status: frozen after Attempt A failed and before the first Attempt B timing
row.

## Why an amendment is required

Attempt A is retained verbatim under `raw/`.  All five supporting components
passed in every repetition.  The primary RR2 component stopped before its
scientific replay because the existing source-package directory contained 13
unmanifested `executed_source/gpu/__pycache__/*.cpython-312.pyc` files.  The
package verifier correctly reported file-set drift.  This is a local working-
tree contamination condition, not an RR2 verdict.

No file is removed from or modified in the existing RR2 evidence directory.
Attempt A is not discarded, renamed, or overwritten.

## Attempt B preparation

Before timing, the Attempt B harness creates a new read-only-in-practice view
under this R40 directory.  It copies exactly the 628 regular, non-symlink files
listed by the immutable RR2 `MANIFEST.json`, plus `MANIFEST.json` and
`MANIFEST.sha256`.  For every listed member, source size/SHA-256 and copied
size/SHA-256 must match the manifest.  Paths must be sorted, unique, relative,
and free of `..`.  The manifest sidecar must bind the manifest, and the copied
package's own verifier must pass before timing starts.

The copy and its validation are one-time input preparation, analogous to the
predeclared safe unpack of the dual-producer archive.  Their wall time is
recorded but excluded from replay timing.  The preparation receipt records the
declared byte/file totals and the unmanifested source paths that necessitated
the view.  No cache flush is performed and no cold-cache claim is made.

## Frozen Attempt B measurement

The profiles, component order, three-repetition count, acceptance criteria,
environment controls, wall-time boundary, RSS interpretation, exclusions, and
unmeasured-cost list are unchanged from `PROTOCOL.md`.  The sole operational
change is that `minimal_core` invokes the same manifest-bound
`replay/run_replay.sh` from the clean manifest-only view.  The script still
verifies all package bytes inside every timed replay.

Attempt B writes only to `raw_attempt_b/` and refuses replacement.  Any failed
row remains retained.  Local Apple CPU replay time must not be described as
H20 capture overhead.

