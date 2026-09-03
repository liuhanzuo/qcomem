# R40 local CI-cost accounting protocol

Status: frozen before the first R40 timing row.

## Question and boundary

This package measures the cost of replaying already-captured evidence on the
local CPU host.  It does not measure H20 capture overhead, online inference
latency, GPU perturbation, network/download time, engineering effort, or the
increment relative to an otherwise identical uninstrumented H20 execution.
Those quantities remain `unmeasured` unless a matched measurement already
exists; no value is inferred from file timestamps or from local CPU replay.

The audited packages are the current primary RR2 package, the R39 independent
preproducer census, the R39 dual-producer repeat, the Falcon-H1 v2 result, the
R39 PDF-only blind-fault package, the R33 designer--executor fault package, the
R35 historical-alias package, and the R30 expanded numerical replay.

## Profiles

`minimal_core` is exactly one invocation of
`evidence/round_04_rr2_package/replay/run_replay.sh`.  It checks the complete
package manifest, 536 raw artifacts, the 96-cell factorial, storage timelines,
eight FP32 attention rows, nine matched fault pairs, and storage-witness tests.

`extended_supporting` executes `minimal_core` followed serially by every
supporting replay that is both locally complete and hash-bound at freeze time:

1. R39 preproducer census audit plus its three deep-copy controls;
2. R39 dual-producer zero-tolerance verifier on a safe, pre-unpacked formal
   archive;
3. R33 five-pair designer--executor detached aggregation, which replays every
   pair from the local sidecars;
4. R35 full local evidence verifier, including fresh extraction and detached
   re-execution from its frozen package; and
5. R30 candidate-import-free NumPy replay of 20 attention and 24 GDN rows plus
   44 seeded controls.

The profile is repeated three times.  Components run serially and never
overlap.  Profile wall time is directly observed from before the first
component until after the last component.  Component wall time is measured by
the harness monotonic clock around `/usr/bin/time -l`; the raw `time` report is
retained.  On Darwin, the raw `maximum resident set size` integer is recorded
as bytes.  The profile-level memory value is only the maximum observed
component peak RSS, not a separately sampled whole-profile process tree.

No filesystem cache is flushed and no cold-cache claim is made.  The three
consecutive rows are reported individually, with median and range.  Package
download and the one-time safe unpack of the dual-producer archive occur
outside replay timing and are not included.

## Predeclared exclusions and blockers

- Falcon-H1 v2 is inventory-audited but excluded from timing because the
  formal result binds a v2 replay source with SHA-256
  `aa62966c00ffeeb1f7c9cf9f619cd082987235a66c3353a08485ed7afe79860f`,
  while that source is not present locally.  The only local Falcon replay
  source is the older v1 file with a different hash.  A formal replay receipt
  is not substituted for executable source.
- The R39 PDF-only blind-fault metadata package is inventory-audited but
  excluded from full replay timing because its 352 full-vocabulary FP32
  sidecars are not present locally.  Its metadata aggregator is not treated as
  equivalent to detached pair replay.
- The frozen H20 log/stage timestamps are audited only as provenance metadata.
  Archive-normalized epoch timestamps and filesystem mtimes are never
  converted into capture duration.
- Existing R29 local-replay and live request-step measurements are retained as
  separate prior cohorts.  They are not pooled with R40 and do not provide an
  uninstrumented baseline for the current extended profile.

## Execution rules

- No QS command, SSH command, CUDA command, or GPU resource access is allowed.
- Existing evidence is read-only.  Every R40 output is created under this new
  directory; source packages and manuscript/PDF files are not modified.
- `PYTHONDONTWRITEBYTECODE=1`, `PYTHONHASHSEED=0`, and an empty
  `CUDA_VISIBLE_DEVICES` are supplied to replay subprocesses.
- Every command, exit code, UTC start/end timestamp, monotonic wall time, raw
  resource report, stdout, stderr, and result digest is retained.
- A failed component remains a recorded row and invalidates that profile
  repetition; it is never silently rerun or discarded.

## Reported quantities

Measured: local CPU replay wall time, component peak RSS where Darwin
`/usr/bin/time -l` reports it, exact local package/archive/trace logical bytes,
file counts, executable-entrypoint hashes, and log-mtime ranges.

Unmeasured: H20 capture wall time for the current packages, GPU slowdown or
perturbation, current-package incremental overhead versus a matched
uninstrumented baseline, cold download/extraction cost, engineering/adoption
effort, service latency, throughput, capacity, and any cross-host cost claim.

