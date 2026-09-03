# Post-hoc independent read-only audit

Date: 2026-08-27

This note was added after the frozen measurement and is not part of its
premeasurement source ledger. It narrows interpretation and does not alter any
timing row, input, or aggregate.

## Verdict

PASS for a single-host, local, fixed-order CPU replay and logical-storage
measurement with cache state uncontrolled. FAIL for H20 capture overhead, GPU
perturbation, cold-start CI latency, online inference cost, or a matched
uninstrumented delta.

The auditor reverified 18/18 Attempt-B component rows, 3/3 serial profiles,
16/16 tests, all 303 non-bulk terminal-manifest entries, both bulk prepared
inputs, and every reported aggregate.

Attempt B is scientifically acceptable: its amendment, tests, and checksums
precede the first Attempt-B timing row; the canonical view contains exactly the
628 immutable manifest payloads, each source/copy pair matches in size and
SHA-256, and the only excluded files are the thirteen unmanifested \`.pyc\`
files that caused Attempt A to fail before scientific replay.

## Required interpretation boundary

- The manifest-only copy, full hash pass, and one verifier run preceded timing,
  and Attempt A had already visited supporting roots. No cache flush occurred;
  these are not cold-cache measurements.
- Profiles always ran in the fixed order primary then five supporting
  verifiers, for three consecutive repetitions without randomization,
  counterbalancing, or cooldown. Report medians with full ranges and do not
  compare individual component performance.
- The 5.598-second paired increment includes fixed-order harness and inter-step
  gaps; it is not pure verifier compute time.
- Darwin \`time -l\` \`ru_maxrss\` is bytes for the timed command and its
  descendants. It is neither a simultaneous whole-profile process-tree sample
  nor GPU memory.
- Extended logical bytes sum disjoint local evidence roots and can contain an
  archive plus its extracted copy. They are not a compressed upload size.

## Narrow paper-safe statement

> On one Apple Mac16,8 CPU host, three consecutive fixed-order replays of the
> verified manifest-only package, without a cache flush, took a median 100.146
> s (range 99.248--110.526 s) for the core RR2 replay and 105.717 s
> (104.846--119.566 s) for the serial profile that additionally ran five
> locally complete supporting verifiers. The paired within-profile increment
> was 5.598 s (5.571--9.039 s). The canonical core distribution contains 630
> files and 892,284,156 logical bytes; 536 raw traces occupy 888,785,811 bytes,
> or 99.62% of manifest-payload bytes. These measurements characterize local
> CPU replay and storage only, not H20 capture cost, GPU perturbation,
> cold-start CI latency, or an uninstrumented-baseline delta.

