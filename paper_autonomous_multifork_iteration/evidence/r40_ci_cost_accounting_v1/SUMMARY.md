# R40 measured local replay and CI-storage cost

Attempt B passed all 18 component rows and all three serial profiles. All timings are local CPU replay measurements on one Apple host; they are not H20 capture overhead, GPU perturbation, or an uninstrumented-baseline delta.

| Profile | Repeats | Wall median (range), s | Peak RSS median (range), MiB | Audited local logical bytes |
|---|---:|---:|---:|---:|
| minimal core | 3 | 100.146 (99.248--110.526) | 2330.625 (2185.391--2331.844) | 892284156 |
| extended supporting | 3 | 105.717 (104.846--119.566) | 2330.625 (2185.391--2331.844) | 1329451516 |

The supporting profile added a measured median 5.598 s after the minimal replay. Its RSS value is the maximum separately observed component peak, not a sampled whole-profile process-tree peak.

## Artifact accounting

The clean primary distribution contains 628 manifest payload files plus two manifest controls: 630 files and 892284156 logical bytes (850.95 MiB). Its 536 raw trace artifacts occupy 888785811 bytes (847.61 MiB; 99.62% of manifest payload bytes).

Attempt A remains retained and invalid: all 15 supporting rows passed, but all three primary rows stopped before replay because the source tree had 13 unmanifested `.pyc` files. Attempt B copied and verified only manifest-listed bytes without modifying the source evidence; its one-time 5.666 s preparation was excluded from replay timing.

## Still unmeasured

Current-package H20 capture wall time, GPU slowdown/perturbation, a matched current uninstrumented baseline, cold download/extraction cost, and engineering/adoption effort remain unmeasured. Falcon-H1 v2 replay is blocked by its absent hash-bound v2 verifier source; the PDF-only blind-fault full replay is blocked by 352 absent FP32 sidecars. Filesystem and archive timestamps are retained only as provenance and never converted to duration.
