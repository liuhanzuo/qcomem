# Reviewer-safe artifact boundary

The Round-1 snapshot includes the current buildable manuscript, current hash-bound claim audit, immutable result/verifier files that contain no author identity, and the visual-provenance record.

It deliberately excludes historical checker logs that embed an absolute development-machine path containing an author account name. It also excludes `remote_snapshot/results/universal_cap_r500_result.json`, whose otherwise scientific JSON contains one such absolute `source` path. These exclusions are for double-blind safety, not because the files failed a scientific check. The full frozen historical checker reports `426 PASS / 0 FAIL / 3 external-provenance` on the unredacted local archive; the build record explicitly limits that result to the frozen historical source and artifacts.

No excluded object is used to claim ordered-online validity, delivered-answer correctness, token/latency saving, or cancellation cost. The headline OMR, shard, OpenR1, RLVE, drift, and margin result JSONs and their reviewer-safe scripts remain included. A final public artifact release can publish the original objects after de-anonymization or replace absolute paths with a documented content-preserving redaction and new manifest.
