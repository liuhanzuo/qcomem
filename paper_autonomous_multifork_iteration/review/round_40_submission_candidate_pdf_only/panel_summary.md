# Round 40 PDF-only panel summary

## Frozen review identity

- Source: `main_r40_submission_candidate.tex`
- Source SHA-256: `166dff9f56da4449a53857c575fbf9f62466c7bd1e84b7c05e59301ffe346c10`
- PDF: `build/r40_submission_candidate_v1/main_r40_submission_candidate.pdf`
- PDF SHA-256: `0906080e3d16c0f8ee071f5d3aa2f6d4d541e7f7d8a2cc3efe9e67c0d0916d5b`
- PDF pages: 29; conclusion and References both begin/occur on page 9, within the
  nine-page main-text limit
- Rubric SHA-256: `df368bb0b31b60a75f81d155a6b01962865aedb2b2984443f3ba6cd8c153d874`
- Mode: exactly three fresh isolated PDF-only reviewers; no repository access,
  no prior-review access, no network, and no scored meta-reviewer

The first attempted third review is invalid because its rubric path did not
resolve. It is excluded. Reviewer 3 is a fresh replacement that verified the
canonical rubric hash before reviewing the complete PDF.

## Aggregate

| Reviewer | Overall | Confidence | Soundness | Presentation | Contribution |
|---|---:|---:|---:|---:|---:|
| 1 | 6 | 4 | 3 | 3 | 3 |
| 2 | 6 | 4 | 3 | 3 | 2 |
| 3 | 6 | 3 | 3 | 3 | 2 |
| **Median** | **6** | **4** | **3** | **3** | **2** |

Minimum overall is 6, mean overall is 6.0, and all three reviewers place the
current evidence ceiling at 6. The panel is marginally above threshold for the
explicitly narrow, honest-process, fixed-stack result; it does not support a
runtime-independent, production-serving, or security-attestation claim.

## Consensus

The strongest contribution is the historical alias result: the defective path
keeps tokens, FP32 logits, request GDN state, and logical KV exact in 8/8 cells
while corrupting persistent base storage; the repair is storage-clean in 8/8.
ForkAudit's demonstrated increment is earlier phase/owner/layer/family
localization, not exclusive detection, because a conventional persistent-base
invariant also catches this defect.

Three material evidence gaps unanimously block a score of 8:

1. A source-distinct live-object binding witness and slot-swap, stale-handle, or
   relabel challenges are needed to reduce the live-binding TCB.
2. A frozen strong conventional suite should be compared head-to-head on blind
   faults, including unique detection, first failure, localization, false
   positives, runtime cost, and maintenance cost.
3. A native continuous/ragged batching or in-flight cancellation setting needs
   matched H20 audit-on/off latency, memory, and throughput measurements.

The panel also recommends naming attention compiled-artifact provenance and GDN
eager-route provenance separately, making the post-execution run-ID correction
more visible, explaining numerical tolerance/sampling, and shortening the path
appendix. These are nonblocking and do not justify mutating the already reviewed
PDF before material evidence changes.

This is an internal automated review simulation, not an ICLR acceptance
prediction or an external reviewer decision.
