# Round 29 true-concurrency readiness v2

Status: dependency-closed, hash-consistent, and ready for one-H20 execution.
No GPU execution occurred under v1.

The immutable scientific design remains
`5c9fc301ec63e2702d097b9d9be9c68758164c653c6c7b53fedad290428a9a96`.
The pre-execution amendment removes an unbound dependency on the later
Round-23 scheduler helper by inlining its small pointer-free lease,
reservation, and zero-scrub checks. It does not change a model call, token,
geometry, concurrency treatment, lifecycle schedule, oracle, success rule, or
claim boundary. The v1 ledger remains at `source-code.sha256`; only
`source-code-v2.sha256` is executable.

Post-amendment readiness checks:

- `sha256sum -c source-code-v2.sha256` passes all seven entries, including
  the retained v1 ledger and amendment;
- both Python sources compile;
- the launcher passes `bash -n`;
- all six focused tests pass;
- the CPU/mock lifecycle and overlap gate passes;
- the runner imports no Round-23 scheduler module or scheduler-contract file;
- the selected 7620... lifecycle source ledger contains every remaining
  upstream implementation dependency.

Resource and isolation remain unchanged: one exclusive H20 (GPU4 is suitable),
one process, no port, no collective, approximately eight minutes, read-only
shared model/data assets, and private output/compiler-cache paths.
