# Round 29 bounded two-stream lifecycle experiment

This package records a prospectively frozen experiment on one H20-3e.  It
tests whether two request-local full-model steps can preserve exact output,
private-page ownership, and cancel--zero--scrub--same-slot replacement
semantics while their execution intervals overlap on two distinct CUDA
streams.

The scientific design remained byte-identical across the retained source
history.  The v1 source freeze was superseded before GPU execution because its
launcher digest did not match the source ledger.  The first v2 GPU attempt
completed model computation and wrote both logit sidecars, but failed before
committing a formal result because PyTorch's private `_CUuuid` object was not
JSON serializable.  Its disclosed traceback and sidecar hashes are retained in
`pre-second-execution-amendment-v3.json`; no overlap or lifecycle outcome was
read, and that attempt is ineligible as scientific evidence.  The equality of
the two disclosed sidecar hashes did expose the logit-equality sub-result, so
the second attempt is not an outcome-blind replication.  The amendment made
only JSON-safety and regression-test changes.  The v3 ledger is the sole
authority for the fresh second attempt.

The valid formal run is under `formal_run_20260825b/`.  Both concurrent phases
used two host workers, two distinct CUDA streams, and a barrier.  Their measured
execution-interval overlaps were 260.802 ms before cancellation and 95.439 ms
after zeroing, scrubbing, and reusing slot 1 at the next epoch.  All four
full-vocabulary logit samples (993,280 cross-arm FP32 scalar pairs) and four
generated tokens matched the serialized arm exactly (maximum absolute error
0); the four lifecycle-indexed samples contain three unique logit digests.
Final active logical KV and GDN digests, ownership receipts, source-document
immutability, and the separate lifecycle replay all passed.  The local CPU
replay reproduced the shipped replay byte-for-byte.  It recomputes logits and
interval overlap with NumPy but shares the archived lifecycle-event helper and
trusts archived ownership/scrub flags, so it is not an independent recapture.

Key raw SHA-256 values:

- scientific design: `5c9fc301ec63e2702d097b9d9be9c68758164c653c6c7b53fedad290428a9a96`
- v3 amendment: `7d03eca752b3a4163168022cac5e5a044a7e03c67d1815899285c47df2578904`
- v3 runner: `401a19314ea3efd24731ed4f2fea9515e961f4532359af566fafe38148c98302`
- formal result: `f110e994536e0d0637109e1b1a76b6d6140626be534f21fd118fb3ba63dce970`
- independent replay: `6f02c84b832e2b2c4e6c93b9cd0f93c50d9f14e356c655a0c884c791e10a7032`
- raw/replay ledger: `d8772a6a972a3c178b3b7b59196b44722772e550a71515562e366fcbdd839789`
- remote/local transfer archive: `2e63beb90b3e98af4d2a4eb5d7181fea4e9dbe12a44858b7ee86dd8c5c86eca8`

Validation from the repository root:

```bash
cd paper_autonomous_multifork_iteration/evidence/r29_true_concurrency/formal_run_20260825b
shasum -a 256 -c receipts/raw-and-replay.sha256
cd ../../../..
PYTHONPATH=gpu python3 gpu/r29_replay_true_concurrent_lifecycle.py \
  --design-preregistration paper_autonomous_multifork_iteration/evidence/r29_true_concurrency/design_preregistration.json \
  --expected-design-sha256 5c9fc301ec63e2702d097b9d9be9c68758164c653c6c7b53fedad290428a9a96 \
  --formal-result paper_autonomous_multifork_iteration/evidence/r29_true_concurrency/formal_run_20260825b/raw/formal-result.json \
  --expected-formal-result-sha256 f110e994536e0d0637109e1b1a76b6d6140626be534f21fd118fb3ba63dce970 \
  --artifact-dir paper_autonomous_multifork_iteration/evidence/r29_true_concurrency/formal_run_20260825b/raw/sidecars \
  --output /tmp/r29-true-concurrency-replay.json
cmp /tmp/r29-true-concurrency-replay.json \
  paper_autonomous_multifork_iteration/evidence/r29_true_concurrency/formal_run_20260825b/replay/independent-replay.json
```

Claim boundary: this establishes that two request-local full-model calls were
simultaneously in flight in CUDA-event-bracketed intervals on distinct streams
for the frozen Qwen3.5/ForkAudit vLLM-Q16 H20 stack, before and after a
quiescent tested lifecycle transition.  The interval includes stream queuing,
resource waits, and host-enqueue gaps.  It does not establish simultaneous
execution of individual CUDA kernels, native vLLM continuous batching,
production-server correctness, throughput or latency gains, cancellation
during concurrent execution, arbitrary in-flight cancellation safety, or
cross-model/runtime/hardware generality.
