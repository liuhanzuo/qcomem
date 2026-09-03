# Round 29 true-concurrency readiness

Status: frozen and ready for one-H20 execution. No GPU result exists at freeze
time.

The prior scheduler experiment is explicitly not reused as concurrency
evidence: it used one CUDA stream and round-major request-step interleaving.
This experiment instead launches two complete request-local Qwen3.5 model
steps from two barrier-released host workers onto two distinct non-default CUDA
streams. CUDA events bind each stream interval to a common origin, and both
pre-cancel and post-reclaim phases must have strictly positive interval
overlap. This demonstrates overlapping stream intervals, not simultaneous
individual kernels.

The treatment is paired against a serialized same-input arm. Four
full-vocabulary logit vectors are retained per arm in raw FP32 sidecars:
survivor round 0, cancelled request round 0, survivor round 1, and replacement
round 0. The primary success rule requires bitwise-equal logits and generated
tokens, equal final logical KV and GDN digests, immutable source document
pages, complete vLLM-Q16 operator ledgers, ownership receipts before and after
replacement, exact zero-scrub/reuse, and independent lifecycle replay.

Focused local readiness checks completed before submission:

- both Python sources compile;
- the shell launcher passes `bash -n`;
- six focused tests pass, including overlap arithmetic, lifecycle replay,
  preregistered claim boundaries, FP32 sidecar round-trip, and tamper failure;
- the CPU/mock gate replays cancel and same-slot replacement with epoch vector
  `[0, 1]`;
- all new source and design files are frozen in `source-code.sha256`.

Resource request: one exclusive H20, one process, no listening port, and no
multi-GPU collectives. Estimated wall time is eight minutes. The model and
PG19 assets are read-only. `RUN_DIR`, Triton cache, and TorchInductor cache must
be unique to this experiment, so it may safely share the eight-card node with
unrelated jobs pinned to other GPU UUIDs.

The claim boundary excludes native vLLM-engine continuous batching,
production-server correctness, kernel-level simultaneous execution,
throughput/latency/capacity, in-flight reclamation, and cross-stack
generality.
