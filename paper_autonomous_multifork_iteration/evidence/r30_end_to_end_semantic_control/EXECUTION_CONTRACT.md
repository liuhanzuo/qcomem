# R30 independent end-to-end semantic control

## Purpose and claim boundary

This is a post-review control on the already fixed stack: Qwen3.5-35B-A3B at
revision `59d61f3ce65a6d9863b86d2e96597125219dc754`, Transformers 5.14.1,
PyTorch 2.11.0+cu129, vLLM 0.26.0, and one H20-3e. It is not a runtime,
hardware, model, scheduler, or workload portability experiment.

The control compares a source-distinct standard-Transformers full-model
recompute, which starts from frozen raw token IDs and model weights in a fresh
process, with the audited four-cell materialized/shared KV/GDN path. Exact
multi-step greedy-token agreement is the primary semantic gate. Full-vocabulary
logit divergence is a required secondary measurement and has no tuned pass
threshold.

## Frozen design

- Input source: the exact 64-book PG-19 train artifact already bound by SHA-256
  in the paper evidence; validation/test and LongBench are forbidden.
- Selection: eight candidate windows are built without model execution using
  seed `2026082504`, 4,095 document tokens, 32 query tokens, stride 271, and
  eight candidate starts per book. A SHA-256 score over input coordinates
  chooses the first two candidates that are absent from the frozen exclusion
  ledger. Each selected document supplies two distinct non-overlapping raw
  32-token query chunks.
- Semantic horizon: four greedy output tokens per request, two requests per
  document, two documents.
- Candidate arms: the complete 2x2 ownership factorial:
  `KV={fresh full copy, shared document reuse}` x
  `GDN={materialized base, borrowed immutable base}`.
- Reference: `r30_e2e_reference.py` may import PyTorch and Transformers but no
  candidate, ForkAudit, CoMem, vLLM, cache, trace, or replay module. It receives
  only the frozen input manifest and model directory. It calls full-model
  `use_cache=False` recomputation from the whole raw-ID history at every step.
- Candidate: a separate process runs the audited Q16 block-pool path. The
  frozen single-token GDN ownership repair must be applied before every
  single-token cached transition. The clean repair regression and its detached
  replay must pass before this experiment is allowed to execute.
- Replay: a third process imports NumPy and the Python standard library only.
  It reopens all FP32 sidecars, verifies their digests and shapes, derives the
  primary token gate, recomputes full-vocabulary metrics, and replays the
  serialized ownership relations.

## Primary and secondary outcomes

The primary gate passes only if every one of the four candidate arms exactly
matches the reference greedy token trajectory for all 16 registered decisions
(2 documents x 2 requests x 4 steps), and all clean ownership/lifecycle gates
pass. A mismatch is a valid negative scientific result, not an infrastructure
failure.

For every arm and step, a second candidate execution is teacher-forced with the
reference token history. This makes all full-vocabulary comparisons
history-matched even if a greedy trajectory diverges. Replay must report, for
all 64 arm-step comparisons, max absolute error, mean absolute error, relative
L2 error, cosine distance, and top-1 agreement. No one of these continuous
metrics is a preregistered acceptance threshold; none may be omitted because
it looks unfavorable.

## Ownership gates

For every arm, both requests are simultaneously live. Replay requires:

1. the source document payload and persistent GDN state are byte-identical
   before and after generation;
2. fresh KV arenas are storage-disjoint from the source and from one another;
3. shared KV requests use the source arena, have pairwise-disjoint private
   reservations, and replace the 127-token partial document-tail block with a
   request-private block before append;
4. materialized GDN bases are private at setup;
5. borrowed GDN bases alias the immutable source at setup but all mutable GDN
   state is source- and peer-disjoint after transition;
6. every full-attention layer records exactly four calls with append schedule
   `[32, 1, 1, 1]` and the same pinned vLLM unified-attention callable; and
7. the single-token repair receipt is schema-valid, ownership-only, and
   idempotent on repeated invocation.

These are trace-relative checks under honest capture. They do not establish
capture honesty, compiled-kernel identity, unseen-fault rates, native dynamic
batching, or production safety.

## Resource and stopping rule

Only physical GPU 4 of QS Trial 1899487 / pod
`qs-249885-1899487-ai-1443683-master-0` is authorized. The launcher must expose
exactly `CUDA_VISIBLE_DEVICES=4`, verify the H20 UUID, and reject an occupied
device. All outputs go to a new R30 directory and are retained.

The reference, candidate greedy result, candidate teacher-forced result,
ownership failure, token mismatch, or counterintuitive logit divergence is a
valid preserved outcome if all static, environment, sidecar, and replay gates
complete. OOM, timeout, missing artifact, source drift, malformed record, or
failed clean alias-repair prerequisite is operationally invalid and must not be
presented as scientific evidence. No input, seed, horizon, arm, or metric may
change after model output is observed.
