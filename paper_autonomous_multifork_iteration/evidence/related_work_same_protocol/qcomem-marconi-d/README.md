# Marconi policy-trace debug D

This directory records the first stable debug execution of RW-B4 on the
persistent 8xH20 node (QuickSilver Job 246593 / Trial 1871681).  The policy
simulator itself is CPU-only; the node was used as the common interactive
debug environment requested by the author.

## Frozen inputs

- official Marconi repository commit
  `08016617b1524e6bf6ac29b680641cc945bda7f0`;
- runner `gpu/run_related_work_marconi_trace.py`, SHA-256
  `64275f6e0bc299fa1ddac537fbbcb5abed03f07097fbae5b5670d555f7891cb4`;
- frozen LongBench data SHA-256
  `1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe`;
- the eight Qasper/2WikiMQA source-index 6--9 workloads and measured SGLang
  answers already registered by RW-B2;
- deterministic workload counts `(32,24,20,16,12,10,8,6)` and cache budgets
  5/10/20 decimal GB.

The official artifact models multi-turn sessions in which each new input
contains the previous input and output.  The trace therefore keeps the eight
real document/question prefixes and measured answers, then appends the fixed
tokenized suffix `Follow-up: repeat only the same short answer.` after each
turn.  It is a synthetic policy trace, not another LongBench quality run.

## Debug result and boundary

At 5 GB, token-hit rates were 2.07% (vLLM+), 90.78% (SGLang+), and 93.86%
(Marconi).  At 10 and 20 GB, SGLang+ and Marconi both reached 93.86%; vLLM+
reached 6.18% and 46.83%, respectively.  Each final tree is within its exact
byte budget.  The wrapper applies a deterministic post-insert call to each
simulator's own eviction policy because the artifact's pre-insert estimate can
under-evict on 4K requests.

These numbers are `debug_only` and `formal_evidence_eligible=false`.  The
simulator retains its native Attention--Mamba2 geometry.  Simulator wall time
and predicted FLOPs are context-only, and none of these rows may be pooled
with Qwen3.5 F1, TTFT, TPOT, or generated-token throughput.  `trace.json` and
`summary.json` are byte-bound by `SHA256SUMS`; a final preregistered rerun is
still required before any appendix citation.
