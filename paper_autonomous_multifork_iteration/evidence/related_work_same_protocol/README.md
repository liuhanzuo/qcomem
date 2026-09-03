# Same-protocol serving baselines

This package contains serving-framework baselines run on the same frozen
Qwen3.5-35B-A3B checkpoint, LongBench slice, H20-3e hardware class, decoding
rule, and eight-workload assignment used by the H20 comparison.  HTTP streaming
timings are reported in a separate serving panel and are not pooled with the
paper's in-process CoMem adapter timings.

## Verified vLLM prefix-cache run

- Platform: Job 246248 / Trial 1870299 (`Complete`).
- System: vLLM 0.26.0, BF16 model, eager execution, one independent server per
  H20-3e, prefix caching off versus `--enable-prefix-caching
  --mamba-cache-mode align`.
- Workloads: Qasper and 2WikiMQA source indices 6--9, 4,096 input-token cap,
  greedy decoding, at most 32 generated tokens.
- Evidence: `related-vllm-prefix-bootstrap-f-20260820a/` is the exact run
  bundle; `vllm_prefix_bootstrap_f.tar.gz` is its frozen transport archive.
- Replay: `python3 validate_vllm_prefix_result.py
  related-vllm-prefix-bootstrap-f-20260820a --output
  vllm_prefix_validation.json`.
- Replay result: 66/66 artifact-ledger entries, 16/16 raw shards, eight unique
  GPU UUIDs, eight observed cache hits, and eight cache-off/on prediction
  matches passed.  Validation report SHA-256 is
  `a03b5a6290b589549016bc4278814a57a512e45555a0a17388ac50665a90404a`.
- Aggregate: mean F1 is 0.3963325653 in both phases. Median client-wall TTFT is
  1.4450776 s (off) versus 0.1789443 s (on); median TPOT is 0.0563092 s versus
  0.0539929 s; median generated-token throughput is 4.6985685 versus
  14.0023403 tokens/s.

The result has four mandatory disclosures retained and replayed from all server
logs: Mamba `align` prefix caching is experimental in vLLM 0.26; eager mode
disables torch.compile/CUDA graphs; vLLM resolved a 1,056-token attention block
with 0.76% Mamba page padding; and no H20-3e-specific MoE config was present, so
the engine used default tactics.  Consequently, the measured row is a pinned
configuration result rather than a claim about the best attainable vLLM
performance.

## Verified SGLang RadixAttention run

- Platform: Job 246306 / Trial 1870703 (formal run completed before the
  persistent debug node was released).
- System: SGLang 0.5.17, BF16 model, one independent server per H20-3e,
  RadixAttention disabled versus enabled with Mamba `extra_buffer` scheduling
  and 64-token pages.
- Workloads and decoding: identical to the vLLM panel above.
- Evidence: `related-sglang-radix-node-20260821c/` contains the 16 raw shards,
  summary, selected logs, protocol and environment bindings, independent
  validation, and package manifest.
- Replay: `cd related-sglang-radix-node-20260821c && python3 validate.py`.
- Replay result: all 16 raw shards and all 80 artifact-ledger entries pass;
  eight unique GPU UUIDs, eight observed cache hits, and eight cache-off/on
  prediction matches are recovered.  Package manifest SHA-256 is
  `0bc433a8fec653e8eeeffad0369b9c30be2ce3e9ddf16d2cffa68178439612c7`.
- Aggregate: mean F1 is 0.3913722478 in both phases.  Median client-wall TTFT
  is 0.2833681 s (off) versus 0.1464699 s (on); median TPOT is 0.0533827 s
  versus 0.0531487 s; median generated-token throughput is 12.7285309 versus
  15.8257456 tokens/s.

CUDA graphs were disabled and requests were single-stream rather than
continuously batched.  Missing Triton 3.6 H20-3e MoE tuning files caused a
fallback to bundled Triton 3.5.1 entries.  The result therefore verifies a
pinned cache-off/on control rather than best attainable SGLang performance.
