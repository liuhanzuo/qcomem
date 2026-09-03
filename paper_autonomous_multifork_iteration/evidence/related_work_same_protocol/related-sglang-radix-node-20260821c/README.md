# SGLang/RadixAttention same-protocol serving control

This package records the completed final run on the persistent QuickSilver
8×H20-3e node (Job 246306, Trial 1870703).  Eight independent SGLang 0.5.17
servers each execute one frozen LongBench workload under two fresh-process
conditions: RadixAttention disabled, then enabled after a document-only prime.

The paired result is a valid positive control: all 8 cache-enabled requests
report cached prompt tokens, and all 8 predictions are exactly equal to their
cache-disabled counterparts.  Mean F1 is 39.1372 in both phases.  Median TTFT
changes from 0.2834 s to 0.1465 s; median TPOT from 0.05338 s to 0.05315 s; and
median generated-token throughput from 12.7285 to 15.8257 token/s.

The timing boundary is OpenAI-compatible streaming client wall-clock.  It is
directly comparable to the companion vLLM serving panel on the same checkpoint,
slice, decoding rule, and one-GPU-per-workload layout.  It is not directly
comparable to the paper's in-process CoMem adapter timings.  CUDA graphs were
disabled, requests were single-stream rather than continuously batched, and
SGLang fell back from missing Triton 3.6 H20-3e MoE tuning files to bundled
Triton 3.5.1 entries; therefore the row is not a claim of best achievable
SGLang performance.  Text-only requests succeeded despite nonfatal optional
TorchCodec/FFmpeg warnings from unused multimodal processors.

Run `python3 validate.py` in this directory to replay the aggregate from all 16
raw shards and validate the local package manifest.
