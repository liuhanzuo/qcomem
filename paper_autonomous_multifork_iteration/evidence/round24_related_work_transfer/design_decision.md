# Round 24 related-work transfer decision

Selected first experiment: the pinned Hydragen attention operation on
hash-bound Qwen3.5 post-RoPE tensors captured by RR2.  The official end-to-end
Hydragen model wrapper is Llama-only, so this is intentionally an operator
transfer rather than a claim that the released model wrapper supports Qwen3.5.

The source cell is RR2 rank 0, layer 3, document length 4,095, suffix length
32.  The 32 recorded suffix queries form last-token requests with unique-cache
lengths 1 through 32.  The N=8 cell uses positions 4, 8, ..., 32; the N=32 cell
uses all positions.  Both cells share the same recorded document K/V exactly.

The direct baseline is Hydragen's own replicated full-KV attention path in the
same environment.  Both outputs are compared with a separately implemented
CPU-FP32 grouped-query attention oracle.  Timings use the pinned official
Hydragen benchmark helper after warmup and CUDA graph capture.  The result is
eligible only if all input hashes, shapes, dtypes, model geometry, official
source hashes, GPU identity, oracle thresholds, and output completeness gates
pass.

Palu is the second-stage candidate.  Its Qwen2 wrapper cannot be reported as a
Qwen3.5 result.  Any port must target only the ten full-attention K/V
projections and must be evaluated end-to-end before its quality or speed is
placed beside CoMem.

## Executed results and selection

The Hydragen transfer passed its frozen numerical gates at N=8 and N=32.  Its
logical shared-prefix footprint was 7.59x and 25.80x smaller than physical
replication, respectively, while the compatibility path was slower than the
matched dense FlashAttention call in both cells.  It is therefore retained as
positive operator-compatibility and logical-storage evidence, not as a speedup.

For Palu, a first plain per-head SVD diagnostic produced high projection error.
The follow-up formal run therefore froze Palu's released activation-covariance
whitening method before new output: PG19 rows 0--7 supplied 2,048 calibration
tokens and row 8 was disjoint held-out evaluation.  At ranks 64/128/192, the
activation-aware K error was 0.487/0.317/0.149 versus plain-SVD
0.711/0.543/0.341; V error was 0.549/0.364/0.194 versus
0.839/0.668/0.459.  Every preregistered finite/monotonicity gate and the
held-out whitening-improvement hypothesis passed.  The remaining error still
forbids end-to-end quality or speed claims.

The defensible combination conclusion is structural and bounded.  Hydragen or
Palu can address full-attention KV while CoMem manages persistent recurrent and
document state, and both released mechanisms were exercised on the frozen
Qwen3.5/H20 stack.  No jointly optimized all-layer model, fused Palu kernel,
LongBench quality run, or combined serving throughput was executed, so none is
claimed.
