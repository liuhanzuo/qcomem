# R30C corrected native-batching evidence package

R30C preserves R30B's full-vocabulary scientific contract and corrects one
receipt dimension.  Qwen3.5's tokenizer has 248,077 encodable IDs, while the
model logits and vLLM `ModelConfig.get_vocab_size()` have 248,320 positions.
R30B incorrectly allocated dense sidecars at the tokenizer width and therefore
failed when vLLM correctly returned all model-vocabulary keys.

The corrected contract requires every public per-step logprob dictionary to
contain exactly keys `0..248319` after sampled-token deduplication.  Each dense
FP32 sidecar is indexed by model token ID, and the detached replay independently
rehashes, reloads, and recomputes comparisons for all six sidecars.  The frozen
`atol=0.005`, `rtol=0`, exact generated-token, finite-mask, native scheduler,
ownership, zero-before-use, and sequential-control gates are unchanged.

Before any model load, the runner verifies both model ledgers, tokenizer and
model vocabulary widths, the model config SHA, eight installed vLLM source
SHAs, and two synthetic sampled-token-deduplication fixtures (one tokenizer ID
and one padded model-vocabulary ID).  Both fixtures passed on the target vLLM
0.26 environment without touching a GPU.

This package remains narrower than ForkAudit.  It can establish native vLLM
scheduler batching and scheduler-visible KV block ownership/lifecycle on one
fixed stack.  It cannot establish that the Transformers CoMem/ForkAudit cache
facade is integrated into vLLM EngineCore, ForkAudit GDN receipts, independent
GPU-memory observation, production safety, performance, or generality.

R30C is frozen but has not been scientifically executed.  Do not execute it or
stop the QS node until the root agent explicitly authorizes the formal run.
