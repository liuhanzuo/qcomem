# HYPIC same-model/same-slice protocol adaptation

This reviewer-safe package records the completed official-code HYPIC comparison on Qwen3.5-35B-A3B. It is a TP=1 adaptation of the authors' released code at commit `98147c01909004e66d98bcb18b886927d41b0ee5` (SGLang 0.5.14), not a reproduction of the paper's TP=2 Qwen3.5 latency panels.

All three modes use the same eight Qasper/2WikiMQA items (indices 6--9), one H20-3e per item, a 4,096-token input cap, greedy decoding, at most 32 generated tokens, and OpenAI-compatible streaming client wall-clock timing. Each formal cell starts from a fresh server/cache after a discarded, length-matched, prefix-disjoint warmup.

The 24 formal cells completed: eight Full Recompute, eight Prefix Cache, and eight HYPIC `transition_rope_recompute` cells with the released eight-token seam. The summary reports F1 and timing without a preregistered pass threshold. HYPIC is approximate: seven of eight prediction texts match Full Recompute, while one differs.

Key bindings:

- formal summary SHA-256: `0543b491e70ddfaf6d40651b1f1babec652bd9c8f2a5f9d0cca7305cc2cb1b3d`
- preregistration SHA-256: `3f4e144e49555c5c58f5d14644a259ed07e825c771c9ee1ee51eb6f101641c70`
- official-source ledger SHA-256: `6907c665dab50d9e63aacd058524df5de4e8452a91fed9afe519cc05ad522fab`
- environment ledger SHA-256: `9f28825925bf090e97624fcf2671bf8fa862de29092dd4547e18f19d09bd68ff`
- terminal artifact ledger SHA-256: `4578b9c238e9ea82efab8f7ed316d9b00b606857d143cc16bc1e2c9b3edf8bb0` (4,404 entries)
- terminal static verification SHA-256: `37b60ab26615475a678d6c6d8a5cc0f831fb808fc03eb830a2a4d16b3dcd4151`

`independent-summary.json` was regenerated locally from the 24 raw cells using the frozen aggregate implementation and is byte-identical to `summary.json`.

The package does not support cross-runtime speedups, scheduler throughput/QPS, continuous batching, capacity, production robustness, or accuracy generality beyond this slice. Published HYPIC numbers are not inserted into the measured table.
