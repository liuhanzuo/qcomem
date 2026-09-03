# A4 design decision: same checkpoint, independent runtime

This affected-path experiment intentionally holds the pinned hybrid checkpoint fixed at `Qwen/Qwen3.5-35B-A3B@59d61f3ce65a6d9863b86d2e96597125219dc754`; there is no other pinned hybrid checkpoint in the evidence inventory. Independence is supplied by the runtime and audit implementation: Transformers `DynamicCache` via `qcomem_torch.TorchSplitCausalLM`, a new producer/blind replayer, raw CPU-FP32 logit bundles, normalized storage-range witnesses, adapter-call and allocator ledgers, independent PG-19/tokenizer reconstruction, and a new five-fault suite. It imports neither current `qcomem_forkaudit_*` components nor vLLM ownership helpers.

The admissible conclusion is therefore limited to a same-model/different-runtime transfer of the seven-target audit contract. Tail-safe paged append is not applicable because `DynamicCache` has no paged partial tail. Dispatch provenance remains partial because Python callables/layer ranges are receipted but compiled kernel binaries, autotune choices, and instruction traces are unavailable. No model-generalization, production concurrency, memory-saving, latency, throughput, or capacity claim is authorized.

Formal work is not yet submitted. The QS specification reserves queue 408, cloud 6, cluster 53, package 183 and the launcher assigns all eight H20 UUIDs to eight independent ranks.
