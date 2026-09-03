# Round-6 A4: Transformers-cache independent-runtime transfer

Status: preregistered, not submitted. This is a formal-only affected-path experiment; it has no smoke, retry-with-changed-settings, or fallback path.

## Boundary and purpose

The checkpoint remains `Qwen/Qwen3.5-35B-A3B@59d61f3ce65a6d9863b86d2e96597125219dc754` because no second pinned hybrid checkpoint is available. The transferred runtime is materially different: standard Transformers `DynamicCache` through `qcomem_torch.TorchSplitCausalLM`, with an independently written producer, storage witness, dense oracle, fault suite, sidecar format, and blind aggregate replay. No `qcomem_forkaudit_*` module or vLLM ownership helper is imported.

This experiment supports a same-model/different-runtime transfer claim only. It does not support a second-model, paged-tail, compiled-kernel identity, concurrency, latency, throughput, capacity, or memory-saving claim.

## Frozen design

- Eight independent ranks use eight distinct PG-19 train books and eight distinct H20 GPU UUIDs. The formal cell is `N={1,2}` with a 256-token document, two distinct 24-token query windows, split depth 7, and two semantic generation steps.
- The deep-materialized arm independently calls `write_lower_replay` per request. The persistent arm constructs one base and calls `LowerReplayState.fork`; all request suffix caches remain live while requests are interleaved on one CUDA stream.
- The dense authority recomputes the full model from frozen raw token IDs, outside either ownership arm. Every dense/arm/fault full-vocabulary logit is stored as canonical little-endian CPU-FP32 bytes. The aggregate reopens the bytes and recomputes hashes, top-1, max-absolute error, relative L2, token equality, cross-arm equality, and cross-N equality. The relative-L2 threshold is 0.005, copied from the prior independently frozen audit and never tuned after A4 output.
- The source preregistration is rebuilt byte-for-byte from the exact PG-19 train64 JSONL plus its source manifest and the local tokenizer before model output. Eight book/source/start/length choices and raw-int64 document/query hashes are frozen.
- Model authority parses the two small ledgers on every rank but reads all fourteen weight payloads only in one pre-output authority pass and one terminal closure pass. Authority entries must equal parsed ledgers; stat paths must equal the artifact/weight union; every target must be a regular file with no owner/group/other write mode bit; and size/device/inode/ctime are checked. The two full-hash receipts must be byte-identical. This is ordinary pathname/workflow mutation evidence for the frozen job, not protection against a privileged root or raw-device adversary.
- A pre-output GPU receipt fixes rank-to-UUID assignment and requires exactly eight unique `NVIDIA H20-3e`, compute capability 9.0, BF16-capable devices. The exact `-3e` suffix is part of the platform-reported identity contract.

## Portable records and seven targets

Identity records bind source, tokenizer, PG-19 tokens, formal config, model ledgers, model authority, environment, geometry, split depth, and GPU assignment. Ownership records use salted opaque storage-base IDs plus normalized, contiguous byte ranges; non-contiguous or empty authorizing inventories fail closed. Execution records cover every lower/suffix adapter call with exact request order, layer range, position/current length, input length, append delta, cache tensor/content/storage before and after, and completion. Accounting records capture CUDA allocated/reserved/max counters at setup, the all-live first-query transition, and final; they are contextual only.

The seven outcomes are:

1. Frozen identity: applicable/full if all bindings replay.
2. Persistent-prefix immutability: applicable/full.
3. Mutable-cache private ownership: applicable/full at setup, first transition, and final, including non-vacuous `N=2` request-request and request-base tensor-range comparisons.
4. Paged partial-tail copy-before-append: not applicable. `DynamicCache` has no fixed-size paged partial tail; prefix immutability and private ownership do not upgrade this target.
5. Dispatch provenance: applicable/partial. Python adapter, per-call layer range, and layer classes are recorded; compiled kernel binary, autotune selection, and instruction trace are exactly missing.
6. Deep-materialized versus persistent-fork equivalence: applicable/full.
7. Cross-N first-request prefix consistency: applicable/full.

## Frozen fault exercises

Every case has a matched clean receipt, raw mutant receipts, an applicable detector vector, a frozen expected predicate, an independently replayed failed-predicate set, classification, and validity.

- T1 is a digest-proven common-mode document-boundary residual content mutation in both arms. Its validity comes from pre/post content change, identical post-mutation digests, stable storage, and persistent residual aliasing to its separately corrupted base—not from the observed cross-arm outcome. The independent dense oracle is expected to fail while cross-arm equality may remain green.
- T2 replaces one request's live mutable-cache leaf with another request's compatible tensor. Normalized overlapping byte ranges are expected to fail private ownership.
- T3 advances `current_length` by one while retaining the frozen document boundary. Position/current-length canonicality is expected to fail.
- T4 changes one element of the Q16 packed document residual. Packed-state content immutability is expected to fail.
- T5 changes one element of a live lower-cache tensor without changing its storage identity/range and continues real model execution. Downstream output consistency is defined only by successful completion plus token/full-logit sidecars; state cross-arm equality and the dense oracle remain separate detectors. Evidence errors, CUDA/runtime errors, OOM, and failure of the unmodified materialized comparator abort. Only a preregistered ordinary `AssertionError` from the mutated persistent call may be serialized as a scientific output-or-exception.

T2--T4 are direct contract-sensitivity exercises. T1 and T5 are mixed downstream faults not defined one-to-one by a relational ownership gate. Wrong-predicate detections, escapes, and clean false positives are representable and are valid negative outcomes; producer-reported classifications never authorize the aggregate.

## Formal execution and stopping rule

The launcher first verifies the source manifest, performs the pre-output full model authority pass, bytewise rebuilds the static preregistration, and freezes the GPU assignment. It then launches exactly one rank on each of eight UUID-isolated GPUs. Shards and logit bundles use temporary-file atomic commits. After all ranks finish, the launcher performs the terminal full model closure, blind-replays all eight canonical shards from the static manifest and FP32 sidecars, writes the aggregate and complete artifact ledger, and only then writes `COMPLETE`.

Infrastructure/evidence failures produce no scientific result. Clean-contract failures, oracle failures, fault escapes, clean false positives, or wrong-predicate detections produce a valid negative aggregate. No parameter, threshold, document/query window, fault, or missingness category may change after formal output.
