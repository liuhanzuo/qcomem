# Same-protocol related-work baseline TODO

Status date: 2026-08-21

## Fixed comparison contract

An entry may share the in-process H20 comparison table with CoMem only if all
of the following are held fixed and independently recorded.  Serving engines
that expose no commensurate reusable-state byte counter instead enter the
separate HTTP-serving panel after satisfying the common model/workload/decoding
and replay conditions; their timings are never pooled with CoMem.

- Qwen3.5-35B-A3B checkpoint and tokenizer;
- the frozen `longbench_validation.jsonl` artifact (SHA-256
  `1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe`);
- Qasper and 2WikiMQA source indices 6--9 (eight workloads total);
- 4,096 input-token cap, greedy decoding, and at most 32 generated tokens;
- one H20-3e per workload under the same worker image and CUDA stack;
- mean answer F1, TTFT, median TPOT, end-to-end generated-token throughput,
  and reusable-state bytes, with the same denominators;
- exact code, model, data, environment, GPU, raw-shard, and aggregate closure.

Existing CoMem rows are immutable inputs to this workstream and are not rerun.
Published values obtained under another model, accelerator, dataset slice, or
timing convention remain in the published-context table only.

## Priority TODO

| ID | Method | Required work | Main-table eligibility | Status |
|---|---|---|---|---|
| RW-B1 | vLLM/PagedAttention with prefix caching | Run the common streaming-serving harness on the eight fixed LongBench workloads and 32-token generation; record cache-on and cache-off controls, F1, streaming TTFT/TPOT/throughput, cache counters, and cache-on/off output agreement. | Serving-panel eligible after raw-shard and terminal replay. It is not pooled with in-process CoMem timing. | `verified_complete`; Job 246248 / Trial 1870299; 16/16 raw shards and 66/66 artifact hashes replay; 8/8 cache hits and prediction matches; validation SHA `a03b5a6290b589549016bc4278814a57a512e45555a0a17388ac50665a90404a` |
| RW-B2 | SGLang/RadixAttention | Pin SGLang 0.5.17; run the same eight workloads with radix cache on and disabled using the RW-B1 client/aggregator. Keep `extra_buffer` Mamba scheduling and 64-token pages fixed across both phases; record request-level cached tokens, metrics, F1, and timings. | Same serving panel as RW-B1 after raw-shard and terminal replay; not pooled with in-process CoMem timing. | `verified_complete`; persistent-node Job 246306 / Trial 1870703; final run `20260821c`; 16/16 raw shards and 80/80 artifact hashes replay; 8/8 cache hits and prediction matches; package manifest SHA `0bc433a8fec653e8eeeffad0369b9c30be2ce3e9ddf16d2cffa68178439612c7` |
| RW-B3 | Prompt Cache | Audit whether a single whole-document prompt module is semantically equivalent to the already measured full-prefix reuse control. The official prototype supports Llama2, Falcon, and MPT rather than Qwen3.5, so do not relabel the existing row as an official Prompt Cache reproduction. | No official-code row under the fixed checkpoint. A clearly named protocol-equivalent whole-document-module control may be discussed separately. | `official_model_unsupported` |
| RW-B4 | Marconi | Construct a separate repeated-request trace from the frozen eight workloads, preregister arrival order, reuse classes, cache budgets, and exact-match rules, then compare vLLM+, SGLang-style, and Marconi policies using the official simulator. Report token hit rate only as the portable outcome; do not mix simulator time or native-geometry FLOP estimates with single-query F1/tok/s. | No; policy/trace appendix table only. | `verified_complete`; persistent-node Job 246593 / Trial 1871681; official commit `08016617...`; preregistration SHA `501448f8...`; 128-event synthetic multi-turn extension; exact-budget formal results in `qcomem-marconi-formal-a/` |
| RW-B5 | Palu | Reproduce official low-rank decomposition, calibration/rank search, compressed-model export, and kernel path before evaluating the fixed workloads. Qwen3.5 hybrid attention/GDN support must be implemented and validated upstream; ordinary Q8/Q4 cache quantization is not Palu. | No until a faithful Qwen3.5 implementation exists. | `official_model_unsupported` |

## Direct CoMem-versus-related-work additions

The following additions answer a narrower question than the published-context
table: under one frozen checkpoint and workload, how do executable baselines
compare with CoMem at an actually shared measurement boundary?  Existing
verified vLLM/SGLang shards are reused; no unaffected baseline is rerun.

| ID | Direct comparison | Frozen cells | Required common boundary | Status |
|---|---|---|---|---|
| RW-D1 | CoMem versus vLLM prefix cache and SGLang RadixAttention | Verified vanilla vLLM cache-off; full-prefix Q16; CoMem Q16, Q8, Q4/Q4/Q8, per-layer mixed, and Q4; verified vLLM cache-on and SGLang radix-on | Exact model/tokenizer, eight LongBench rows, 4096-token cap, greedy 32-token decode, one H20-3e per row, and the same `stream_completion` client-wall TTFT/TPOT/throughput parser.  A local OpenAI-compatible CoMem wrapper may adapt the internal split interface, but wrapper time remains inside the client-wall interval.  Do not run a second wrapper-specific dense path: the already verified vLLM cache-off phase is the vanilla row. | `implementation_in_progress`; only the new CoMem/full-prefix streaming cells will run |
| RW-D2 | CoMem paged Q16 versus official Hydragen versus replicated dense | N=8 and N=32 resident queries on the frozen rank-0/layer-3 post-RoPE capture | Same Q/K/V bytes, causal/GQA geometry, CPU-FP32 oracle, H20-3e, warmup/iteration counts, and timing harness.  Report output relative-L2, logical/allocated K/V bytes, and GPU latency; disclose that Hydragen batches fan-out while the current CoMem contract is sequential. | `design_ready`; reuse the completed Hydragen/dense cells and run only the missing CoMem cells |
| RW-D3 | CoMem versus HYPIC position-independent caching | HYPIC `full_recompute`, `prefix_cache`, and `transition_rope_recompute` (official seam width 8), plus the matched CoMem rows from RW-D1 | Official HYPIC commit `98147c01909004e66d98bcb18b886927d41b0ee5`, Qwen3.5-35B-A3B, H20-3e, the same eight LongBench rows, 4096-token cap, greedy-32 decode, and one common client-wall streaming parser.  The document is the reusable segment and the query is the uncached suffix.  Before GPU scoring, require that concatenating the per-segment token IDs exactly reproduces the separator-free prompt token IDs. | `implementation_in_progress`; highest-priority external baseline because official code directly supports the target hybrid model |
| RW-D4 | LMCache + vLLM hybrid prefix cache qualification | One cold/warm LongBench row first; eight rows only after qualification | vLLM 0.26.0, LMCache 0.5.4, `mamba-cache-mode=align`, unified block/chunk size 1056, separate object groups, and the same model/data/decoder.  Require a real full-block store/retrieval, bounded score loss, no hang/OOM, and raw output retention. | `debug_qualification_in_progress`; appendix-only unless the gate passes |

RW-D1 and RW-D3 are the primary leaderboard candidates because they compare
complete generation paths and answer quality.  HYPIC is the closest external
method: unlike ordinary token-KV compression, it explicitly composes recurrent
state for hybrid stacks and its official code names the exact target model.
RW-D2 is an operator-level mechanism comparison, not an end-to-end throughput
substitute.  RW-D4 is admitted only after its compatibility qualification.
Palu and Marconi remain outside these leaderboards unless future work supplies
respectively an all-layer Qwen3.5 implementation or an identical
admission-policy trace applied to every state representation.

### RW-D3 acceptance and stop rules

- Freeze the official HYPIC source commit and run only its published
  `full_recompute`, `prefix_cache`, and `transition_rope_recompute` modes; do
  not optimize or tune HYPIC against the eight scored rows.
- The separator is a control-plane delimiter, not a model token.  For every
  workload, independently verify that the concatenated per-segment token IDs
  equal the separator-free prompt token IDs byte-for-byte before generation.
- Prime the document segment with a dummy suffix, then measure the same target
  query used by CoMem.  Require positive `cached_tokens` in both the HYPIC and
  prefix-cache warm phases.
- Report HYPIC as approximate and keep the per-row generated text.  Preserve
  any quality loss relative to full recomputation; never replace it with a
  paper-reported number.
- HYPIC's official Qwen3.5 recipe defaults to TP=2.  Before any formal output,
  run a debug-only TP=1 load and one unscored request on one H20-3e.  If that
  resource-matched path succeeds, freeze TP=1 for the formal comparison and
  record that this is a fairness amendment to the official default; otherwise
  retain TP=2, report two H20s explicitly, and do not normalize it into a
  single-H20 speed claim.

### RW-D4 acceptance and stop rules

- Use a single scored row until the LMCache server registers all hybrid cache
  groups, stores and retrieves at least one complete 1056-token block, and
  terminates without hang or OOM.
- The official recipe states that fresh and restored GDN outputs need not be
  bit-exact.  Use a predeclared score-level tolerance, retain token/output
  differences, and do not demand byte equality as the qualification gate.
- A torch-baseline LMCache transfer path without its compiled CUDA extension
  may establish compatibility but is not eligible for a main-table performance
  claim; either install the matching extension or disclose the result only in
  the appendix.

### RW-D1 acceptance and stop rules

- The CoMem wrapper must accept the same token-ID OpenAI completions requests
  and emit the same SSE/usage schema consumed by the frozen baseline client.
- Cache-off includes document-state construction inside measured TTFT;
  cache-on primes outside the measured request and must record a positive hit.
- Each configuration must expose exact physical persistent-state bytes from
  the executed state object; a theoretical estimate is insufficient.
- Cache-on/off predictions must agree for every workload.  Any disagreement is
  a valid negative and is not averaged away.
- The aggregate must recompute F1 and all timings from the eight raw client
  traces.  The existing vLLM/SGLang aggregates are imported by SHA-256 rather
  than rerun.

### RW-D2 acceptance and stop rules

- Use the already frozen Hydragen capture and official-source binding; do not
  recapture model tensors or rerun Hydragen/dense unless a shared input changes.
- A CoMem cell must bind the same post-RoPE Q/K/V sidecars and independently
  replay output relative-L2 from FP32 raw output bytes.
- Latency is reported both as total N-request service time and per-request
  equivalent.  It must not be called concurrent throughput for the sequential
  CoMem path.
- If the present paged kernel cannot consume the frozen ragged fan-out without
  changing its supported contract, record that incompatibility instead of
  inventing a proxy number.

## Execution order and stop rules

1. Implement and CPU/static-test RW-B1 without rerunning CoMem.
2. RW-B2 is complete after isolated debugging and one final full 8-rank run;
   earlier incomplete attempts remain registered but supply no paper numbers.
3. RW-B1 and RW-B2 remain separate evidence packages so either result can be
   replayed independently.
4. Add a row to either comparison table only after raw-shard replay,
   metric-denominator replay, and terminal closure pass.  Rows without a
   commensurate physical reusable-state counter belong only in the separate
   serving panel.  A valid negative is registered but does not become a
   positive speed/quality row.
5. RW-B4 formal trace is complete.  Its official simulator uses native
   Attention--Mamba2 geometry and a disclosed synthetic multi-turn extension,
   so it remains appendix-only.  It is not required for the current
   fixed-configuration feasibility claim and must never be inserted into the
   H20 F1/throughput panel.
6. Leave RW-B3/RW-B5 as explicit compatibility gaps unless faithful official
   support becomes available. Never fill missing cells from paper-reported
   numbers or from a different model.

## Frozen RW-B1 submission preview

- job name: `liuhanzuo-qcomem-related-vllm-prefix-20260820a`;
- YAML: `qs/qcomem-related-vllm-prefix-20260820a.yaml`, SHA-256
  `35b2b4f5b4654d2147eaa4128c2e27cea46cb6f0c246ef27908c41dbe2708f1c`;
- code ledger SHA-256:
  `d56988ce98b95682ce1a6d32ea30b9f7286f02b0b84a5b0bdb23aecc67713f01`;
- protocol SHA-256:
  `1153563554a6f615c6721368da4187cc659f9adb899d6435480f2256d76b6af3`;
- resource: queue 408, cloud 6, cluster 53, package 183
  (`8Gpu/170C/1800Gi`), one worker;
- remote code closure: five files uploaded and downloaded byte-for-byte;
- output root:
  `.../runs/qcomem/related-vllm-prefix-20260820a`, confirmed absent before
  create;
- deduplication: exact-name search returned zero jobs;
- dry run: exact POST body resolved successfully; no create request sent.

## Acceptance checks for a new main-table row

- eight unique `(dataset, source_index)` rows and eight distinct GPU UUIDs;
- cache-enabled and cache-disabled paths produce the same generated tokens for
  each workload, or any divergence is reported rather than averaged away;
- F1 is recomputed from raw generated text and frozen references;
- TTFT/TPOT/throughput are recomputed from raw timestamps with warmup excluded;
- reusable-state bytes are physical engine/cache accounting, not a theoretical
  tensor estimate;
- no result from another model, accelerator, or LongBench slice appears in the
  same numeric column.

## RW-D5: affected-only HYPIC retained-state byte inventory

The completed HYPIC run establishes cache hits, output/F1, and streaming
timings, but its OpenAI-compatible usage receipt exposes only
`cached_tokens`.  It does not expose the physical bytes retained for one
document.  Do not substitute NVML, `memory_reserved`, or whole-process deltas:
SGLang preallocates KV and recurrent-state pools, so those quantities measure
pool capacity rather than the table's retained-document payload.

Run only the affected Prefix Cache and HYPIC arms on the same eight frozen
workloads; do not rerun Full Recompute, CoMem, RR2, GDN, serving controls, or
unrelated baselines.  After prime and before the measured query, emit a
read-only cache-entry receipt containing:

- the exact cached segment hashes, token counts, and full-attention KV slot
  indices;
- the associated `mamba_state_slot` for each HYPIC segment;
- dtype, shape, element size, backing-storage identity, and byte range for the
  full-attention KV payload and the recurrent/PIC `conv`, temporal,
  transition, and `conv_tails` tensors;
- metadata bytes separately from tensor payload bytes; and
- a unique-storage, overlap-aware sum that deduplicates aliased backing
  storage and is recomputed blindly from the raw receipt.

The reported `Store (MiB)` is the median physical payload owned by the cached
document under this receipt, not allocated pool capacity.  Accept the row only
if the cached-entry inventory covers every reported cache hit, every storage
range lies inside its frozen pool tensor, the unique-byte sum replays exactly,
and terminal cache-entry removal restores the corresponding ownership count.
Until this affected-only run passes, keep HYPIC Store as `n/r` and make no
CoMem-versus-HYPIC storage-ordering claim.

### Invalid attempt receipt: Trial 1876986

Trial `1876986` is invalid and contributes no paper evidence.  It stopped
before `0/16` authorized cells produced raw or store receipts.  All eight
`/model_info` endpoints had become ready, but each SGLang scheduler continued
an internal 80-token warmup for approximately 46 seconds.  The C launcher then
issued a one-shot `/server_info` request with a 30-second timeout, so the first
server-receipt stage failed before any measured workload.

The failed exit also exposed a lifecycle bug: the eight server process groups
were not reaped and each H20-3e still showed 90,968 MiB allocated.  The process
groups were subsequently identified from that run directory's exact PID files,
sent `TERM`, and checked after 10 seconds; all eight GPUs were then at 0 MiB / 0%
and no SGLang process remained.  This cleanup is an operational recovery, not
an experiment result.  Freeze C is retired.  Freeze D must add evidence-bearing
`/server_info` polling with a bounded total deadline and an idempotent
`EXIT`/`ERR`/`INT`/`TERM` cleanup path before another GPU submission.

### Freeze D retirement and freeze E release conditions

Freeze D was not submitted to GPU and supplies no experimental evidence.  A
fresh independent audit found two pre-execution lifecycle/identity blockers:
its cleanup sent `TERM` and then performed a potentially unbounded `wait`
before reaching the bounded liveness loop, and its readiness object could be
exchanged across ranks because the frozen cell and endpoint were not closed
independently at replay.

Freeze E keeps the scientific design, workload set, Store denominator, and
affected-only two-arm scope unchanged.  Its only changes are: (1) cleanup now
uses `TERM`, bounded `kill -0` polling, `KILL` for survivors, and a final reap;
failure writes `FAILED` and removes `COMPLETED` before cleanup, while success
finishes cleanup before creating `COMPLETED`; and (2) every readiness receipt
is bound to the exact mode, rank, base URL, `/server_info` endpoint, launch PID,
ordered poll attempts, and frozen 300/3/1-second polling parameters.  GPU
submission remains prohibited until the new E STOP passes independent audit.

### Freeze E retirement and freeze F expected-cell closure

Freeze E was not submitted to GPU and supplies no experimental evidence. A
fresh audit constructed a fully self-consistent, fully re-signed rank-1 chain
(worker, server, readiness, target, store, terminal, raw, process environment,
and all downstream hashes) and placed it in the frozen rank-0 file location.
E's replay verified internal consistency but had no caller-supplied external
mode/rank/snapshot/workload anchor, so it accepted the forged cell.

Freeze F preserves all scientific code and the E lifecycle repair. `replay_one`
now requires the externally expected mode, rank, snapshot ID, and frozen
workload ID, while `replay_all` derives those values from its immutable loops
and `EXPECTED_PAIRS`. Before append, `replay_all` independently rechecks the
returned row against the exact file position. The terminal readiness attempt's
response SHA is also required to equal the readiness-level and server-level
`server_info` SHA. The fully re-signed rank exchange must pass under its forged
rank-1 expectation yet fail under the actual rank-0/qasper-6 location in both
`replay_one` and `replay_all`. GPU submission remains prohibited until F is
independently audited GREEN.

### Freeze F runtime invalidation and G component dtype gate

Freeze F passed static audit but its formal attempt is invalid before results.
All eight Prefix servers reached evidence-bearing readiness, then failed in
`maybe_emit_owned_state_snapshot -> _mamba_payload_ranges` with
`ReceiptError: temporal dtype`. The run has `FAILED=1`, exactly stages 00, 01,
02, and Prefix 10, eight targets, sixteen readiness/server receipts, and zero
raw or store receipts. It contributes no scientific negative result and no
paper number. The exact log hashes and failure disposition are registered in
`invalid-attempt-freeze-f-trial-1876986.json`.

The G gate replaces the retired unified dtype with a component-level contract:
KV and convolution use BF16; temporal state uses FP32; transition follows the
temporal dtype; and convolution tails follow the convolution dtype. Producer
and blind replay must each reject the legacy unified field and validate the
component dtype, element size, shape, stride, and selected byte range. Before
G can freeze, Trial 1879097 may run only one GPU sequentially for Prefix and
HYPIC, emitting a debug-only component inventory and no formal raw/store
receipt. Only after both live inventories match official pool construction may
G enter independent STOP audit; the full 16-cell formal run remains forbidden.

### J live component gate passed; K formal refreeze

The one-GPU debug-only J run on Job 247512 / Trial 1879097 completed
terminally with `COMPLETED_DEBUG_ONLY`, no failure marker, zero files in the
formal-receipt directory, a passing artifact ledger, and GPU0 released to
0 MiB / 0%. It is runtime-contract evidence only, never a paper Store result.
The exact 19-file local mirror is frozen under
`live-debug-j-trial-1879097/`; its manifest SHA-256 is
`59530c0c8bc10cedbf4b0bde51d04e5490adeaf369e8738d9df363fc83941026`.

Prefix observed BF16 convolution and FP32 temporal state. HYPIC observed BF16
convolution/tails and FP32 temporal/transition state. Both had 30 recurrent
layers, exact allocator-slot axes, C-contiguous live tensors, the expected
cache classes, explicit bfloat16/float32 environment, and official commit
`98147c0`. These observations close the component dtype uncertainty that made
F invalid.

Freeze K embeds the exact debug mirror and revalidates its local/remote hash
ledgers, terminal, raw/validation/run/target chains, component semantics, and
topology before and after formal output. Producer and blind replay both require
the resulting live-debug binding. K preserves the existing physical-byte
denominator and authorizes only the 16 affected Prefix/HYPIC cells. Formal GPU
submission remains prohibited until exact K STOP receives independent GREEN.

### Final r34 disposition (Job 247699 / Trial 1892234)

The affected-only Store experiment is complete at the measurement-cell level:
8/8 Prefix Cache and 8/8 HYPIC cells produced raw, pre-measurement, and terminal
receipts.  A strengthened external replay accepts 16/16 cells and independently
recomputes median retained-document tensor payloads of 146,309,120 bytes for
Prefix Cache and 339,834,880 bytes for HYPIC.  The accepted evidence package is
`hypic-retained-state-r34-trial1892234/`.

The original launcher terminal disposition is retained and is not reclassified.
Accordingly, the paper claim is limited to external cell acceptance and does
not include whole-launcher completion, native terminal-static closure, NVML or
process memory, capacity, timing, continuous batching, or ForkAudit ownership.
No further GPU rerun is required for this bounded Store result.
