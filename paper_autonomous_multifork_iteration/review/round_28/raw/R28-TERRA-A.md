# R28-TERRA-A

Concise summary: ForkAudit is a receipt-and-replay contract for auditing state ownership when a long prefix is forked across hybrid LLM continuations. On one sequential Qwen3.5-35B-A3B/vLLM-plus-GDN stack, it logs lifecycle, storage-range, COW, and Python-call receipts; replays them under an honest-capture-producer assumption; and evaluates four KV×GDN ownership configurations at N={1,8,32}. It reports exact registered-observable equality, selected attention/GDN operator checks, designed-fault gate localization, and a shared-KV allocation reduction.

Strengths:

- The paper identifies a real gap: token/logit agreement does not establish isolation of mutable recurrent state.
- It states a concrete audit contract with unusually clear target definitions, trust boundary, memory denominators, and what remains open.
- The factorial design and phase-aware storage witnesses directly exercise relevant ownership/COW invariants.
- The target-gate suppression matrix is a thoughtful demonstration that several designed violations can leave tokens and registered logits unchanged.
- The manuscript is transparent that its selected numerical oracles, mutations, and context tables are not general detection, end-to-end, concurrency, or capacity evidence.
- Reproducibility documentation is unusually detailed, including artifact maps, manifest-based replay, hashes, and a disclosed post-execution aggregation correction.

Weaknesses:

- **Must fix:** The central evidence is only as strong as the “honest capture producer,” while the producer appears coupled to the stack being audited. Receipt replay detects inconsistencies in represented records, not omission/fabrication or compiled-dispatch deviations. This leaves the empirical result closer to a well-specified self-audit than independent assurance.
- **Must fix:** Practicality is not evaluated: the paper explicitly does not measure capture, replay, or instrumentation overhead, despite an approximately 851 MiB artifact footprint. This is important for a proposed adoption workflow.
- **Must fix:** Generality is very limited: one model, one hardware family, BF16 KV, batch size one, sequential single-stream execution, short decoding, and only selected operator rows. No concurrent kernels, continuous/ragged batching, eviction, or production scheduler are exercised.
- The nine faults are designed around known gates. The suppression study is careful not to call itself a detection rate, but it still provides limited evidence that ForkAudit finds realistic, unforeseen implementation defects.
- The novelty is primarily an integration/specification of established ownership, metamorphic-testing, logging, and oracle ideas. The paper needs a sharper argument or evidence that this integration yields materially new assurance beyond disciplined existing systems tests.
- A large amount of unpooled context and related-system material makes the contribution feel diffuse relative to the narrow primary result.

Questions for authors:

1. What component is the independent trusted capture producer, and how is it isolated from the runtime paths whose ownership behavior it records?
2. What are capture-time overhead, offline replay time, storage overhead, and code-integration burden for a representative deployment?
3. Can the approach detect at least one naturally occurring or independently introduced bug not authored to match a registered gate?
4. Which parts of the contract remain valid under concurrent CUDA streams/continuous batching, and what new records or observers would be needed?
5. Why should this be viewed as a broadly useful new method rather than a rigorous, one-stack engineering validation protocol?

Reproducibility assessment: Moderate-to-good for replay of the reported receipts. The PDF specifies software/hardware, fixed geometry, artifact paths, hashes, and offline replay coverage. However, replay is not producer recapture or compiled-binary attestation; much of the numerical state is digest-based, full reproduction requires pinned model/data/hardware, and runtime/overhead details are absent.

Overall score: **4/10 (Weak Reject).** The paper is careful, technically thoughtful, and unusually candid, but its strongest claims are conditional self-audit evidence on one narrow sequential stack. Missing independent observation, overhead/practicality evidence, broader runtime coverage, and realistic defect validation prevent me from finding the contribution sufficiently established for ICLR.

Confidence: **4/5.**

