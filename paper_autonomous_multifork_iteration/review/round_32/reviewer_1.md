## 1. Summary and claimed contributions

This paper presents ForkAudit, a phase-aware trace-validation protocol for ownership of hybrid LLM KV-cache and recurrent GDN state after a shared-prefix fork. It records byte-bound lifecycle, storage-range, copy-on-write, dispatch, and semantic receipts; absent mandatory evidence leaves a target open. On one Qwen3.5/H20 stack, it reports 96 ownership configurations, targeted fault injections, selected captured-boundary numerical oracles, and a shared-KV allocation reduction. The claimed assurance is explicitly conditional on an honest, event-complete capture producer.

## 2. Strengths

- The paper identifies a real systems correctness gap: equal tokens/logits do not establish isolation of mutable state.
- The contract is unusually explicit about phases, mandatory evidence, byte bindings, target-specific coverage, and the distinction between a passing replay and complete coverage.
- The 2x2 ownership factorial, transition-time GDN checks, storage-range replay, tail-COW test, and fault-gate suppression study give coherent evidence for the narrowly stated case.
- The authors carefully separate primary evidence from unpooled contextual experiments and candidly disclose substantial limitations, including incomplete compiled-dispatch attestation and absence of independent recapture.
- The manuscript is polished and detailed; tables and appendices make the claimed boundaries and artifact mapping concrete.

## 3. Weaknesses

- The central guarantee is trace-relative and depends entirely on the candidate-side producer being honest and event-complete. The “source-distinct” observer still shares process, tensor objects, labels, and PyTorch storage API, so it does not substantially address coherent omission/fabrication or transient mutation.
- Generality is very limited: one model/adapter, H20 family, page size, partial-tail geometry, registered 32-token transition, N<=32, and batch-one sequential execution on one stream per rank. This excludes the production settings where ownership is especially difficult: continuous/ragged batching, multi-document serving, eviction, general scheduling, and in-flight cancellation.
- The semantic tests are mostly self-consistency relations. The numerical checks begin from producer-captured post-RoPE or post-normalization intermediates and cover selected operator boundaries; they do not validate upstream construction or end-to-end behavior. Targeted, author-designed faults demonstrate sensitivity but cannot support a broader detection-rate claim.
- Dispatch provenance remains partial because no per-call compiled-binary or autotuning binding exists. This materially limits the strength of the call-contract assurance.
- The paper is very dense, with many rounds and supporting cohorts. Although transparent, the main methodological novelty versus an extensive systems-testing integration could be articulated more crisply, and the main results are somewhat obscured by contextual tables.

## 4. Questions for the authors

- What practical path would make capture source-diverse or independently observable, rather than trusting the producer and its storage API?
- How does ForkAudit’s capture/replay overhead scale with resident requests and context length, and what CI/debugging workload is realistically feasible?
- Can the contract be demonstrated under native continuous batching, dynamic admission/eviction, and genuine in-flight cancellation?
- How should users interpret a “pass at Python-call scope” when compiled-dispatch identity and autotuning remain unbound?
- Which elements of the proposed protocol are essential and novel beyond combining established metamorphic testing, storage checks, and lifecycle instrumentation?

## 5. Reproducibility and ethics

The paper provides unusually strong PDF-level reproducibility documentation: frozen geometry, detailed schemas, artifact paths, hash-bound ledgers, and replay boundaries. However, reproducibility remains difficult to independently assess from the manuscript alone and likely requires a costly pinned H20 software/hardware stack; CPU replay cannot independently recapture the live execution. Ethics coverage appropriately notes the use of PG-19 training data and lack of human-subject evaluation. The paper should also discuss compute/energy cost of its heavy instrumentation and clarify that it does not evaluate privacy, safety, or effects of enabling broader deployment.

## 6. Overall score

6 — The paper offers a careful, technically plausible, and useful audit methodology with strong transparency and well-scoped empirical evidence. I am not fully convinced the conditional, same-process trace evidence and narrow sequential setting justify stronger general systems assurance claims.

## 7. Confidence

4 — The paper and appendices clearly specify the method, evidence, and limitations, but the underlying artifacts and live implementation cannot be independently verified from the PDF.

## 8. Verdict

Accept
