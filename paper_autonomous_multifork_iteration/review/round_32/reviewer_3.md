## 1. Summary and claimed contributions

ForkAudit is a trace-validation protocol for hybrid KV and recurrent GDN state under prefix forking. It combines phase-indexed lifecycle, storage-range, copy-on-write, call-contract, and semantic receipts with fail-closed coverage reporting. The paper evaluates a fixed Qwen/H20 case study, selected captured-boundary numerical oracles, and targeted fault injections, while explicitly limiting claims to honest capture and declared observation points.

## 2. Strengths

The contract is unusually clear about what constitutes evidence, separating trace coverage from replay success and carefully stating the trust boundary. The treatment of recurrent mutable state, transition-time rebinding, storage overlap, and partial-page copy-on-write addresses a real gap left by output-only cache comparisons. The empirical protocol is extensive within its declared setting, combines physical ownership evidence with semantic invariance, and uses targeted controls to demonstrate that selected faults can evade token and logit equality. The paper is exceptionally candid about its limitations and largely avoids overstating its systems results.

## 3. Weaknesses

The central assurance statement depends on an honest, mandatory-event-complete capture producer. Consequently, the validation cannot rule out coherent omission, fabricated receipts, transient mutation, or kernel-level behavior not represented in the trace. The source-distinct observer still shares process, objects, labels, and storage APIs, so it does not materially solve this key independence problem.

The empirical evidence is narrow: one model stack, hardware family, prefix geometry, short continuation, and predominantly sequential single-stream execution. It does not evaluate native continuous or ragged batching, realistic concurrent scheduling, general cancellation, capacity, or diverse workloads. The numerical checks begin from producer-captured intermediates and cover selected operator boundaries, not upstream construction or end-to-end correctness. Likewise, seeded, preregistered faults show localized sensitivity but do not establish robustness to held-out or naturally occurring defects.

The work is a thoughtful integration and audit framework, but its novelty relative to existing metamorphic testing, cache lifecycle work, and systems tracing is primarily in this specific composition. The paper would be stronger with clearer evidence that the protocol yields actionable findings on real implementations beyond constructed controls, and with quantified capture overhead.

## 4. Questions for the authors

How could ForkAudit be extended with an independently trusted live observer or lower-level runtime evidence, rather than relying on the producer’s storage and lifecycle records?

What is the runtime, memory, and engineering overhead of mandatory capture, replay, and artifact retention, and which parts are practical in CI versus debugging only?

Can the protocol be demonstrated under native batched serving, broader concurrency, and in-flight cancellation, with compiled-kernel and autotuning provenance?

Do you have held-out real defects or independently developed fault suites that demonstrate detection beyond the designed mutations?

## 5. Reproducibility and ethics

The manuscript provides a detailed manifest-first replay plan, typed record schema, frozen geometry, and limitations, which are strong reproducibility practices. However, from the PDF alone the accompanying artifacts cannot be inspected; reproduction appears to require a highly pinned, resource-intensive environment, and most runtime tensors are represented by hashes rather than archived contents. The ethics discussion appropriately notes the absence of human-subject data and bounded systems scope, but should more directly address dataset licensing/copyright considerations for PG-19 and any implications of using auditing tools in production-serving environments.

## 6. Overall score

4 — Weak reject. This is careful and technically useful systems-validation work, but its central evidence remains trace-relative under a strong trust assumption, and the narrow, constructed evaluation does not yet justify broader methodological impact.

## 7. Confidence

4 — The paper is detailed and explicit about boundaries, though the underlying artifact and runtime behavior cannot be independently verified from the submission PDF.

## 8. Verdict

Reject
