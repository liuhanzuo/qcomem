(1) Summary and claimed contributions

The paper presents ForkAudit, an offline, fail-closed trace-validation protocol for hybrid LLM inference that shares document KV state while managing mutable GDN recurrent state across forks. It records lifecycle, storage-range, copy-on-write, transition, call, and semantic receipts, then replays seven registered targets. On a fixed Qwen3.5/H20 stack, it reports complete passing evidence for six targets, partial dispatch coverage, exact cross-arm/cross-fan-out equality over 96 configurations, constructed fault localization, selected captured-boundary numerical checks, and one retrospective alias-defect reproduction and repair.

(2) Strengths

- The paper is unusually explicit about its assurance boundary. It correctly distinguishes trace-relative evidence from security attestation, independent execution, transient-write exclusion, and general fault-detection claims.
- The phase-aware storage contract is well specified. The pointer-free normalized-range formulation, mandatory-record rules, and separation of coverage from verdict make the method conceptually clear and potentially useful for regression testing.
- Empirical reporting is careful: the paper avoids pooling incompatible cohorts, identifies deterministic outcomes rather than statistical replicates, and clearly labels contextual serving measurements as non-validation evidence.
- Positive controls are more informative than output-only comparisons: the demonstrated historical alias case makes the central motivation concrete.
- Presentation, tables, appendices, limitations, reproducibility statement, and ethics statement are all strong and highly readable.

(3) Weaknesses

- The central contribution is largely a rigorous instrumentation and validation protocol rather than a broadly evaluated systems or ML advance. Its practical novelty beyond established invariant checking, metamorphic testing, storage tracing, and COW validation is not fully demonstrated.
- The main conclusion is strongly dependent on the producer/capture TCB: the same system selects slots, emits storage descriptors, and captures inputs. The process-separated observer does not independently enumerate state or validate producer semantics. This is acknowledged, but substantially limits the strength of “ownership validation” in practice.
- Evidence is limited to one fixed model, hardware family, page geometry, sequential batch-one schedule, and mostly one partial-tail/transition geometry. There is no native continuous/ragged batching, multi-document workload, production scheduler, optimized recurrent kernel, or in-flight cancellation validation.
- The fault evidence is constructed and mostly designed around the protocol’s own predicates. The designer–executor separation is useful but does not test whether ForkAudit finds realistic unseen defects or quantify false positives/false negatives. The historical case is only one defect and is also caught by a conventional persistent-base invariant.
- Numerical checks begin from producer-captured intermediate activations and cover selected rows/layers. They provide useful operator-boundary checks, but not end-to-end independent validation of the model path.
- Reproducibility is described as local replay of archived artifacts; from the PDF it remains unclear how easily an external user can reproduce capture, rerun the system, or adapt the framework to other hybrid architectures.

(4) Questions for the authors

- Can the authors demonstrate portability to at least one materially different hybrid model/runtime and a native batched serving schedule, rather than only replaying the fixed Qwen3.5 adapter?
- What concrete engineering effort is required to add a new state type or runtime backend, and which parts of the schema and predicate set remain reusable?
- Could the artifact include an independently runnable capture pipeline, not only manifest-bound replay, plus a minimal reproducer for the historical bug?
- How does the protocol behave under naturally occurring bug reports or a held-out fault suite not designed around the existing predicate vocabulary?
- What is the runtime and memory overhead of full capture and replay at realistic CI scales?

(5) Reproducibility and ethics

The paper gives detailed frozen environments, protocols, artifact paths, record schemas, and replay scopes, which is commendable. However, reproducibility of the reported verdicts appears stronger than reproducibility of fresh capture or portability to new systems; the latter should be demonstrated more directly. The ethics discussion is appropriate for a systems study using PG-19, though use of the training split and any associated dataset licensing/governance should be made explicit in released materials.

(6) Overall score—4

Weak reject. This is a careful, technically coherent, and clearly scoped validation protocol with unusually honest limitations. However, the contribution is narrowly demonstrated on one heavily instrumented stack and depends substantially on trusted producer-side capture. The empirical evidence does not yet establish broad practical utility, generality, or robust detection beyond protocol-aligned constructed controls at the level expected for acceptance.

(7) Confidence—4

The PDF provides enough methodological and empirical detail to assess the paper confidently, including extensive appendices and explicit limitations. My uncertainty concerns the practical value and generality of the proposed audit, which cannot be resolved from the PDF alone.

(8) Verdict—Reject
