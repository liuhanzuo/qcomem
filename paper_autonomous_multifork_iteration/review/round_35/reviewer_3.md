(1) Summary and claimed contributions

ForkAudit is an offline, fail-closed trace-validation protocol for hybrid LLM inference state. It records phase-indexed ownership, lifecycle, call, storage, and semantic receipts to verify that shared KV prefixes and recurrent GDN state are correctly isolated across forked requests. On one Qwen3.5/H20 stack, it reports complete passing coverage for six targets and Python-scope partial dispatch coverage; 96 ownership configurations, constructed mutations, selected numerical oracles, a process-separated observer, and one historical alias bug reproduction support the bounded claim. Reviewed source: :codex-file-citation{path="/Users/liuhanzuo/MacLLM-Bench/paper_autonomous_multifork_iteration/review/round_35/pdf_only_input/forkaudit.pdf" purpose="source"}

(2) Strengths

- The assurance boundary is unusually clear. The paper explicitly distinguishes trace coverage from verdicts and carefully limits conclusions to non-adversarial, registered observation points.
- The phase-aware ownership formulation is technically thoughtful: it covers prefix immutability, tail COW, recurrent rebinding, request-base/peer disjointness, and semantic relations rather than relying on output equality.
- Evaluation is transparent about what is and is not evidence. The constructed fault campaigns, suppression study, numerical checks, and historical defect are not overstated as population-level detection rates.
- The pointer-free storage schema and byte-range replay rules are concrete and potentially useful for systems practitioners. The paper is well written, with strong tables and appendices.

(3) Weaknesses

- The central evidence remains heavily conditional on a trusted producer that enumerates slots, emits storage identities, and captures inputs. The out-of-process observer does not independently recapture state or validate the producer’s enumeration; thus the headline “ownership validation” is narrower than its framing may initially suggest.
- Empirical support is restricted to one model, one hardware family, one page size, a sequential batch-one schedule, and a narrow set of continuation/prefix geometries. The extensions do not establish native continuous batching, general scheduling, in-flight cancellation, optimized recurrent kernels, or production serving behavior.
- Novelty appears primarily to be an integration and engineering protocol built from known ownership, metamorphic-testing, trace, and numerical-replay ideas. The paper needs a crisper argument and evidence that this integration yields capabilities unavailable from a simpler combination of immutable-base guards, storage-range checks, and existing runtime tests.
- The fault evidence is mostly designed by the authors. Even the designer-executor separation is not a held-out predicate vocabulary or a natural-bug evaluation; the sole historical defect is caught by a conventional persistent-base invariant as well. This weakens the claim that ForkAudit materially improves practical bug detection beyond more conventional invariants.
- “Exact” semantic comparisons commonly depend on candidate-produced/captured objects, while numerical oracles begin after candidate-produced boundaries. This supports consistency at those boundaries, not independent end-to-end correctness.

(4) Questions for the authors

- Can you evaluate a small corpus of independently discovered hybrid-cache bugs or regressions, comparing ForkAudit against a specified baseline suite of conventional content, ownership, and output invariants?
- What is the incremental value of the full protocol over a simpler baseline combining persistent-base digests, storage-overlap checks, and end-of-request semantic equality?
- Can the trusted producer boundary be reduced—for example, through independently generated slot inventories or capture mechanisms—and can this be demonstrated empirically?
- What are capture/replay overheads and false-positive operational costs for a realistic CI workflow?

(5) Reproducibility and ethics

The PDF provides unusually detailed replay scopes, schemas, artifact paths, versions, frozen geometries, and explicit non-reproducible boundaries, which supports replayability. However, based only on the PDF, I cannot verify artifact availability, execute the package, or assess whether a third party can reproduce live GPU capture rather than replay existing traces. Ethics concerns are limited: the work reports systems experiments using PG-19 training data and acknowledges that efficiency may scale deployment; it does not involve human subjects. The paper should additionally state licensing/handling considerations for the model and dataset artifacts used.

(6) Overall score—state exactly one allowed integer and concise justification

4. The paper is careful, technically detailed, and potentially useful, but the core contribution is an incremental, highly trusted and fixed-stack validation protocol whose practical advantage over simpler checks and ability to detect naturally occurring defects are not yet convincingly demonstrated.

(7) Confidence—state one integer 1–5 and concise justification

4. The paper clearly exposes its protocol, evidence, and limitations, enabling a confident assessment; uncertainty remains about novelty relative to systems-testing practice and artifact behavior not observable in the PDF.

(8) Verdict—Accept or Reject

Reject
