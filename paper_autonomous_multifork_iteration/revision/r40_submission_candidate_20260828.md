# R40 submission-candidate revision

## Identity

- Reviewed source baseline: `main_r39_revised.tex`
- Baseline SHA-256: `ef43415eb94a40b4a239c8f0fb8d3017ad9bc07ef75ff4895343292d92f546cb`
- Candidate source: `main_r40_submission_candidate.tex`
- Candidate-v1 source SHA-256: `166dff9f56da4449a53857c575fbf9f62466c7bd1e84b7c05e59301ffe346c10`
- Candidate-v1 PDF: `build/r40_submission_candidate_v1/main_r40_submission_candidate.pdf`
- Candidate-v1 PDF SHA-256: `0906080e3d16c0f8ee071f5d3aa2f6d4d541e7f7d8a2cc3efe9e67c0d0916d5b`
- Candidate-v1 PDF geometry: 29 pages total; main text, conclusion, and the
  reproducibility statement end on page 9, where references begin.

The reviewed R39 source is preserved byte-for-byte.  R40 is a separate
candidate rather than an in-place replacement of a reviewed checkpoint.

## Evidence-driven changes

1. Integrated `E-R40-PRIMARY-COMPILED-DISPATCH-V11-A` and changed target 5
   from partial to pass only at the declared honest-process fixed-stack scope.
   The authorized result covers 209,920 registered attention calls, 635,520
   GDN eager-route calls, eight rank replays, and 224 rejected bound negative
   controls.
2. Preserved the exact dispatch boundary: per-call attention evidence binds a
   selected fully hashed Triton kernel-cache artifact and compile configuration
   before invocation and seals after normal launcher return.  It does not
   attest device-side completion, driver/device binaries, compiled GDN,
   underlying ATen/CUDA identity, a malicious runtime, or cross-stack transfer.
3. Promoted the historical alias regression into the abstract, introduction,
   worked example, discussion, and conclusion.  The manuscript states that it
   is one defect mechanism observed at eight frozen coordinates, not eight
   independent defects.
4. Removed the eight-item Store--F1 result from the abstract, contribution
   list, teaser headline, and conclusion.  It is now an explicitly illustrative
   appendix measurement and not ForkAudit validation evidence.
5. Replaced the teaser with a deterministic R40 asset that foregrounds the
   seven-target closure, bounded dispatch counts, historical alias, census,
   and allocator denominator.
6. Moved a concise reproducibility statement before the bibliography and kept
   counts, replay scope, evidence chronology, and artifact mappings in the
   detailed appendix.

## Provenance and packaging changes

- Removed a duplicate obsolete compiled-dispatch registry entry that pointed
  to an absent independent-audit file and used overbroad launcher-artifact
  wording.
- Added the active and rejected-expansion claim rows to
  `evidence/claim_evidence_map.tsv`.
- Added receipt, finalizer, and local-integration method rows to
  `evidence/method_provenance.tsv`.
- Added `scripts/validate_r40_compiled_dispatch_integration.py`, which checks
  ten pinned products, all 46 frozen source-ledger rows, all ten minimal-mirror
  archive members, aggregate counts and verdicts, and ledger closure without
  claiming a local H20 or full-raw replay.
- Added `scripts/validate_r40_submission_candidate.py`, a read-only candidate
  audit over manuscript language, includes, registry uniqueness, claims,
  methods, figure provenance, anonymous metadata, and dispatch integrity.
- Added `supplement_r40_candidate/` as an anonymous, checksum-bound candidate
  index.  It explicitly says that it is not yet a self-contained full raw
  replay package.

## Validation

- Candidate static/evidence audit: pass.
- Compiled-dispatch local integration audit: pass for local mirror integration.
- LaTeX: direct `latexmk` build with shell escape disabled; pass.
- Final build log: zero undefined citations, zero undefined references, zero
  overfull boxes, and no fatal errors.
- Visual QA: all 29 pages rendered; contact sheets plus pages 1, 3, 9, 22, 28,
  and 29 inspected at full render size.  No clipping, overlap, broken table,
  unreadable figure, or anonymity defect was observed.
- New GPU, held-out-fault, and live-binding executions: none.  The held-out and
  live-binding staged lines remain non-evidence/HOLD.

## Blind review status

Candidate-v1 was sent to fresh PDF-only ICLR-style reviewers that did not see
the handoff, prior reviews, state, source history, or evidence registry.  The
valid panel scored the exact frozen PDF **6/6/6**, with confidence **4/4/3**
and dimension medians **3/3/2** for soundness/presentation/contribution.  All
three reviewers judged the narrow fixed-stack result marginally above threshold
and placed its current evidence ceiling at 6.  They independently converged on
three material requirements for an 8: reduce the live-binding TCB with a
source-distinct witness, compare against a strong conventional suite on blind
cases, and add production-like scheduling plus matched H20 capture cost.  The
panel record is stored at
`review/round_40_submission_candidate_pdf_only/panel_summary.md`.  One initial
third-review attempt with an unresolved rubric path is explicitly invalid and
excluded; the replacement verified the canonical rubric hash.  Reviewer
judgments are process records, not experimental evidence or acceptance
predictions.

The separate isolated meta-review also scores the candidate **6** with
confidence **4** and adopts dimension scores **3/3/2**.  It finds no supported
critical defect that invalidates the narrow score-6 case, but confirms that the
same three material experiments are required for an evidence-based score-8
case.  Round 23 therefore remains the repository-level best checkpoint under
the existing lexicographic selection rule; R40 is the latest retained candidate,
not a quality-passed final checkpoint.
