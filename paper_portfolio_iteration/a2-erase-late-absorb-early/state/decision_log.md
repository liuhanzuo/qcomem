# Decision log

## D-0001 — Freeze the public baseline separately from the remote candidate

- Date: 2026-08-22
- Evidence: public `paper.tex` SHA-256 is
  `d53d4efb6476df3ed35475a881d259f0e5cdd9e9acfa6abc165437682a825d72`;
  remote `paper.tex` is
  `e9d9a84ccd3072c3bc9eadcb462f1070a1c84305494bce7e5e6296598d0e3b3b`.
- Decision: Public claims will be mapped only to the public baseline. Remote
  outputs are labelled `remote_only` and cannot close public evidence issues.
- Reason: The remote diff materially broadens/narrows theorem and empirical
  wording. Mixing versions would make result provenance false.

## D-0002 — Treat the portfolio screen as triage, not as a project review round

- Date: 2026-08-22
- Evidence: the screen explicitly says that it is a portfolio triage, not a
  five-reviewer full panel; it reports `6/6/4`, meta `6` for this project.
- Decision: preserve its substantive objections in the issue ledger but do not
  manufacture reviewer JSON, medians, or a selected checkpoint.

## D-0003 — Choose a deployment discrimination experiment as the first
## decision-changing work

- Alternatives considered:
  1. Polish the current controlled-theory story. Low cost, but cannot establish
     deployability or resolve the screen's central objection.
  2. Import remote EQRA-loss results. Apparently cheap, but prohibited by the
     source/version mismatch and still fails on the remote's harmless MNIST
     precision check.
  3. Freeze one observable, label-free-at-decision-time enable signal and test
     it with matched paths in natural human-noise and benign-hard clean-data
     scenarios. Selected.
- Reason: option 3 directly falsifies the practical claim and can produce a
  useful negative result without overstating the theory. See the protocol.

## D-0004 — Preserve the theory boundary

- Date: 2026-08-22
- Decision: Any future draft must state T1 as fixed quadratic, T2 as
  strong-convex full-norm return with its stated directional precondition, and
  T3 as the stated affine conditional-mean recursion. Logistic, shuffled-SGD,
  image, and nonconvex results are transfer tests, not automatic theorem
  corollaries.
- Reason: this is both the baseline's own limitation and a review-critical
  distinction.

## D-0005 — Select the bounded-theory story and withdraw unsupported empirical prose

- Date: 2026-08-22
- Alternatives: S1 controlled-dynamics theory; S2 a general quarantine method;
  S3 recovery of remote results. The full comparison is in
  `state/story_architecture.md`.
- Decision: select S1. The manuscript asserts only T1 fixed-quadratic, T2
  stated strong-convex full-norm, and T3 stated affine conditional-mean
  conclusions. Historical numerical figures, experiments, and appendix content
  are excluded from the active revision rather than relabelled as verified.
- Reason: S2 needs PLAN-ENABLE-001 and S3 fails the `d53d...`/`e9d...` version
  reconciliation. This maximizes evidence fidelity, not apparent breadth.

## D-0006 — Reuse remote TeX dependencies only as identified build dependencies

- Date: 2026-08-22
- Decision: copy `math_commands.tex`, `ai_use_statement.tex`, bibliography,
  and official-style parity files into `manuscript/`, recording their source
  hashes. These bytes are not experiment evidence.
- Reason: the manager verified that the remote `iclr2027_conference.sty` and
  `.bst` hashes match the official 2027 style zip. Exact dependency provenance
  is sufficient for a build dependency, unlike scientific-result attribution.

## D-0007 — PLAN-ENABLE-001 remains preflight-blocked

- Date: 2026-08-22
- Decision: do not submit or run a GPU/remote job. The remote candidate lacks a
  scoped working launcher and locally available CIFAR data; its fixed fraction
  policy cannot satisfy the new decision-time restriction without redesign.
- Reason: `experiments/execution_readiness.md` identifies the missing frozen
  N/H manifests, A1 comparator, H endpoint, environment, output schema, and
  launcher. These are infrastructure gaps, not negative scientific evidence.

## D-0008 — Make the theory source active and remove inactive submission text

- Date: 2026-08-22
- Decision: remove all eight inactive conditional blocks from manuscript/paper.tex;
  preserve their provenance through the immutable pre_revision_01 checkpoint
  and evidence/historical_inactive_material.tex. Replace proof sketches with
  active proofs for T1, T2, and T3.
- Reason: inactive historical source remains visible to reviewers and obscures
  the active scope. A bounded theory paper must make its actual proof
  dependencies directly inspectable without compiling historical material.
- Verification: source scan found no inactive conditional, historical pilot
  marker, or future-transfer promise; an isolated four-pass TeX build succeeded.

## D-0009 — Select truthful narrowing rather than an unsupported comparator theorem

- Date: 2026-08-22
- Evidence: Round-1 technical and clarity reviews identify that no theorem
  defines an intervention/reference pair or bounds their difference. T1 is a
  fixed quadratic, T2 begins from an arbitrary iterate, and T3 is a scalar
  recursion. `D` is unused by all three results.
- Alternatives considered: (1) retain the path/intervention framing; (2) add a
  comparator-path theorem; (3) state the three results as separate scoped
  product/contraction results. Option 1 is unsupported. Option 2 needs a new
  formal result and would materially alter the paper. Option 3 is selected.
- Decision: remove `D`, path/intervention/unlearning identity language, and the
  clean-tail-return label from active claims. Add a compact scope matrix and
  explicitly label future empirical or comparator work as future work only.
- Reason: the revision must describe what is proved, not a broader motivating
  problem. This does not resolve the separate research-novelty ceiling.

## D-0010 — Repair T2 with the exact ambient premise used by its proof

- Date: 2026-08-22
- Evidence: `TECH-T2-001` correctly observes that revision_01's segment-level
  wording did not explicitly imply the spectral bound on the segment-average
  Hessian. The Round-1 counterexample exploits this distinction.
- Decision: T2 now assumes pointwise ambient bounds
  `mu I <= Hessian L_A(w_A^star+r(w_t-w_A^star)) <= L I` for every actual tail
  iterate and every `r in [0,1]`. The proof records that integration preserves
  the Loewner bounds. The directional conclusion also prints `lambda_e>0`.
- Verification: `review/revision_02_t2_scope_audit.md` checks the integral,
  norm argument, and exclusion of the former counterexample.

## D-0011 — Preserve the novelty ceiling and reconcile citation status

- Date: 2026-08-22
- Evidence: the Round-1 meta-review finds no theorem-level distinction from
  direct prior theory in the frozen record; `closest_work_matrix.md` also used
  stale pending-lock language that conflicted with `citation_lock.json`.
- Decision: make no individual, joint, or unifying novelty claim. Reconcile
  every listed source to the citation lock and explicitly state that the
  already locked sources do not provide a verified theorem-level antecedent
  comparison for all T1--T3.
- Consequence: the current manuscript is a scoped exposition, not an
  established research advance. The novelty blocker stays open until a
  source-verified distinction, a nontrivial common theorem, or new evidence is
  supplied.

## D-0012 — Replace inherited build assertion with a controlled revision_02 chain

- Date: 2026-08-22
- Evidence: `R1-PROV-001` directly verifies that revision_01 recorded the
  wrong `ai_use_statement.tex` hash and lacked a sufficiently pinned toolchain
  to bind its source to its archived PDF.
- Decision: build revision_02 from fresh clean copies with a fixed
  `SOURCE_DATE_EPOCH`, record the exact TeX Live/pdfTeX/BibTeX/latexmk
  versions and every consumed input hash, and accept the output only if a
  second clean copy yields the same PDF SHA-256. The new PDF becomes the
  revision_02 archived artifact; the revision_01 PDF remains frozen history.

## D-0013 — Accept the revision_02 controlled build and retain the residual blocker

- Date: 2026-08-22
- Verification: two fresh input-only source copies built with TeX Live 2026,
  pdfTeX 1.40.29, BibTeX 0.99e, `TZ=UTC`, `LC_ALL=C`, and
  `SOURCE_DATE_EPOCH=1789948800` both produced
  `5a3f1b781e38e790660b3a64356c45d3aed27bcd699b414a7735302369cad11e`.
  This matches `manuscript/paper.pdf`. All eight consumed inputs, including
  `ai_use_statement.tex`, are hashed in `build/build_record.json`; the final
  log has no undefined citation/reference or overfull-box warning.
- Visual verification: all five pages were rendered at 144 dpi and inspected.
  The title, scope matrix, theorem/proof blocks, future-work statement,
  references, appendix checklist, and AI-use statement are legible and
  unclipped.
- Decision: close the Round-1 build provenance and citation-status issues for
  revision_02. Do not treat this as closure of `NOV-001`: reproducible scope
  correction does not create a demonstrated research contribution.

## D-0014 — Stop the current-evidence ICLR rescue path and seal the final checkpoint

- Date: 2026-08-22
- Evidence: the frozen Round-2 panel records `[4, 2, 2, 2, 4]`, median `2`,
  lower quartile `2`, and Contribution median `1`; its independent meta-review
  records meta-score and current-evidence ceiling `2`. The later
  `novelty_feasibility_round_03.json` audit independently concludes that T1--T3
  share only a trivial time-ordered affine-recurrence presentation and do not
  establish ICLR-level novelty. The current PDF
  `manuscript/paper.pdf` and both controlled clean builds have SHA-256
  `5a3f1b781e38e790660b3a64356c45d3aed27bcd699b414a7735302369cad11e`.
- Decision: select revision_02 as the final best verified artifact, not as an
  ICLR-submission candidate. Stop manuscript-only and cosmetic iteration; keep
  both frozen review rounds unchanged.
- Resume condition: reopen only with either (a) a genuinely new U3 matched
  two-trajectory/time-ordered data-block perturbation theorem, including its
  estimand, noncommutation or meaningful restriction, counterexample/falsifier,
  and a primary-source antecedent matrix showing a material residual novelty;
  or (b) a new claim-linked reproducible experiment with frozen same-version
  code, data/splits, configuration, matched comparators, seeds, outputs, and a
  decision-time information audit. A U0/U1 restatement, generic U2 tail
  contraction, historical/remote-only material, or further wording changes do
  not satisfy this condition.
