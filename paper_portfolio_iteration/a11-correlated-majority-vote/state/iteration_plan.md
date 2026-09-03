# Phase 1 to first revision plan

## Gate 0 — Preserve and verify the baseline (complete)

- Baseline manifest recorded; three paper copies hash-identical.
- Remote snapshot checker and isolated LaTeX build audited in `build/build_record.json`.

## Gate 1 — Evidence architecture (complete)

- Inventory all supplied paper, audit, result, and citation artifacts.
- Map only model-scoped claims to existing evidence; label endpoint/cost facts as missing.
- Import, rather than re-score, the portfolio initial screen.

## Gate 2 — Freeze the decisive experiment before execution (registered; execution not authorized)

1. Select an independently captured model--task carrier with gold answer/verifier and retain every rollout in generation order.
2. Freeze task IDs, prompt rendering, decoding parameters, max `N`, FIT/CAL/TEST partition, policy family, and telemetry schema.
3. Pre-register primary endpoint: gold correctness of the online returned answer.  Register full-vote agreement as secondary only.
4. Pre-register realized billable output tokens, request count, time-to-final-answer, time-to-first-token, cancellation attempts/acknowledgements, and completed-after-cancel tokens.
5. Run the smallest formal design in `experiments/decisive_experiment_protocol.md`; write a registry entry whether the result is positive, negative, or an infrastructure failure.

Revision 01 readiness audit: `experiments/execution_readiness.md` finds no
existing launcher that can directly perform the ordered + correctness + cost
protocol. The parent SSH preflight confirms a previously used RLVE raw carrier
is available for schema/answer-parsing harness work, but it lacks `rlve_eval`
integration and is not an independent confirmatory carrier. No task was
submitted.

## Gate 3 — Result interpretation and claim boundary (blocked on Gate 2)

- If online correctness is non-inferior within the preregistered margin and online cost is actually reduced, add only claims supported by those results.
- If full-vote agreement is high but correctness degrades, retain it as an explicit negative result; do not call the stopper safe for task accuracy.
- If cancellation dominates cost or latency, report it and revise the purported savings boundary.

## Gate 4 — Literature and venue audit (partially complete)

- Central closest-work citations are verified and recorded in
  `literature/closest_work_matrix.md`; background citations remain open and
  cannot pass the citation-integrity gate.
- Official ICLR 2027 author/reviewer instructions and official-style parity
  are verified. Final page, anonymity, and PDF checks remain build gates.

## Gate 4.5 — Revision 01 claim and build package (complete pending final build)

- Select the replay-scoped story and record all alternatives in
  `state/story_architecture.md`.
- Narrow all online-order, full-vote-versus-correctness, and rollout-count
  versus token/latency/cancellation language without inventing results.
- Copy style, bibliography, appendix, table, and figure dependencies into
  `manuscript/`; compile only from that directory and record hashes.
- Revision 01b aligns the theorem, terminal DP semantics, and limitations:
  every reachable $k=N$ state has $x_N=K$ and zero binary pass-count replay
  flip. It also separates the frozen-source numeric checker from a
  manuscript-hash-bound current-prose audit.
- The generated Figure 1 is now recorded as a conceptual layout with
  evidence-linked quantitative labels, including immutable source history,
  prompt/edit fingerprints, asset hash, and result-label audit; no new image
  generation is authorized or needed.

## Gate 5 — First full review round (only after an auditable revised snapshot)

- Compile and visually inspect the new manuscript, then freeze a blind snapshot.
- Run five fresh isolated reviewers and a separate meta-reviewer per the local ICLR 2026 proxy process.

## Gate 6 — Round-4 claim-scope and provenance repair (complete)

- Restrict Theorem 1 to oracle true-$H_\star$ conditional identity/tower
  theory; restrict BAYES-H's operational statement to Theorem 2's
  FIT-frozen CAL empirical-Bernstein/Bonferroni marginal replay screen.
- Record $J=64$, $\delta_r=.05/64$, FIT/CAL task-i.i.d. and independence,
  bounded task-level loss, and descriptive-only TEST status at the theorem and
  protocol sites.
- Rebuild Figure 1 with code-native, hash-pinned inputs; preserve the former
  generated raster as historical-only. The image itself carries replay-not-
  chronology, oracle-versus-fitted/CAL, and synthetic/exploratory scope.
- Integrate the recovered derived-manifest replay boundary and closest-work
  matrix. Do not claim raw-parquet clean-room reconstruction, secondary-runner
  reruns, online order, correctness, cost, coverage, natural-shift robustness,
  or a general allocation/mechanism law.
- Complete two isolated builds, static/current/frozen audits, manifest/ledger
  consistency checks, and targeted visual QA before freezing the Round-4
  submission.

Completion record: two isolated builds are byte-identical at PDF SHA-256
`fefe25a9...5dbb3`; the current claim audit is 49/49 PASS; the derived-manifest
replay reproduces `b114c72d...c8a5f7`; frozen audit/snapshot checks pass within
their stated scope; pages 2, 6, 8, 9, 14, and 16 were visually inspected. The
remaining ordered/correctness/cost/provenance limits are evidence gaps, not
unresolved source or build defects.

## Gate 7 — Round-5 formal/package repair and checkpoint selection (complete)

- Repair the even-$N$, $K=N/2$ center case; define an odd early budget only
  when its feasible set is nonempty; keep deterministic FULL-$N$ outside the
  $J=64$ CAL family.
- Distinguish an infeasible TV-radius set from a feasible critical radius of
  exactly zero, and remove latent-dependence attribution from the mechanical
  task-demeaning identity.
- Make Figure 1 package-local and deterministic; label Table 5 TEST outcomes
  descriptive and the OMR `alpha=.01` row CAL-only.
- Position the WoR-CS majority stopper as a direct, unevaluated same-endpoint
  comparator and make no unsupported efficiency-separation claim.

Completion record: independent change verification is resolved for the
selected scope; 54/54 active claims pass; formal CPU audits cover 21,824
reversal, 528 center, and 32 no-feasible-odd/FULL-fallback cases; conditional
bundle replay remains byte-exact; two fresh noaux builds produce identical
16-page PDFs at `e5d9f783...2b2c89`; affected pages pass visual QA. Round 5 is
frozen at root `b889ed62...a6b9d7` and selected as the best verified artifact.
It is unscored: the latest full-panel median/meta remain 4 and the evidence
ceiling remains 6.

## Gate 8 — Next score-moving evidence (not authorized)

1. Run the predeclared same-endpoint WoR-CS and BAYES-UNIF comparison on the
   frozen reviewer-safe manifest without TEST-driven method selection.
2. If practical online claims are desired, execute the ordered/gold/cost
   protocol with chronology and cancellation integrity.
3. Recover raw-to-manifest and secondary-runner provenance if stronger
   reproducibility or cross-carrier evidence is retained.

No further manuscript-only iteration is planned before one of these evidence
gates changes.
