# Decision log

## D-0001 — Freeze the supplied baseline

- Date: 2026-08-22
- Decision: Use `baseline/` as the immutable baseline and create `baseline/MANIFEST.sha256`.
- Evidence: the baseline, manuscript, and remote-snapshot `paper.tex` and `paper.pdf` hashes match exactly.
- Rationale: This preserves a reproducible starting point while leaving the user-owned manuscript untouched.

## D-0002 — Do not promote replay validity to online deployment validity

- Date: 2026-08-22
- Decision: Treat the count-exchangeable replay theorem as a conditional/model-scoped contribution, not a certificate for arbitrary ordered online rollout streams.
- Evidence: the manuscript explicitly states the main carrier has counts rather than rollout order; the imported portfolio screen identifies the absence of real ordered rollouts as the decision-driving blocker.
- Rationale: Ordered sampling can violate the exchangeability condition.  Synthetic drift stress is useful boundary analysis but not a replacement for a real ordered trace.

## D-0003 — Make correctness and realized cost the next decisive evidence

- Date: 2026-08-22
- Alternatives considered: (a) add more replay/TV/geometry analyses; (b) only rewrite limitations; (c) collect ordered traces with gold correctness and online telemetry.
- Choice: (c), specified in `experiments/decisive_experiment_protocol.md`.
- Rationale: Existing artifacts already extensively analyze the replay mechanism.  The imported screen shows the highest information gain comes from independent ordered traces, correctness, and token/latency/cancellation measurement.  New theory cannot establish those facts.

## D-0004 — Do not run an experiment in Phase 0/1

- Date: 2026-08-22
- Decision: Register the smallest falsifiable formal protocol but do not execute or submit it.
- Rationale: Execution was not authorized.  The protocol separates preflight failure from scientific outcomes and preserves a one-read TEST design.

## D-0005 — Select the count-exchangeable replay story for revision 01

- Date: 2026-08-22
- Alternatives considered: (A) deployment-safe online stopping; (B) exact
  certificate for count-exchangeable replay; (C) robustness geometry of
  discrete stopping states.
- Choice: (B), with (C) retained as supporting appendix analysis; details and
  rejected claims are recorded in `state/story_architecture.md`.
- Evidence: supplied artifacts establish full-vote flip and mean rollout count
  only under replay. No artifact supplies chronological online traces, gold
  correctness of the returned answer, or operational token/latency/cancellation
  telemetry.
- Rationale: Option A would make the most consequential claim but is not
  evidence-backed. Option B is falsifiable by the theorem/DP artifacts and
  preserves the decision-driving limitation. Option C cannot close that gap.

## D-0006 — Make endpoint and cost proxy distinctions explicit everywhere

- Date: 2026-08-22
- Decision: revise the abstract, figure caption, contributions, experiments,
  drift discussion, limitations, conclusion, and evidence matrix to name
  `full-N` agreement/replay flip and $1-\bar k/N$ replay count reduction.
- Evidence: `evidence/claim_evidence_map.tsv` C01--C04 and issue ledger
  A11-I001--I003.
- Rationale: Full-vote agreement is neither gold correctness nor a user-utility
  guarantee, and rollout count is neither generated tokens nor latency or
  cancellation cost. Synthetic orders are a boundary stress, not observed
  deployment order.

## D-0007 — Verify ICLR 2027 rules and retain the supplied official-style files

- Date: 2026-08-22
- Decision: use the verified official ICLR 2027 author/reviewer guidance and
  copy the style, bibliography, appendix, table, and figure dependencies into
  `manuscript/` for a self-contained build.
- Evidence: official Author Guidelines and Reviewer Guidelines; supplied style
  SHA-256 parity with the downloaded official zip (`.sty`
  `797deef4…d2098ea6`, `.bst` `2d67552d…ad1844c5`).
- Rationale: the earlier source-hash equality made the remote package the
  authoritative dependency source, but a revision must build from
  `manuscript/` alone. The mandatory AI-use statement, recommended
  reproducibility statement, 9-page main-text limit, and double-blind rule are
  now tracked as official requirements rather than unknowns.

## D-0008 — Treat the full-$N$ boundary as a zero-flip threshold state

- Date: 2026-08-22
- Decision: define every reachable endpoint by $x_N=K$ and make
  $c_H(x_N,N)=0$ explicit in the theorem, proof, DP description, and
  limitations; remove the obsolete ``population absorption'' explanation.
- Evidence: `remote_snapshot/rlve_n8_r474.py` computes `cert[k][x]` as
  `num / den if den > 0 else 0` and, after the loop over `k < N`, reaches only
  `x=K` at `N`; its terminal comparison therefore contributes no binary
  pass-count replay flip, including the frozen-table zero-denominator
  convention.
- Rationale: the terminal decision has observed all binary outcomes, so it
  must agree with itself. This fixes a semantic inconsistency without changing
  any supplied result, frozen source, or experiment.

## D-0009 — Disclose Figure 1 as a generated conceptual layout with evidence-linked labels

- Date: 2026-08-22
- Decision: retain the frozen, byte-identical Figure 1 without regenerating it,
  but remove the inaccurate ``non-evidentiary'' characterization. Mark the
  caption as conceptual with artifact-checked quantitative labels, and record
  the generator/edit history, prompt fingerprints, asset hashes, and label
  cross-check in `evidence/visual_asset_provenance.json`.
- Evidence: frozen `remote_snapshot/FIG1_HISTORY.json`,
  `edit_a11_earlystop_fig1.py`, four frozen fix requests, and source result
  artifacts; the current `manuscript/fig1_earlystop.png` has the same SHA-256
  `efffd18…295a32` as the frozen selected PNG.
- Rationale: the generated composition is explanatory, but its panel-(c)
labels summarize reported values. Their provenance must be auditable and
they cannot be presented as either figure-originated evidence or a wholly
non-evidentiary decoration.

## D-0010 — Round-2 boundary-first repair

- Date: 2026-08-22
- Decision: Repair Lemma 1 by separating the zero-probability $F_0$ atom
  before any ratio division; retain the ratio only on positive support.
- Verification: `manuscript/audit_monotonicity_boundary.py` uses exact rational
  arithmetic for all $2\le N\le64$, including $(32,17,29)$, and checks the
  atom identity, support split, and monotonicity inequality.
- Claim boundary: Cross-shard fitted-TV calculations are deterministic
  fixed-rule sensitivity over an assumed ball, not confidence statements for
  an unknown population. Margin thresholds lack a frozen selection record and
  are exploratory/descriptive.
- Provenance: bundle a snapshot-local audit input set and canonical root-hash
  specification, but retain raw split/duplicate/selection/environment absence
  as a residual external-replay blocker. Add Figure 2 renderer/result hashes;
  do not claim clean regeneration without its missing environment.

## D-0011 — Redraw Figure 2 from the frozen JSON and put the scope in the image

- Date: 2026-08-22
- Decision: Replace the inherited Figure 2 raster with a publication-styled
  redraw produced only from the frozen 399-point-per-carrier result JSON.  The
  renderer refuses an input whose SHA-256 differs from the registered hash.
- Style: Apply the local `figures4papers` scientific-figure conventions:
  sans-serif type, redundant color/line encodings, open top/right spines,
  minimal axis marks, frameless legend, tight layout, and 300-dpi export.
- Claim boundary: Put ``Fixed-rule sensitivity over an assumed TV ball'' and
  ``not target-population containment'' directly in the image, caption, and
  Limitations.  This redraw is deterministic presentation work, not a new
  experiment or population-transfer certificate.
- Verification: Snapshot-local claim audit 29/29 PASS; controlled builds are
  byte-identical at PDF SHA-256 `a1aa3b0e...d967b3`; PDF pages 9 and 15 and the
  raw figure were visually inspected.

## D-0012 — Withdraw Appendix E(g) as a policy repair and correct its TV geometry

- Date: 2026-08-22
- Decision: Retain the certificate-ordered per-final-count calculation only as
  an oracle K-wise profile-capping sensitivity diagnostic. Withdraw all claims
  that it is executable, deployable, a prefix stopping policy, a Theorem-1
  instance, or a policy-level repair; withdraw its 70-of-72, universal-budget,
  extra-vote/cost, domination, edge-law, crossing-set, and full-simplex
  interpretations. Delete Proposition 2's globally affine TV identity and its
  base-driven crossing conclusion.
- Evidence: `review/round_02/adjudication/appendix_repair.json` confirms that
  membership is selected separately for the unobserved final count $K$ and
  gives a counterexample to the all-radius affine TV formula. It also confirms
  that Theorem 1 depends instead on the conditional certificate, tower
  property, terminal convention, and prefix measurability.
- Replacement statement: the general capacity-constrained TV LP is
  piecewise linear. Its valid endpoint fact is $B_1$ equals the simplex, hence
  $V(1)=\max_K g(K)$ and a fixed profile has full-radius validity exactly when
  this maximum is at most $\alpha$.
- Rationale: a per-$K$ family $\{S_K\}$ cannot decide at a prefix, where $K$
  is unobserved. An executable claim would require one K-independent
  $S(x,k)$, a mechanical prefix-measurability test, and recomputation of every
  profile and capacity-aware LP. No such reconstruction or experiment is
  authorized or reported here.

## D-0013 — State Theorem 2's sampling and split conditions at the theorem site

- Date: 2026-08-22
- Decision: Make Theorem 2 conditional on a FIT-frozen finite rule family and
  explicit independent i.i.d. FIT/CAL task sampling. Treat its
  empirical-Bernstein plus Bonferroni statement as a CAL selection guarantee;
  label the fixed benchmark TEST split as one-time descriptive readout.
- Rationale: the previous wording could be read as an unconditional population
  assertion about a rule informed by CAL or as a TEST confirmation result.
  This is a theorem/protocol scope clarification; it adds no data or result.

## D-0014 — Remove withdrawn Appendix history from reviewer-facing source

- Date: 2026-08-22
- Decision: Delete the complete historical `\\iffalse...\\fi` block containing
  the withdrawn per-$K$ policy and globally affine TV claims. Preserve its exact
  history only in the immutable Round-2 snapshot and adjudication record.
- Additional hardening: Expand the $B_1$ endpoint into the explicit two-step
  simplex proof: any two simplex points have $\ell_1$ distance at most $2$, and
  a linear functional attains its maximum at a simplex vertex.
- Verification: post-cleanup claim audit 36/36 PASS; exact-rational monotonicity
  audit 21,824/21,824 PASS; two isolated final builds are byte-identical at PDF
  SHA-256 `d957fde4...5d931`; pages 5, 16, and 17 pass visual inspection; the
  independent post-cleanup verifier marks the requested scope resolved.
- Rationale: frozen checkpoints already provide traceability. Keeping invalid
  historical text in live submission source creates needless source-review and
  naive-search ambiguity even when LaTeX excludes it.

## D-0015 — Round-4 oracle/fitted claim separation and code-native Figure 1

- Date: 2026-08-22
- Decision: Reserve the exact conditional identity and tower theorem for the
  true law $H_\star$. Name BAYES-H's $\widehat H_{\rm FIT}$ quantity a fitted
  stopping score everywhere, and make Theorem 2 the sole operational
  guarantee: a FIT-frozen, $J=64$ empirical-Bernstein/Bonferroni CAL screen
  under independent i.i.d. FIT/CAL task sampling.
- Figure decision: replace the active generated Figure 1 with a reproducible,
  code-native Matplotlib schematic using hash-pinned frozen JSON inputs. Keep
  the old raster historical-only. Put random replay (not chronology), oracle
  versus fitted/CAL, and synthetic exploratory drift scope inside the new
  figure and caption.
- Scope decision: delete the cross-carrier TV mechanism narrative and retain
  only assumed-ball fixed-rule sensitivity; label all drift and margin rows
  synthetic, exploratory, and nonconfirmatory. No natural-shift, fallback,
  coverage, allocation-law, or deployment claim is added.

## D-0016 — Incorporate recovered replay bundle and closest-work matrix

- Date: 2026-08-22
- Decision: cite the reviewer-safe OMR manifest/replay bundle as conditional
  byte-exact main-JSON reproduction, not raw-data reconstruction. Record
  22,230 source rows, 12,423 valid rows, 11,607 deduplicated problems,
  $K=32$, split 4000/4000/3607, seed 20260815, and main JSON hash
  b114c72d...c8a5f7; state that secondary runners are preserved but not
  locally rerun.
- Positioning decision: cite Waudby-Smith and Ramdas (2020) as the closest
  finite-population without-replacement precedent, and Rossell and Müller
  (2013) plus Novikov (2010) for Bayesian sequential stopping. Limit novelty
  to independent FIT/CAL calibration of a finite frozen plug-in score family
  for exact task-level replay loss.
- Privacy decision: evidence records use only reviewer-safe relative paths and
  no private host, account, or absolute remote source location.

## D-0017 — Compact Figure 1 for page-width readability

- Date: 2026-08-22
- Decision: replace the wide three-column Figure 1 with a compact 7.8 x 4.05
  inch, two-row code-native layout. The top row separates replay-prefix and
  oracle/fitted/CAL semantics; the lower full-width row carries the frozen
  replay and synthetic-drift labels. Match the caption directions to the
  stacked `(b, top)` and `(b, middle)` layout.
- Rationale: the prior 13.8-inch canvas reduced 8--10pt source labels to
  roughly 4--5pt at manuscript text width. The compact figure's smallest raw
  label is 8.25pt, approximately 6.9pt effective at a 6.5-inch text width,
  meeting the page-width readability gate without extending the main text
  beyond Conclusion page 9.
- Verification: hash-pinned renderer and asset audit pass; frozen
  `saving_vs_full=0.8085` uses Decimal half-up display `80.9%`; two isolated
  builds are byte-identical at `fefe25a9...5dbb3`; PDF page 2 was inspected
  at 100% page rendering with no clipping, overlap, or stray mark.

## D-0018 — Round-5 boundary semantics and package-local Figure 1 repair

- Date: 2026-08-22
- Decision: Separate Lemma 1's even-$N$, $K=N/2$ hypergeometric center from
  reflection (which applies only for $K<N/2$); define the odd early budget only
  for a nonempty feasible set, with deterministic FULL-$N$ as a separate
  fallback outside the $J=64$ CAL family. Define the TV radius only when
  $V(0)\leq\alpha$ and use an explicit infeasibility sentinel otherwise. Remove
  the source/dominance interpretation of the mechanically demeaned correlation.
- Figure decision: make the active Figure 1 renderer consume only the sibling
  recovered evidence JSON. A temporary reviewer-package-shaped copy rendered
  twice byte-identically under fixed metadata before the controlled active
  regeneration; the scientific pixels were unchanged.
- Positioning decision: state Waudby-Smith--Ramdas without-replacement
  confidence-sequence majority stopping as an unevaluated direct endpoint
  comparator. Retain only model-assisted FIT/CAL screening as the implemented
  contribution and make no confidence-sequence or efficiency-separation claim.
- Verification: exact CPU audit covers 21,824 reversal cases, 528 center cases,
  and 32 no-feasible-odd/FULL fallback cases; the TV sentinel audit passes; the
  hash-bound source audit is 54 PASS / 0 FAIL. The main bundle replays
  byte-exactly conditional on its derived manifest at `b114c72d...c8a5f7`.
  Two fresh noaux builds are byte-identical at `e5d9f783...2b2c89`, have 16
  pages and Conclusion on page 9; affected pages 2, 4, 5, 6, 9, and 16 pass
  visual QA. Frozen Round-3 and Round-4 verifiers remain unchanged.
