# R40 post-review evidence-safe edit plan

## Contract and frozen identity

- Mode: reviewer-guided, audit-only planning. This file does not modify the
  manuscript, any direct `\input`, or the reviewed PDF.
- Frozen source inspected: `main_r40_submission_candidate.tex`, SHA-256
  `166dff9f56da4449a53857c575fbf9f62466c7bd1e84b7c05e59301ffe346c10`.
- Frozen PDF reviewed by the panel/meta-review: SHA-256
  `0906080e3d16c0f8ee071f5d3aa2f6d4d541e7f7d8a2cc3efe9e67c0d0916d5b`.
- Review authority: `review/round_40_submission_candidate_pdf_only/` (panel
  6/6/6; independent meta-review 6). Review judgments guide presentation; they
  do not create scientific evidence.
- Locations below are R40-v1 source line numbers plus stable semantic anchors.
  Use the anchors after earlier edits shift the line numbers.

The paper identity to preserve is: ForkAudit is a fail-closed, phase-indexed
ownership-trace contract for hybrid KV/recurrent state, evaluated under an
explicit honest-capture TCB on bounded fixed stacks; its demonstrated increment
is ownership/lifecycle localization beyond semantic equality, not a new cache
policy, security attestation, production scheduler, or general correctness
oracle.

## Semantic locks

The following meanings must not change during the eventual edit:

1. The primary result has seven registered targets. Any textual split of target
   5 must remain two sub-obligations under the same target unless a new protocol
   formally changes the target schema.
2. Attention evidence binds a selected fully hashed Triton kernel-cache artifact
   and compile configuration before invocation, plus the selected autotuner or
   explicit no-autotuner state, and seals only after normal return from the
   original launcher. Normal return is not device completion.
3. GDN evidence binds only the frozen mutually exclusive eager source route and
   functional cache rebind. It does not establish compiled GDN or the identity of
   underlying ATen/CUDA operators.
4. The historical experiment is one defect mechanism at three archived
   coordinates plus five additional frozen inputs, not eight independent defects
   or a natural-defect sample.
5. Tokens, full FP32 logits, terminal request GDN state, and logical KV are exact
   for the historical pre-fix/materialized comparison in all eight cells; the
   persistent-base content invariant also fails in all eight pre-fix cells. The
   repair is storage-clean and semantically exact in all eight.
6. The R28 run-ID amendment is post-execution. It changes only a generated
   inherited-RR2 comparison-row `run_id` from null to the derivation-verified
   preregistered ID. Candidate bytes, raw rank files, FP32 sidecars, outputs,
   classifications, frozen builder, and preexecution sources are unchanged.
7. Numerical rows are deterministic, preregistered captured-boundary checks, not
   random samples, statistical replicates, or evidence of input/length robustness.
8. Immutable evidence IDs and artifact names, including
   `E-R40-PRIMARY-COMPILED-DISPATCH-V11-A`, must not be renamed. Only manuscript
   shorthand and target/sub-obligation labels should be clarified.

## Evidence audit: what the current record supports

### Dispatch terminology

The current evidence supports an umbrella target called **host-side call
provenance**, with two explicitly different sub-obligations:

- **compiled-attention artifact/configuration provenance**; and
- **GDN eager-route provenance**.

It does not support calling the whole hybrid path “compiled dispatch,” calling
GDN compiled, or claiming device-side execution/completion attestation. This
clarification is editorially available now from
`E-R40-PRIMARY-COMPILED-DISPATCH-V11-A`; no new execution is needed.

### Historical alias and conventional invariant

`E-R35-HISTORICAL-ALIAS-REGRESSION-A` and its preregistered comparison matrix
support the following ordering:

1. output and terminal-state differentials miss this one alias;
2. a conventional within-lane persistent-base content invariant catches it in
   all eight frozen cells; and
3. ForkAudit's supported increment is the registered first transition-binding
   failure with owner/layer/family localization and unified fail-closed accounting.

The included `tables/r35_historical_alias_table.tex` already states this safely:
its caption calls “Base inv.” a conventional state-invariant catch, and the
pre-fix rows report `fail 3/3` and `fail 5/5`. It should be treated as the
wording authority; no table edit is required for this issue.

### The 0.005 tolerance and row selection

There is enough evidence to explain the protocol, but not to claim a statistical
or theoretical calibration:

- **Primary attention 0.005:** the release manifest labels it an
  independently pre-fixed engineering tolerance. A frozen 80-row prior FP32
  diagnostic is explicitly contextual only and did not select or tune the
  threshold. Its maximum relative-L2 was `0.001977153355255723`, so the fixed
  threshold is `2.528888306366759` times that prior maximum. The prior rows use a
  disjoint 1,025-token geometry; the primary RR2 rows use 4,095 tokens.
- **Primary attention selection:** before candidate output, rank `r` was bound
  to layer `4r+3` and generation round `r`, for `r=0,...,7`, in the unique
  `N=1`, shared-document-KV/borrowed-GDN witness arm. This is deterministic
  coverage across eight rank/layer/round coordinates, not random sampling.
- **GDN 0.005 output tolerance:** the four-row GDN oracle preregistered output
  relative-L2 `0.005` and state relative-L2 `1e-4`, and the expanded sweep reused
  those tolerances without tuning. The inspected evidence does **not** provide a
  matching 80-row margin audit or a theoretical derivation for the GDN `0.005`.
  Do not transfer the attention margin rationale to GDN.
- **GDN selection:** the original oracle uses one deterministic rank-0 window,
  its first 128 document tokens, the first four query tokens, and layers
  `0,10,20,38`. The record freezes these rows but does not establish that they are
  representative of all 30 GDN layers or other lengths/inputs.
- **Expanded sweep:** a later preregistered sweep uses two deterministic windows
  and fixed query-bank rows, all ten attention layers, and 12/30 GDN layers,
  covering 160 attention positions and 192 GDN transitions. It strengthens
  bounded layer/input coverage but remains deterministic captured-boundary
  evidence, not robustness or a population sample.

Therefore the manuscript may explain “pre-fixed engineering threshold,
contextual margin, deterministic selection, and explicit coverage denominator”
now. A calibrated error model, representative sampling claim, length robustness,
or a cross-stack tolerance claim must wait for new evidence.

### Evidence anchors used for this plan

- Dispatch: `evidence/experiment_registry.json`, entry
  `E-R40-PRIMARY-COMPILED-DISPATCH-V11-A`.
- Historical case: `evidence/r35_historical_alias_regression/design_decision.md`,
  `evidence/r35_historical_alias_regression/formal_h20/r35-historical-alias-20260826a/aggregate.json`,
  and `tables/r35_historical_alias_table.tex`.
- Run-ID correction:
  `evidence/r28_full_detector_matrix/postexecution-rr2-run-binding-correction.json`
  and
  `evidence/r28_full_detector_matrix/formal_run_20260824a/postexecution-correction-replay-receipt.json`.
- Primary attention threshold/selection:
  `evidence/round_04_rr2_package/upstream/preregistration/release-manifest.json`,
  its bound `prior-fp32-context-manifest.json`, and the frozen
  `review-experiment-response-plan.json` under the primary package's
  `executed_inputs/` tree.
- GDN and expanded numerical checks:
  `evidence/gdn_transition_oracle_preregistered_20260819d/preregistration.json`
  and `evidence/r30_expanded_oracle_sweep/preregistration.json`.

## Prioritized revision map

| Priority | Reviewer concern | Exact R40-v1 anchors | Diagnosis | Action | Status |
|---|---|---|---|---|---|
| P0 | Attention compiled-artifact and GDN eager-route evidence are named too broadly as one “compiled dispatch” result | Abstract 52--56; teaser caption 153--156; target 5 at 260--272; protocol 382--391; Results 451--460; Discussion 584--589; Conclusion 610--612; cohort row 751; witness row 794; schema 816--833; expanded limitation 1237--1243 | Terminology can imply compiled GDN or whole-path attestation even though the prose later disclaims it | Rename the manuscript shorthand and split target 5 into 5A/5B sublabels while preserving seven targets and every boundary | **CURRENT-EVIDENCE: DO NOW** |
| P0 | Historical alias could still read as uniquely detected by ForkAudit | Abstract 57--59; contribution 124--126; worked failure 334--341; Results 531--538; Discussion 578--582; Conclusion 605--609 | The base invariant catch is appendix/result-visible but absent from the headline framing | Make the conventional base-invariant catch adjacent to the alias claim; state transition/owner/layer/family localization as ForkAudit's increment | **CURRENT-EVIDENCE: DO NOW** |
| P0 | Post-execution run-ID correction is too deep in the appendix | main Reproducibility Statement 617--624; detailed replay paragraph 634--645; target-gate detail 1191--1205; artifact-map R28 rows 1359--1360 | Governance fact is disclosed, but a reader can miss it before the artifact appendix | Add one compact main-text reproducibility sentence with timing, exact field-only scope, unchanged bytes/results, and byte-identical replay; preserve fuller appendix disclosure | **CURRENT-EVIDENCE: DO NOW** |
| P1 | The 0.005 threshold and numerical row selection need explanation | selected numerical support 343--351; audit sequence 404--420; Results 502--514; oracle selection table 1130--1150; expanded sweep table 1172--1187; limitations 1272--1282 | “Preregistered” establishes chronology, not why the threshold/rows are informative | Add the bounded engineering-threshold and deterministic-selection explanation above; explicitly deny representativeness/calibration | **CURRENT-EVIDENCE: DO NOW, bounded wording only** |
| P1 | Abstract is count-heavy and allocator evidence competes with the ownership story | Abstract 38--68 | Exact call, census, relation, transfer, and allocator counts obscure the problem–method–evidence chain | Retain only the seven-target/96-configuration scope and the all-eight historical outcome; move exact call/census/Falcon/allocator numbers to Results | **CURRENT-EVIDENCE: DO NOW** |
| P1 | Artifact appendix is path-dense | artifact section 1317--1375, especially longtable rows 1352--1374 | Long file paths consume space without improving the claim boundary; exact paths already live in the registry and claim map | Replace with a compact evidence-ID/replay-root table; retain R28 correction visibility and the existing label | **CURRENT-EVIDENCE: DO NOW** |
| P0-deferred | Reduce the live-binding TCB | threat model 213--232; Discussion 584--589; limitations 1224--1231 and 1283--1287 | Current census does not independently validate slot-ID-to-live-tensor binding | Update only after a scientifically valid, source-distinct live-binding result is registered and audited | **MUST WAIT FOR NEW RESULT** |
| P0-deferred | Claim a unique advantage over a strong conventional suite | fault results 516--538; limitations 1290--1306 | Current historical baseline supports localization, not unique detection or a head-to-head rate/cost comparison | Update only after frozen blind faults and a matched conventional-suite comparison | **MUST WAIT FOR NEW RESULT** |
| P0-deferred | Production-like scheduling and audit-on/off cost | primary schedule 141--143 and 415--425; Discussion 591--599; limitations 1214--1224 and 1298--1301 | Current evidence is sequential primary execution plus bounded two-stream intervals and local CPU replay cost | Update only after valid native batching/in-flight cancellation and matched H20 cost results | **MUST WAIT FOR NEW RESULT** |

## Directly executable edit instructions

### 1. Split target-5 naming without changing the target count

**Current-evidence action.** Use the following terminology consistently:

- umbrella: `host-side call provenance`;
- sub-obligation 5A: `compiled-attention artifact/configuration provenance`;
- sub-obligation 5B: `GDN eager-route provenance`;
- experiment shorthand in prose: `primary call-provenance rerun` or
  `dispatch-provenance rerun`, not `compiled-dispatch rerun`.

Apply this at the following anchors:

1. Abstract 52--56: replace “formerly partial dispatch target” framing with the
   umbrella plus the two sub-obligations. Remove exact call counts here under the
   abstract-compression action; retain them in Results.
2. Introduction 89--97 and contribution item 109--115: name both sub-obligations
   in parallel. Do not let “selected attention kernel-cache artifact” appear to
   modify GDN.
3. Teaser caption 153--156: change “fresh compiled-dispatch rerun” to “fresh
   call-provenance rerun.” The figure asset itself should be inspected before any
   later regeneration; do not silently edit the reviewed bitmap.
4. Contract target 5, lines 260--272: keep one enumerated target, retitle it
   `Host-side call provenance`, and introduce inline `(5A)` and `(5B)` clauses.
   Preserve the target's existing common callable, query/mask/position/append,
   no-fallback, and no-full-KV-concatenation contracts before those clauses.
   End with the existing negative boundary: no device completion, compiled GDN,
   ATen/CUDA, or driver/device attestation.
5. Minimal decision rule 320--325, protocol geometry 382--391, Results 451--460,
   Discussion 584--589, Conclusion 610--612, Reproducibility 649--657, cohort row
   751, witness row 794, dependency schema 816--833, closest-precedent row 976,
   and expanded limitation 1237--1243: normalize to the same umbrella/5A/5B
   vocabulary.
6. Artifact map/registry citations: preserve the immutable identifier
   `E-R40-PRIMARY-COMPILED-DISPATCH-V11-A`; describe it as the evidence record
   for both bounded sub-obligations rather than renaming it.

**Must wait.** Do not remove the “host-side,” “normal return,” honest-process,
fixed-stack, or TCB qualifiers unless a new registered result directly closes
them. A positive live-binding experiment would be a new supporting result; it
would not turn normal launcher return into device completion or GDN eager routing
into compiled-GDN attestation.

### 2. Reframe the historical alias around shared detection and finer localization

**Current-evidence action.** Use the following claim order everywhere the result
is summarized:

> The pre-fix path preserves registered outputs and terminal request state while
> corrupting the persistent base in all eight frozen cells. A conventional
> persistent-base content invariant also catches all eight. ForkAudit's supported
> increment is the registered first transition-binding failure with
> owner/layer/family localization; the repaired path is storage-clean and exact.

This is a content specification, not text already inserted into the manuscript.
Apply the logic at:

1. Abstract 57--59: add the base-invariant catch and localization increment;
   keep “one historical alias” and “all eight frozen cells,” not “eight bugs.”
2. Contribution item 116--128: place the conventional catch immediately after
   the historical semantic miss, before the allocator result.
3. Worked ownership failure 334--341: state explicitly that semantic equality
   misses the case, the base invariant catches it, and ForkAudit localizes the
   registered transition relation.
4. Results 531--538: lead with the comparison, then the 8/8 evidence, then the
   bounded interpretation. Preserve the sentence that this is one known
   mechanism and not population coverage.
5. Discussion 578--582 and Conclusion 605--609: replace any implication that
   “the audit exposes” means exclusive detection. Use “localizes” or “records the
   first failing transition binding.”
6. Expanded limitation 1308--1315: retain as the strongest existing scope
   sentence; use it as a consistency check after editing earlier sections.
7. `tables/r35_historical_alias_table.tex`: no change needed. Preserve its
   archived/additional split and conventional-invariant caption.

**Must wait.** Unique-detection, comparative false-positive/false-negative,
maintenance-cost, or broad conventional-suite superiority claims require the
planned blind matched baseline experiment. The current eight cells cannot
support those claims.

### 3. Promote the run-ID correction into the main reproducibility statement

**Current-evidence action.** Insert one sentence in the main Reproducibility
Statement after lines 619--622, before the appendix pointer. It must include all
of these facts:

- the correction was created after candidate execution, after all 18 cases had
  been persisted and strict CPU aggregation encountered the field-source error;
- the detached manifest intentionally lacked a top-level `run_id`, whereas all
  eight manifest-bound RR2 shards carried the same derivation-verifiable
  preregistered receipt;
- the wrapper changes only generated comparison-row `run_id` from null to that
  verified value; and
- candidate outputs/bytes, raw artifacts, classifications, frozen source, and
  preexecution material are unchanged, and replay reproduces the corrected
  summary byte-for-byte.

For main-text economy, the sentence may say “after all candidate cases were
persisted” and leave the exact 18-case count to the appendix, but it must retain
`post-execution`, `run_id only`, `candidate bytes/results unchanged`, and
`byte-identical replay`.

Then:

1. Add a cross-reference from the target-gate-suppression paragraph at
   516--521 or from the main Reproducibility Statement to the detailed appendix
   disclosure at 1191--1205.
2. Keep the detailed appendix paragraph, but remove repeated path prose after
   the artifact-map compression.
3. In the compact artifact table, retain a visible row for
   `E-R28-FULL-DETECTOR-MATRIX` with the note “disclosed post-execution
   comparison-row ID correction; byte-replayed.”

**Must wait.** Do not reclassify this as a preexecution amendment or erase the
disclosure. Only a new valid execution whose canonical aggregation does not use
the correction could supersede this governance note; it would not rewrite the
history of the current R28 evidence.

### 4. Explain tolerance and sampling at the correct evidentiary strength

**Current-evidence action.** Add one short protocol paragraph, preferably after
Selected numerical support (343--351) or Controls, schedules, and oracles
(413--425), with four distinct statements:

1. primary attention `0.005` is a preregistered, independently pre-fixed
   engineering tolerance, not a confidence bound;
2. the frozen 80-row prior attention diagnostic is contextual only, did not tune
   the threshold, and had maximum relative-L2 `0.001977` (about a `2.53x` margin)
   on a disjoint 1,025-token geometry;
3. the primary eight attention rows follow the frozen rank/layer/round map rather
   than random sampling; and
4. GDN thresholds and rows were preregistered, but no equivalent empirical or
   theoretical calibration was found, so their authority is the frozen
   engineering decision plus bounded wrong-operator controls.

Use the oracle tables to expose the denominator rather than adding more prose:

- Table `tab:oracle-selection`, lines 1139--1150: retain the exact primary
  attention map and GDN layer/window selection; add “deterministic, not
  representative” in its caption or boundary column.
- Table `tab:r30-expanded-oracle`, lines 1172--1187: retain two inputs, all 10
  attention layers, 12/30 GDN layers, 160 positions, and 192 transitions; state
  that this expanded sweep reused, rather than calibrated, thresholds.
- Results 504--514: report observed maxima and rejected controls, but do not call
  the tolerance “validated,” “calibrated,” or “statistically justified.”
- Expanded limitations 1272--1282: keep the no-stochastic-replication,
  captured-boundary, length/input, and partial-GDN-layer limitations adjacent.

**Must wait.** Claims of tolerance optimality, false-accept/false-reject rates,
statistical representativeness, length robustness, all-GDN-layer coverage, or
cross-stack numerical portability require new preregistered calibration and
sampling evidence.

### 5. Reduce abstract count density

**Current-evidence action.** Rewrite only the result half of the abstract
(48--68) around one claim thread:

`seven-target bounded pass -> 96-cell ownership evidence -> distinct 5A/5B call
provenance -> historical alias/base-invariant/localization -> bounded supporting
cohorts -> TCB limitations`.

Keep in the abstract:

- seven targets;
- the 96-configuration primary scope; and
- the all-eight historical outcome, expressed once.

Move out of the abstract, without deleting from Results/Reproducibility:

- `209,920` attention and `635,520` GDN call counts;
- `1,080` observations and `96,660` relations;
- the detailed Falcon token/logit/state/ownership count list; and
- allocator endpoints `4.901 -> 2.229 GiB` and `54.5%`.

The allocator result remains valid in Introduction 126--128, Results 477--486,
Conclusion 612--614, and Appendix Table `tab:rr2-memory`; removing it from the
abstract corrects emphasis rather than narrowing or discarding evidence. Do not
remove the honest-capture, slot-binding, IPC, dispatch, compiled-GDN, or
device/driver qualifiers to save words.

**Must wait.** Do not replace removed counts with pending live-binding, blind
fault, or H20 overhead outcomes until those runs are scientifically valid,
registered, integrated, and independently checked.

### 6. Compress the artifact-path appendix

**Current-evidence action.** Replace the long path inventory at 1331--1375 with
a compact table that preserves `\label{tab:artifact-map}` and uses three columns:

`claim family | stable evidence ID(s) | replay root / boundary`.

Recommended row groups, using existing IDs only:

1. Core ownership factorial and seven-target closure:
   `E-RR2-OWNERSHIP-MUTANTS` + `E-R40-PRIMARY-COMPILED-DISPATCH-V11-A`;
   roots `evidence/round_04_rr2_package/` and
   `evidence/r40_compiled_v11_postrun_audit_mirror/`.
2. Lifecycle/scheduler extensions:
   `E-RR4-LIFECYCLE-TRANSFER` + `E-R23-A2-SCHEDULER-INTERLEAVE-FORMAL-A` +
   `E-R29-TRUE-CONCURRENT-LIFECYCLE-A`; roots
   `evidence/lifecycle_transfer_reviewer_package/`,
   `evidence/round23_a2_scheduler_interleave/`, and
   `evidence/r29_true_concurrency/`.
3. Designed and historical fault evidence:
   `E-R28-FULL-DETECTOR-MATRIX` +
   `E-R33-PDF-ONLY-FRESH-HELDOUT-FAULTS-B` +
   `E-R35-HISTORICAL-ALIAS-REGRESSION-A`; roots
   `evidence/r28_full_detector_matrix/`, `evidence/r33_fresh_faults/`, and
   `evidence/r35_historical_alias_regression/`.
4. Captured-boundary numerical oracles:
   `E-R6-FULLY-PREREGISTERED-GDN-ORACLE` +
   `E-R30-EXPANDED-CAPTURED-BOUNDARY-ORACLE-A`; roots
   `evidence/gdn_transition_oracle_preregistered_20260819d/` and
   `evidence/r30_expanded_oracle_sweep/`.
5. Out-of-process capture, census, and repeat:
   `E-R33-OUT-OF-PROCESS-GDN-CAPTURE-A` +
   `E-R39-INDEPENDENT-SLOT-CENSUS-A` +
   `E-R39-DUAL-PRODUCER-REPEAT-A`; roots
   `evidence/r33_independent_capture/`,
   `evidence/r39_independent_slot_census/`, and
   `evidence/r39_dual_producer_repeat/`.
6. Bounded second-model transfer: `E-R39-FALCON-H1-TRANSFER-V2-A`; root
   `evidence/r39_falcon_h1_transfer_v2/`.
7. Deployment/serving context: `E-MAC-M4-COMMON-MODE-CONTROL`,
   `E-H20-DEPLOYMENT-CONTEXT`, `E-RW-VLLM-PREFIX-SAME-PROTOCOL`,
   `E-RW-SGLANG-RADIX-SAME-PROTOCOL`,
   `E-R25-HYPIC-SAME-PROTOCOL-FORMAL-C`,
   `E-R34-HYPIC-STORE-EXTERNAL-ACCEPTANCE`,
   `E-R24-HYDRAGEN-QWEN35-TRANSFER-FORMAL-A`,
   `E-R24-PALU-WHITEN-QWEN35-TRANSFER-FORMAL-B`, and
   `E-RW-MARCONI-POLICY-TRACE-FORMAL-A`; group their roots under
   `evidence/mac_m4_motivation/`, `evidence/h20_deployment_benchmark/`,
   `evidence/related_work_same_protocol/`, and
   `evidence/round24_related_work_transfer/`, and label the row “context, not
   core validation.”
8. Local replay/storage cost: `E-R40-LOCAL-REPLAY-STORAGE-COST-V1-A`; root
   `evidence/r40_ci_cost_accounting_v1/`.

Precede the table with one sentence: exact artifact members, hashes, replay
commands, and prohibited expansions are authoritative in
`evidence/experiment_registry.json`; claim-level links are in
`evidence/claim_evidence_map.tsv`. Keep one replay root per group only. Preserve
the R28 post-execution correction note and the dispatch/live-binding boundaries
in the compact boundary column.

Do not claim that `supplement_r40_candidate/` is a self-contained full raw replay
package; the R40 revision record explicitly says it is only an anonymous,
checksum-bound candidate index.

**Must wait.** A pending result receives a table row only after it has a unique
active evidence ID, verified hashes, an explicit replay boundary, and a
claim-evidence-map entry. Never insert a HOLD package as completed evidence.

## Direct-include audit and planned touch set

All direct `\input` files were inspected. For the six requested issues, the
smallest safe touch set is the main source only; no included table currently
requires a factual correction:

- `tables/r35_historical_alias_table.tex`: already exposes the conventional base
  invariant and the archived/additional split; preserve.
- `tables/rr2_mutant_table.tex`, `tables/first_gate_localization_table.tex`, and
  `tables/r33_fresh_heldout_table.tex`: retain fixed-case/non-rate boundaries.
- `tables/h20_deployment_table_r40_candidate.tex` and other serving/context
  tables: remain appendix-only and explicitly illustrative/unpooled; do not
  promote them while reducing abstract allocator prominence.
- `tables/rr2_memory_table.tex`: retain the allocator denominator and endpoints;
  abstract compression does not alter this result.
- `math_commands.tex`: no edit relevant to this plan.

If later wording changes require a table-caption edit, make it in a new
post-review candidate and record it explicitly; do not mutate the reviewed
R40-v1 source/PDF in place.

## Execution order for the eventual new candidate

1. Copy R40-v1 to a new post-review source; preserve the reviewed source and PDF
   byte-for-byte.
2. Apply target-5 terminology normalization first, because it affects abstract,
   contract, results, limitations, and tables.
3. Reframe the historical alias and then compress the abstract, using the
   historical table/expanded limitation as semantic locks.
4. Add the main run-ID disclosure and tolerance/sampling explanation.
5. Compress the artifact map last, after all active evidence IDs for that
   candidate are known.
6. If any new experiment finishes before the edit, integrate it as a separate
   evidence-led change only after validity, registry, claim-map, and boundary
   checks; do not silently fold it into the current-evidence edits above.

## Verification gates after editing

- Semantic diff: seven targets remain seven; target 5A/5B remain sub-obligations.
- Forbidden-language scan: no whole-path compiled claim, compiled GDN, device
  completion, malicious-runtime resistance, production scheduler, broad
  detection rate, or cross-stack generality.
- Historical-case scan: every headline occurrence pairs the base-invariant catch
  with ForkAudit localization and says one mechanism/eight coordinates or cells.
- Governance scan: `post-execution`, `run_id only`, unchanged candidate
  bytes/results, and byte-identical replay are all visible before References.
- Numerical scan: attention contextual margin is not attributed to GDN;
  deterministic rows are not called representative; every tolerance and count is
  unchanged.
- Artifact scan: every compact-table evidence ID exists uniquely in
  `experiment_registry.json`; no HOLD item appears as completed evidence.
- LaTeX/static checks: labels, references, citations, equations, environments,
  and direct includes remain intact; `tab:artifact-map` still resolves.
- Build/visual QA: compile the new candidate with shell escape disabled, then
  inspect page 1, the main Reproducibility Statement, oracle tables, and the
  shortened artifact appendix for overflow or unreadable text.
- Fresh review: review only the newly hashed PDF. The existing 6/6/6 panel and
  meta-review remain attached to the frozen R40-v1 PDF and must not be presented
  as reviews of the edited candidate.
