# R40 conventional-baseline detector matrix protocol

## Status and chronology

This is a retrospective, read-only analysis of already archived executions.  The
detector families below were selected from the user's requested conventional
suite before the matrix generator was executed.  The agent inspected source
schemas and a subset of outcomes first to establish that every rule was
computable.  Consequently this is **not** an outcome-blinded preregistration and
must never be described as one.  The rules are frozen here before generating
`results/` and are not changed in response to the resulting catch counts.

No new model execution is performed.  No GPU or QuickSilver resource is read,
created, changed, stopped, released, or evicted.  Existing evidence is opened
read-only.  The analysis writes only under
`evidence/r40_baseline_detector_matrix_v1/`.

## Question

For the nine primary M1--M9 mutants, five designer--executor-separated faults,
one historical alias mechanism evaluated at eight frozen coordinates, and their
matched clean controls, which faults are caught by a realistic conventional
test suite, which are caught by ForkAudit, where is the first available
localization, and which catches are unique or redundant?

The cases are fixed constructed faults or repeated coordinates of one historical
defect.  Counts are case counts only.  They are not estimates of defect
prevalence, unseen-fault recall, precision, or a population detection rate.

## Frozen conventional suite

The suite intentionally uses ordinary checks that a careful unit/integration
test could implement without ForkAudit's typed record schema:

1. `B1_RUNTIME_ASSERTION`: a built-in production assertion or an unexpected
   non-assertion runtime failure rejects the case.  A deliberate fault-payload
   abort is excluded because the injected sentinel is not a detector.
2. `B2_TOKEN_EQUALITY`: generated token sequences must exactly match the
   matched clean or materialized control.
3. `B3_FULL_LOGIT_EQUALITY`: complete CPU-FP32 full-vocabulary logits must be
   byte-exact when call cardinality is comparable.  A non-comparable trace is
   recorded as `not_comparable`, not a catch.
4. `B4_STRUCTURAL_SEQUENCE_ARGUMENT`: plain assertions check sequence binding,
   canonical positions, mask representation/shape, and fixed scalar call
   arguments such as the configured attention scale.
5. `B5_PERSISTENT_BASE_IMMUTABILITY`: persistent KV/GDN content digests must be
   unchanged between the recorded setup and post-transition/final point.
6. `B6_SIMPLE_ALIAS_OVERLAP`: once a request is write-ready or completed, its
   mutable reservation/storage identifiers and byte ranges must not overlap the
   persistent base or a peer request.  Initial borrowed sharing that is allowed
   before privatization is not tested, preventing a false positive on the
   repaired borrowed control.
7. `B7_BASIC_LIFECYCLE_CARDINALITY`: tail detach must precede the first append
   write, committed call count must equal the expected count, and a rebinding
   generation/token must advance rather than remain stale.

Detector order is temporal, not an optimization over outcomes:

`pre_model_validation` < `dispatch_or_transition` <
`post_transition_state` < `terminal_token` < `terminal_logit`.

Within one stage the fixed tie-break order is B1, B4, B6, B7, B5, B2, B3.

## Two evidence-strength views

`baseline_detected` uses every applicable rule above, but each decision carries
an evidence mode:

- `executed_independent_observer`: the archive directly contains a separately
  evaluated output, production assertion, or explicitly preregistered
  conventional state-invariant result;
- `computed_from_raw_receipt`: the rule is recomputed from raw hashes, values,
  event order, counts, or storage relations in the archived case;
- `projected_from_suppressed_event`: for the primary campaign only, a plain
  assertion is projected from a target-gate-suppression event.  This is useful
  counterfactual evidence but is not an independent detector execution;
- `not_evaluated`, `not_comparable`, or `not_applicable`.

`strict_independent_baseline_detected` includes only caught decisions with
`executed_independent_observer`.  This second view prevents projected receipt
checks from being presented as an independent head-to-head experiment.

## Campaign bindings

### Primary M1--M9

- ForkAudit outcome and first gate: the all-gates-on RR2 shard referenced by
  `r28_full_detector_matrix/.../detector-matrix-v2-summary.postexec-corrected.json`.
- Conventional output/production observers and projected plain assertions: the
  separately executed R28 target-gate-suppression clean/mutant pairs.
- These are separate executions.  A matrix row must preserve that distinction.

`KERNEL_CALLABLE_ID` is deliberately excluded from B4: callable provenance is
the method-specific dispatch observer being evaluated, not a conventional
shape/sequence/argument assertion.  An M8 fault-payload abort is also excluded
under B1.

### Designer--executor faults

The five R33 clean/mutant pairs supply complete raw cases, detached pair replay,
semantic comparisons, physical-prefix digests, event order, call counts, scalar
arguments, binding-token counts, and storage rows.  Plain rules are recomputed
from those fields.  The same capture is reused; it is not an independently
instrumented baseline run.

### Historical alias regression

Each of eight R35 `historical_pre_fix` lanes is one coordinate of the same
historical defect mechanism; the corresponding `repaired_borrowed` and
`materialized_control` lanes are clean controls.  Output equality and the
preregistered persistent-base content invariant are direct baseline observers.
The post-model request/base and request/peer interval comparison is recomputed
from raw relation rows.  Initial allowed borrowed sharing is ignored.

## Case classification

- `forkaudit_unique`: ForkAudit catches and the full conventional suite does not.
- `redundant_both`: both catch.
- `baseline_only`: only the conventional suite catches.
- `neither`: neither catches.
- Clean cases are not placed in those fault categories; any catch is a false
  positive for that evaluated clean case.

First localization is the earliest caught detector under the frozen stage
order.  It is a location in the archived execution/receipt, not wall-clock
latency and not a claim that all real implementations would run detectors in
that order.

## Acceptance checks

The generator must:

- load exactly 9 primary fault rows and 9 primary clean rows;
- load exactly 5 R33 fault rows and 5 R33 clean rows;
- load 8 R35 historical defect-coordinate rows and 16 clean control rows;
- validate every RR2 shard hash referenced by the corrected R28 summary;
- reject missing, duplicate, malformed, or operationally invalid inputs;
- report all clean-case catches;
- emit per-detector decisions, first localization, strict/full baseline views,
  ForkAudit outcomes, relation labels, input SHA-256 bindings, and deterministic
  JSON/CSV summaries;
- never convert `not_evaluated` or `not_comparable` into a pass or catch.

Any failed acceptance check blocks a comparative claim.  The honest fallback is
the machine-readable blocker, not an imputed result.

