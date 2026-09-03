# EXP-ORDERED-CORRECTNESS-COST — minimal formal protocol

Status: **planned, not authorized to run**.  This protocol is deliberately narrower than a large sweep.  It tests the three missing facts that could falsify the paper's practical story: chronological online validity, task correctness, and actual cost.

## Question and hypotheses

For a newly captured model--task carrier with a deterministic gold evaluator, does a frozen BAYES-H policy return an answer with no more than 2 percentage points lower gold correctness than a full-`N` majority policy while lowering measured completed output-token cost and end-to-end latency?

Primary endpoint is **gold correctness of the returned answer**.  Prefix/full-vote agreement is secondary diagnostic only.  The study must not describe agreement as correctness, safety for user utility, or a replacement for the primary endpoint.

**Terminology boundary.** In the current count-exchangeable replay manuscript,
`full-N` denotes the binary pass-count decision $\mathbf{1}\{K>N/2\}$, not a
delivered-answer majority. The future `FULL-N` episode below is deliberately a
different, prospective policy that would return an actual answer only after a
new ordered carrier and answer aggregator are frozen; its result must not be
retroactively imputed from the count-only replay artifacts.

The result falsifies the proposed practical claim if any one of the following occurs on TEST:

1. the one-sided 95% lower confidence bound for `accuracy(BAYES-H) - accuracy(FULL-N)` is below `-0.02`;
2. the 95% confidence interval for mean completed output-token reduction does not exclude zero in the favorable direction;
3. the prespecified online trace audit finds a policy decision that used a future rollout, an unlogged retry/order change, or a missing cancellation record;
4. cancellation telemetry reveals that token completion after stop erases the claimed cost reduction; or
5. any required evaluator, split, model revision, decoding configuration, or trace field is missing or changed after CAL selection.

No result is automatically a success merely because full-vote agreement is high.

## Frozen design before data collection

1. **Carrier and task eligibility.** Select one model--task pair that has a deterministic/verifiable gold evaluator, stable prompt rendering, permission to retain raw outputs, and a fixed model/version identifier.  Record provider/model revision, task dataset version/license, evaluator hash, prompt-template hash, tool versions, region, and date.  Unknown fields block execution; do not silently substitute a count-only carrier.
2. **Task allocation.** Before any rollouts, hash unique task IDs with a published seed into disjoint FIT/CAL/TEST partitions of 3,000 tasks each.  If fewer than 9,000 eligible tasks exist, the formal run is not authorized under this protocol; register that as an infrastructure/design limitation rather than changing sample size after observing results.
3. **Maximum work.** Freeze `N=32` sequential rollout slots per episode, decoding parameters, context/window limits, retry policy (disabled unless predeclared), and concurrency policy.  Each rollout gets an immutable `(task_id, split, episode_id, rollout_index, request_id)` key.  `rollout_index` is assigned before request dispatch and records actual chronological completion and arrival times separately.
4. **FIT and CAL.** Collect complete ordered `N`-rollout traces for FIT and CAL.  Fit the count prior only on FIT.  Freeze the exact BAYES-H tables and all alpha/policy choices using CAL only, including its empirical-Bernstein/Bonferroni rule family.  Save hashes before TEST begins.
5. **TEST episodes.** For every TEST task, run two independently seeded, task-paired episodes in randomized episode order:
   - `FULL-N`: generate all 32 sequential rollouts, return the full majority answer, do not issue cancellation.
   - `BAYES-H-online`: dispatch only the next needed sequential rollout; after the frozen stopper fires, return the current majority answer and issue no further rollout requests.  If the service implementation has prefetch/in-flight work, issue cancellation immediately and log request/acknowledgement/completion fields below.

   The two episodes have independent sampling seeds but share task ID, prompt, model revision, and fixed configuration.  This deliberately measures the deployed answer rather than reconstructing a counterfactual saving from a fully completed trace.
6. **Secondary full-vote diagnostic.** Select a deterministic 10% TEST audit subset before TEST.  For this subset, execute a separately labeled shadow-completion episode to `N` after observing the frozen online decision.  It is excluded from online-cost primary analysis and is used only to measure prefix/full-vote agreement and trace-audit consistency.

## Required immutable records

Write append-only JSONL/Parquet ledgers (paths must be versioned before run):

- `task_manifest`: task ID, split, task/prompt hash, gold-evaluator version, eligibility decision.
- `rollout_trace`: task/episode/rollout keys; dispatch, first-token, final-token, response-arrival timestamps; raw output hash and retained output location; parsed answer; gold correctness; input/output/completed token counts; error/retry status.
- `stop_decision`: prefix index, prefix answers/counts, frozen certificate-table hash, certificate value, alpha, stopping decision, returned answer, decision timestamp.  No future-rollout field may be read by the policy.
- `cancellation_ledger`: in-flight request IDs at stop, cancellation dispatch timestamp, provider acknowledgement timestamp/status, final completion timestamp/status, output tokens after stop, and billable tokens when provided.  Use `unknown` when the provider does not expose a field; do not set it to zero.
- `run_manifest`: exact command, source revision/hash, environment, model revision, seeds, split hash, table hash, start/end times, logs, and output checksums.

## Prespecified analysis

1. **Correctness (primary).** For paired TEST tasks, compute task-level difference `gold_correct(BAYES-H-online) - gold_correct(FULL-N)` and a stratified-by-split, task-paired bootstrap 95% one-sided lower bound (10,000 resamples; seed frozen before analysis).  Report both accuracies, difference, interval, abstention/parser failures, and evaluator failures.
2. **Cost (co-primary operational endpoint).** Report mean and median, with paired bootstrap intervals, for completed output tokens, provider billable tokens if available, request count, `time_to_final_answer`, and `time_to_first_token`.  Separately report cancellation attempt rate, acknowledgement rate, in-flight requests at stop, post-stop completed tokens, and unknown telemetry rate.  The primary token result uses completed output tokens; billable tokens remain a separate endpoint if reliably provided.
3. **Agreement (secondary).** On the fixed shadow subset, report prefix/full-`N` agreement/flip and compare it with the frozen certificate scope.  It cannot rescue a failed correctness or cost endpoint.
4. **Online-validity audit.** Mechanically verify every stop decision references only trace rows with `response-arrival <= decision_timestamp`; verify monotone chronological indexes, table/split hashes, and no duplicate task/episode keys.  Any violation makes the affected TEST run invalid rather than a negative scientific result.
5. **Failure taxonomy.** Label failures as `infrastructure/preflight`, `data/evaluator`, `online-policy-integrity`, or `scientific-negative-result`.  Only completed, integrity-valid TEST episodes can be used as scientific evidence.

## Minimal preflight and stopping rule

Before submitting a formal job, run only focused checks: imports/compilation; one non-billed schema fixture per ledger; frozen split/table/config hashes; gold evaluator fixture; duplicate-key prevention; output path safety; and cancellation API recording.  Do not run a GPU smoke merely by default.  Once those gates pass and execution is authorized, use the last known working launcher/CLI; do not invent infrastructure.

## Deliverables and interpretation

Register the run in `evidence/experiment_registry.json` with exact command, hashes, logs, outputs, extracted metrics, and validation status.  Positive outcomes support only the named model--task carrier and frozen operating point.  A failed non-inferiority or cost test is an informative scientific negative and must remain in the registry; it requires claim narrowing, not deletion or post-hoc policy search.
