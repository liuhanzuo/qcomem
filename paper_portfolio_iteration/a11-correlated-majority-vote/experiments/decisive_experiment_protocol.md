# EXP-ORDERED-CORRECTNESS-COST — frozen minimal formal protocol

Status: **authorized and prepared; not yet submitted pending the mandatory QS
create confirmation**.  The executable freeze is
`experiments/qs_ordered_formal/protocol.json`.  This study is deliberately
narrow: it tests chronological online execution, stopped-answer correctness,
and realized request/token/latency cost on one new model--task carrier.

## Question and claim boundary

On the frozen BoolQ/Qwen2.5-7B-Instruct carrier, does a FIT-fitted and
CAL-screened `BAYES-H-online` policy have no more than 2 percentage points lower
gold accuracy than a paired `FULL-N` policy while reducing completed output
tokens?

The observable bit is a parsed **Yes vote**, not correctness:
`Z_i=1{rollout i parses to Yes}`, `x_k=sum_{i<=k} Z_i`, and ties return No.
Gold is absent from prompt rendering, acquisition, fitting, calibration,
stopping, and decision records.  It is joined only after both TEST decisions
for every paired task have been append-only and hash-sealed.  The old
OpenMathReasoning pass/fail prior, table, and numerical results do not transfer
to this carrier.

Positive results support only this public pooled BoolQ carrier, model revision,
prompt, scheduler, and operating point.  They do not establish long-CoT cost,
provider billing, useful cancellation, general ordered-exchangeability, or
population-level safety.  BoolQ is public and possible model pretraining
contamination is a limitation.

## Frozen carrier and allocation

- Dataset: `google/boolq` revision
  `35b264d03638db9f4ce671b711558bf7ff0f80d5`; canonical JSONL SHA-256
  `13c2f4143ae320a0191c6de5be919248a20c15515f58c6deb7d3732068f2d31a`.
- Model: `Qwen/Qwen2.5-7B-Instruct` revision
  `a09a35458c702b33eeacc393d103063234e8bc28`; exact 14-file snapshot ledger
  SHA-256
  `3ee6c9510b7e50bfcd46d6df33cafa3e2019f13a6a09bf1d2f9e80cdfe1164e8`.
- Allocation seed: `20260822-A11-BOOLQ-v2-passage`.  Strip passage and question
  text; define task and passage IDs by the hashes in `protocol.json`; retain one
  outcome-blind hash-selected question per passage; hash-rank passage
  representatives into FIT/CAL/TEST of 3,000 tasks each.  The three splits have
  zero exact-passage overlap.  The selected-manifest SHA-256 is
  `c1cb98d45600db7c234396c73161921905c4fa414a0cdd57f02b8d304d5505d8`.
- A separate outcome-blind hash fixes 300 TEST tasks for the shadow diagnostic;
  shadow-manifest SHA-256 is
  `89fd7fb6b42dad981b564116e755aecbc3ab2c41dcd14e5e0a65b7bd1053d013`.

## Frozen generation and policy

- `N=32`, minimum stopping index 3, temperature 0.8, top-p 0.95,
  `max_tokens=4`, no retries, and one live request per task.  FULL and BAYES use
  the same stateless request primitive and independently namespaced,
  deterministic signed-int32-compatible request seeds.
- The prompt requires exactly `Yes` or `No`.  The frozen parser uses the first
  canonical whole word; a noncanonical result deterministically maps to No and
  is reported rather than retried or dropped.
- FIT collects all 32 votes and fits the empirical histogram of full-count
  `K`.  For a prefix `(x,k)`, the policy thresholds the resulting
  hypergeometric posterior flip score.  A zero posterior denominator never
  authorizes a stop: the episode continues to `FULL-N`.
- There is one predeclared candidate (`J=1`), with
  `alpha_stop=0.05`.  CAL evaluates its exact DP replay loss `g_r(K)` on all
  complete CAL counts and applies the frozen empirical-Bernstein upper bound at
  `delta_cal=0.05`.  If the UCB exceeds 0.05, execution terminates before TEST,
  records a calibration-screen scientific rejection, and performs no retuning.

## TEST execution

Each TEST task receives independently seeded `FULL-N` and `BAYES-H-online`
episodes.  An outcome-blind balanced ordering makes FULL first for exactly
1,500 tasks and BAYES first for exactly 1,500 tasks.  A request completes at
final response/EOS; the online policy checks for stopping only then.  Because
dispatch is strictly sequential within a task, a BAYES stop has zero in-flight
requests.  This is recorded as no-prefetch/not-applicable, not as evidence of a
successful cancellation mechanism.

All 6,000 primary TEST decisions and their hashes are sealed before any shadow
request is dispatched.  For the frozen 10% subset, the shadow phase then
continues the same BAYES trajectory with its preassigned suffix seeds to
`N`; it does not restart an episode.  Shadow rows are excluded from all primary
cost endpoints.  This global deferral prevents shadow traffic from contaminating
primary latency.

## Endpoints and success rule

The paired TEST primary endpoints are:

1. one-sided 95% paired task-bootstrap lower bound for
   `accuracy(BAYES-H-online)-accuracy(FULL-N)`, which must be at least `-0.02`;
2. 95% paired bootstrap interval for completed output-token reduction, whose
   favorable lower endpoint must exceed zero.

Both are required for a positive result.  Also report both accuracies,
Yes/No-gold strata, ties, strict parser noncompliance, input/output tokens,
request count, time-to-final-answer, TTFT, and all failures.  TTFT is not
presumed to improve.  Provider-billable tokens remain unknown unless exposed.
The shadow prefix/full-vote diagnostic is secondary and cannot rescue either
primary endpoint.

## Integrity and failure semantics

The runner writes append-only task, rollout, stop, cancellation-status, and
episode ledgers.  UTC wall timestamps are descriptive; all causal ordering uses
a monotonic clock.  The final audit reconstructs votes, counts, returned answer,
certificate, the first eligible stop, FULL fallback, seed namespaces, hashes,
and shadow continuation to `N`.  Missing/failed episodes are not silently
dropped.

Import, checksum, image/model, server-health, output-path, or trace-integrity
failures are infrastructure/preflight failures and are not paper evidence.  A
CAL rejection is a registered scientific calibration result.  A completed,
integrity-valid TEST that fails accuracy noninferiority or token reduction is a
scientific negative and remains in the registry.  No GPU smoke precedes this
formal run.
