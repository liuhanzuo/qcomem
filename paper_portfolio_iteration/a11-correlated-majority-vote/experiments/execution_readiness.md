# Execution readiness — `EXP-ORDERED-CORRECTNESS-COST`

## 2026-08-22 execution amendment

The historical read-only audit below has been superseded operationally.
`experiments/qs_ordered_formal/` now contains a frozen BoolQ protocol,
stdlib runner and focused tests, an eight-H20 offline launcher, and a resolved QS
preview.  The user has authorized the experiment.  The run remains **prepared
but not submitted** until (i) the exact model and code bytes are staged and
read-back verified, (ii) the independent launcher audit has no result-invalidating
blocker, and (iii) the mandatory QS pre-create confirmation is received.

Current focused gates already pass locally: passage-disjoint allocation replay
(3,000 FIT / 3,000 CAL / 3,000 TEST), selected and shadow manifest hashes,
rendered prompt-length budget, Python compilation, launcher syntax, and the
runner unit suite.  The frozen policy uses parsed Yes votes rather than a
correctness bit; gold is joined only after paired TEST decisions are sealed.
The single predeclared CAL rule is evaluated with exact DP loss and must pass
its fixed empirical-Bernstein gate before TEST.  Shadow continuation is deferred
until all 6,000 primary decisions are sealed.

QS target: workspace `829312787779653`, queue 436
(`Reasoning_Rollout`), cloud 6, cluster 53, resource package 183
(`8Gpu/170C/1800Gi`), one eight-H20 worker, no overuse or restart, image ID
593 / literal tag `vllm017.latest`.  Capacity, workspace, output-path absence,
and all staged hashes must be rechecked immediately before creation.  Queue
436 supersedes the original queue-400 preview because queue 400 fell from 40
to zero remaining H20 at the final execute gate; queue 436 was the compatible
same-cloud/cluster/package rollout queue with the largest remaining capacity.
No GPU
smoke is planned.  A pre-TEST infrastructure failure is not scientific evidence;
a valid fixed-CAL rejection is registered as a scientific calibration result.

Audit date: 2026-08-22. Scope: read-only inspection of `remote_snapshot/` and
its supplied result artifacts. No GPU, API, remote, or billable job was
submitted.

## Historical read-only verdict

**No existing launcher can directly execute the required ordered + correctness
+ cost protocol.** The package contains a mechanical claim checker and
CPU-oriented replay/re-analysis scripts, but no end-to-end client that captures
chronological responses, evaluates gold correctness of returned online answers,
or records token/latency/cancellation telemetry.

| Requirement | Evidence in supplied package | Ready? | Reason |
|---|---|---:|---|
| Exact count-replay certificate / FIT-CAL-TEST computation | `claim_check.py`; supplied `fit_cal_test_r469_result.json`; replay scripts | Partial | Existing scripts rely on absent external parquet/original workspace paths for a fresh rerun. |
| Ordered rollout acquisition | No scheduler/API client/launcher found | No | Counts and sample IDs are not an append-only chronological online trace. |
| Gold correctness of online returned answer | No online evaluator integration found | No | Existing artifacts expose full-vote flip or offline reward/count labels, not a paired online returned-answer primary endpoint. |
| `FULL-N` vs `BAYES-H-online` paired TEST episodes | No episode orchestrator found | No | No immutable task/episode/request key management or randomized episode runner. |
| Token and latency accounting | No telemetry schema/client found | No | `1-\bar k/N` is a replay rollout-count proxy, not generated/completed/billable tokens or latency. |
| Cancellation accounting | No cancellation client or ledger found | No | No request acknowledgements, post-stop completions, or unknown-field handling. |
| Policy causality audit | No trace-audit checker found | No | No verifier that each decision depends only on rows arriving before its decision timestamp. |
| Result registration | `evidence/experiment_registry.json` and formal protocol exist | Partial | Registration schema/protocol exist, but no execution output path or launcher implementation is supplied. |

## Reported RLVE preflight candidate (not a confirmatory result)

Parent-agent SSH read-only preflight on Singapore host `47.84.140.142`
verified that the original RLVE raw carrier is available at
`/newcpfs/user/qixuan1/01_p5_share/run/iclr27_theory_k3_20260806_r1/agents/A11/workspace/earlystop_drift_r474`,
with six parquet files (about 1.1 GB) and schema fields including `index`,
`sample_id`, `prompt`, `response`, `rewards`, `metadata`, and `answer`.
The remote Python 3.12 environment has `pyarrow` and can read this schema;
these fields can support a schema-level answer-parsing and ordered-trace
harness preflight.

This local revision environment could not access that remote-host path and
lacks `pyarrow`, `rlve_eval`, and `math_verify`; therefore it did not inspect
rows, read TEST outcomes, install dependencies, or run an evaluator. The
remote preflight reports `math-verify` 0.9.0 but no `rlve_eval`, leaving an
RLVE evaluator/parser integration blocker. This is an infrastructure-access
observation, not a scientific negative result.

Even if the reported carrier is available, it is an already-used paper carrier
and must not be silently promoted to an independent confirmation. A formal
reuse would need an explicitly frozen reuse amendment (including split and
selection restrictions); the preferred confirmatory design remains a new,
frozen model--task carrier with a deterministic evaluator. In either case,
the missing online launcher, paired episode execution, and cancellation/token
telemetry remain blockers.

## What was found

- `remote_snapshot/claim_check.py` is a stdlib-only mechanical consistency
  checker. It reads supplied local JSON and frozen `remote_snapshot/paper.tex`,
  reports three external-provenance items, and is not an experiment launcher.
  `manuscript/claim_audit.py` is a separate hash-bound audit for the current
  revised prose/package; it also is not an experiment launcher or a numerical
  rerun.
- `openr1_m2_pilot_r473.py`, `rlve_n8_r474.py`, and the scripts in
  `remote_snapshot/results/` are offline/replay or re-analysis programs. Their
  headers explicitly state no GPU/network use; several refer to parquet paths
  outside this package (for example `../earlystop_drift_r467/...`).
- `edit_a11_earlystop_fig1.py` can call an image-generation API for a
  conceptual figure. It is unrelated to ordered rollout execution and must not
  be repurposed as an experiment route.
- No `*.sh`, scheduler manifest, service endpoint client, request-cancellation
  implementation, or online JSONL/Parquet ledger writer was found.

## Minimal pre-authorized implementation gap

Before a future execution authorization, an author must supply or implement a
versioned launcher that follows `decisive_experiment_protocol.md` exactly:

1. freeze an eligible model--task carrier, deterministic gold evaluator, and
   FIT/CAL/TEST task manifest;
2. write append-only `rollout_trace`, `stop_decision`,
   `cancellation_ledger`, and `run_manifest` records;
3. execute sequential, causally audited `FULL-N` and `BAYES-H-online` TEST
   episodes plus the preselected shadow-completion subset;
4. preserve unknown token/billing/cancellation fields as `unknown`, never zero;
5. run only the focused preflight in the protocol, then register both valid
   results and pre-scientific failures separately.

This audit does not authorize implementation or submission. The experiment
remains `planned_not_authorized_to_run` in the registry.
