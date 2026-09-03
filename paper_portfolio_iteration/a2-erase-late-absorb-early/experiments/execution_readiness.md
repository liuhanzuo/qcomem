# PLAN-ENABLE-001 execution-readiness audit

Audit date: 2026-08-22. Status: **not execution-ready; no job submitted and no GPU/remote task run.**

## Candidate implementation surface (remote-only; not public evidence)

| requirement | observed candidate surface | status |
|---|---|---|
| N data scenario | `remote_snapshot/code/pilot19_cifar10n.py`: CIFAR-10 images + `CIFAR-10_human.pt`, aggregate-vs-clean disagreement, clean test metric | partly specified, but data directory is missing locally |
| N arms / schedule | `never`, `always`, `drop@120`, injections; default T=240, split=120, eta 0.05/0.005, batch 256 | not the required A0/A1/A2 policy design |
| candidate A1 comparator | `eqra_loss_salvage.py` has `drop@120`, but A1’s fixed cohort/fraction and relation to A2 are not frozen | open |
| policy implementation | remote EQRA uses known `q_frac=0.10` and candidate clean/reference partitions | invalid for PLAN-ENABLE-001 decision-time restrictions |
| H scenario | `eqra_loss_p3_precision.py` uses a clean MNIST subset | not the protocol’s independently frozen benign-hard endpoint; no hard metric |
| seeds | full scripts list `[0,1,2,3,4,5]` | candidate only; register anew |
| output schema | candidate JSON emits seed lists, aggregate metrics, and some per-arm arrays | insufficient: no decision inputs, BIC, cohort stability, event log, shared-randomization record, or standardized failure state |
| environment | historical doc says Python 3, NumPy 1.24.4, torch 2.3.0a0, CUDA | current local package/data availability not established; safe Python environment probe did not complete in the allotted audit window |
| launcher | no working scheduler/CLI launcher found in the scoped remote candidate | blocked |

## Required frozen manifest before authorization

1. Name source code revision and a new public result directory; prohibit writing under `remote_snapshot/`.
2. Lock N/H dataset versions, licenses, split hashes, preprocessing, and H’s clean-label hard endpoint.
3. Define the mixture family, BIC implementation, checkpoints, fitting/tie-break seed, failure-closed behavior, operational mass range, and Jaccard calculation.
4. Define A0, A1, A2 (and separated diagnostic A3), including the fixed A1 cohort/fraction, re-admission rule, and exact paired RNG contract.
5. Lock six seeds `[0,1,2,3,4,5]`, command, environment lock, output directory, duplicate guard, and an atomic run manifest.
6. Require JSONL event records for decision inputs/outcomes, hashed cohort IDs, branch/RNG state, metrics, disabled/failure reasons, and file hashes; aggregate only from these records.
7. Run CPU-only unit checks for determinism, forbidden-field access, fail-closed branch, RNG/event logging, schema validation, and duplicate/output-path protection. Do not treat a quick GPU run as the formal result.

## Readiness verdict

PLAN-ENABLE-001 has a good falsification protocol but lacks a frozen implementation, N/H data manifests, A1 comparator definition, H endpoint, environment lock, current data access, and launcher. These are preflight blockers; they are infrastructure/reproducibility blockers, not scientific negative results. Once closed and the user authorizes execution, use the last known working route if one is recovered and submit the frozen formal job without adding an exploratory GPU smoke.
