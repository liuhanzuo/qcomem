# Repository agent instructions

## Paper experiment execution policy

When reviewer feedback requires new experiments and the user authorizes execution:

- Prefer the smallest falsifiable formal run over building a maximal audit framework first.
- Block submission only for a known defect that could invalidate, contaminate, mislabel, or make the result irreproducible.
- Reuse the last known working scheduler or CLI submission route before inventing new deployment infrastructure.
- Submit after focused correctness, input, configuration, duplicate, and output-path checks pass; let the frozen formal launcher perform its full preflight.
- Do not add a GPU smoke when the user requests direct formal execution unless a concrete unresolved defect makes the formal run likely to be uninterpretable or waste substantial resources.
- Continue non-authorizing hardening, adversarial audits, packaging, and manuscript refinement in parallel with queued or running jobs.
- Run independent experiments in parallel when they have no shared-state or dependency constraint and resources permit.
- Record infrastructure/preflight failures separately from scientific negative results. Never cite a run that stopped before scientific execution as experimental evidence.
- If the user says to submit now and the minimal gates above pass, submit immediately; do not reopen settled design questions solely for stronger assurance.

For the detailed 2026-08-17 retrospective, read `paper_autonomous_multifork_iteration/state/decision_log.md` under “Experiment-submission delay retrospective.”
