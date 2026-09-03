# Round-4 transfer design decision (pre-result)

Date: 2026-08-19 (Asia/Shanghai)

Issue: `A-EXPERIMENT-07`

## Alternatives considered

| Alternative | Reviewer impact | Scientific independence | Feasibility with frozen assets | Overclaiming risk | Disposition |
|---|---|---|---|---|---|
| Second independently implemented hybrid model/runtime | highest | highest | unavailable: no second hybrid model plus verified adapter/environment is present | high if a wrapper is mislabeled as independent | not executed |
| Materially different recurrent backend | high | high | unavailable: the repository exposes the same Qwen3.5 GDN functional seam | high if packaging changes are treated as a backend | not executed |
| Aligned prefix plus cancellation, reclamation, and slot reuse | medium-high | medium | executable on the frozen Qwen3.5 + vLLM Q16 path | low under a lifecycle-only claim | selected |

The selected experiment crosses two explicit boundaries of RR2: a 4096-token
page-aligned document and a request lifecycle with cancellation, zero-scrub,
exact private-reservation reassignment, and epoch-bound stale-handle rejection.
It includes an uninterrupted control, exact full-logit/KV/GDN comparisons, and
one setting-specific stale-lease fault with a matched clean replacement.

This is not a second model, recurrent implementation, runtime, or production
scheduler. A positive outcome can raise confidence in lifecycle and geometry
transfer, but cannot by itself establish implementation portability.
