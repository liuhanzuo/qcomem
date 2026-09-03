# ForkAudit aligned-prefix lifecycle transfer

## Frozen question

Does the existing ForkAudit ownership contract continue to hold when the
document prefix is exactly page aligned and request slots are cancelled,
scrubbed, reclaimed, and reused under an epoch-bound lease?

This is a transfer across geometry and lifecycle on the same Qwen3.5-35B-A3B,
Transformers, and vLLM Q16 adapter. It is not a second independently developed
model or serving runtime and must not be described as one.

## Formal design

- Eight H20 ranks and eight distinct PG-19 train books.
- Document length 4096, page size 128, query length 32.
- Four resident slots, four semantic rounds, suffix slots 2 and 3 cancelled
  after two rounds.
- One uninterrupted clean control and one lifecycle arm per rank.
- Cancelled private pages are zero-scrubbed before the exact reservation slots
  are reassigned with epoch 1.
- Surviving and replacement requests must reproduce the control's generated
  tokens, every full-vocabulary logit tensor, final logical KV, and final GDN
  state exactly.
- Immutable document K/V is hashed before and after each cell.
- Aligned geometry must record zero partial-tail staging bytes.
- The preregistered fault schedules one epoch-0 cancelled handle after its
  slot has been reassigned. It must fail at `STALE_SLOT_LEASE`; a matched
  epoch-1 replacement must pass the same guard and execute normally.

## Claim boundary

Positive results authorize only aligned-page and cancellation/reclamation
lifecycle transfer for the existing adapter. They do not authorize claims
about another model, recurrent implementation, serving engine, true concurrent
kernel execution, latency, throughput, NVML capacity, or production scheduler
integration.
