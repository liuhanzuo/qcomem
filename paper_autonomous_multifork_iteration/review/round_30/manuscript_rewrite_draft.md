# Round 30 manuscript rewrite draft

This is a staging note, not manuscript source.  It deliberately excludes every
R30 lane that has not reached terminal evidence closure.

## Proposed title

`ForkAudit: Trusted-Capture Validation of Ownership Traces on One Fixed Hybrid LLM Stack`

## Proposed abstract spine

1. Hybrid shared-prefix inference creates a state-ownership problem that output
   equality does not resolve.
2. ForkAudit is a trusted-capture trace validator, not an independent live
   monitor.  Under an honest, mandatory-event-complete producer, detached
   replay validates frozen identity, immutable document state, request/base and
   request/peer ranges, tail COW, the registered GDN transition, and call
   receipts at declared observation points.
3. Primary RR2: one fixed Qwen3.5/H20 stack, sequential batch-one schedule,
   `8 x 3 x 4 = 96` configurations and 288 adjacent-fan-out comparisons; exact
   canonical token/logit/KV/GDN relations; selected numerical controls; nine
   designed first-gate faults; `4.901 - 2.229 = 2.672 GiB` allocated-delta
   reduction at `N=32` in the stated ownership cell.
4. R30 bounded captured-boundary sweep: 20/20 attention rows across all 10
   attention layers and 24/24 GDN rows across 12/30 GDN layers pass; all 44
   seeded wrong-operator controls reject.  Keep the two-input, post-RoPE / post-
   q/k-normalization boundary adjacent.
5. Operational positioning: full live capture is an offline/debug/CI mode in
   the measured implementation (`4.321x` on one frozen request-step); it is not
   a serving-overhead or capacity result.
6. Close with the conditional boundary, without a long list of unrelated
   runtime/generalization exclusions.

## Conditional trace-validity statement

Let `Sigma` be the frozen case specification, `E` an execution, and
`tau = C_Sigma(E)` the producer trace.  For target `i`, let `M_i(Sigma)` be the
mandatory record slots and `Phi_i` the registered replay predicates.  Mandatory
trace coverage is complete only when every slot in `M_i(Sigma)` has exactly one
schema-valid, byte-bound record.  The replay verdict is

```
Pass_i(tau) iff Coverage_i(tau) = complete
                 and Bind_Sigma(tau)
                 and every phi in Phi_i evaluates true on tau.
```

Conditional conclusion: if capture is honest and mandatory-event-complete and
the manifest, digest, and replay implementations are correct, `Pass_i(tau)`
implies that every registered predicate held at its captured observation point.
A missing or modified mandatory record, or a represented predicate violation,
cannot produce `pass`.  This is trace-relative only; it says nothing about
coherent omission/fabrication, transient restored writes, unrecorded compiled
dispatch, common-mode semantics, or unseen-fault error rates.

## Coverage and verdict terminology

- Coverage: `complete`, `partial`, or `open`.
- Replay verdict: `pass`, `fail`, or `not evaluated`.
- Remove `receipt-complete (RC)` everywhere.
- In the seven-target table, use separate `Trace coverage` and `Replay verdict`
  columns.  Dispatch is `partial / pass at Python-call scope`; targets 1--4 and
  6--7 are `complete / pass` for the declared fixed case.
- Canonical exactness means equality after the declared CPU-FP32 or typed-digest
  serialization; it is not bitwise comparison of an unarchived device tensor.

## Exact 39-token explanation

Use this wording in both the protocol paragraph and the geometry table:

> Here “state-appended” counts tokens consumed into KV/GDN state: the first
> model call consumes the 32-token query, the next seven consume generated
> tokens 1--7, and generated token 8 is output but is not fed into another
> model call, giving `32 + 7 = 39`.

Geometry-table generation row:

> `8 output tokens; 39 state-appended input tokens (32 query + 7 feedback)`

## Results order

1. Primary RR2 trace validation and seven-target coverage/verdict status.
2. Relational equality and ownership witnesses.
3. Expanded captured-boundary attention/GDN oracle sweep.
4. Designed-fault comparison and lifecycle controls.
5. Memory endpoints.
6. Full-capture and local-replay cost, explicitly positioned as offline/debug/CI.
7. User-requested Mac and H20 main tables, clearly labeled unpooled deployment
   context.
8. Move the detailed five-pair live-cost table to the appendix; retain only its
   bounded summary in the main results.

## Novelty boundary

The reusable contribution is the integration of (a) a typed, phase-indexed
trace schema, (b) mandatory coverage semantics that fail open rather than
silently treating absent records as not applicable, and (c) ownership-specific
relations spanning token-indexed KV and mutable recurrent state.  The
Qwen3.5/H20 implementation is evidence that this protocol can be instantiated
on one stack.  Paging, prefix reuse, COW, metamorphic relations, hashing, and
cache policy are prior mechanisms, not claimed novelties.

## R30 evidence currently eligible

- `evidence/r30_expanded_oracle_sweep/validation_report.json`
- Status: `verified_bounded_fully_preregistered`
- Candidate attempts: one
- Attention: 20/20 clean pass, all 10 attention layers, two inputs, 160 query
  positions, maximum relative L2 `0.0018973927068607452`.
- GDN: 24/24 clean pass, 12/30 layers, two inputs, 192 transitions, maximum
  output/state relative L2 `0.002072614929712376` / `2.0848835241919874e-7`.
- Positive controls: 20/20 attention and 24/24 GDN rejected.
- Hard boundary: captured post-RoPE and post-native-q/k-normalization inputs;
  not end-to-end or independent capture.

## R30 evidence not eligible yet

- Native batching B: terminal FAIL due a frozen tokenizer-vs-model-vocabulary
  receipt dimension mismatch; retained internally.
- Native batching C: correctly frozen full-model-vocabulary protocol, not
  executed because QS control-plane access is blocked by REDpass office ACL.
- Post-discovery alias repair clean C: valid development regression and replay,
  explicitly not a prospective/held-out manuscript claim.
- Fresh-input end-to-end control: input manifest frozen; candidate not run.
- Independently authored detector-blind fault set: author freeze in progress;
  candidate not run.
