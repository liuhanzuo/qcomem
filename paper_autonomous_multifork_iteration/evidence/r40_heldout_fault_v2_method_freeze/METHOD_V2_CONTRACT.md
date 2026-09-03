# ForkAudit method-v2 frozen contract

## Scope

Method v2 is an offline, non-adversarial regression contract for a fixed hybrid
KV/GDN execution.  It is not security attestation.  Correct semantic-to-live
state binding, framework storage semantics, synchronization, and the live-state
reader remain trusted.  A pass is relative to the registered observations.

This freeze is method-only.  It contains no v2 fault definition or outcome.
All R28/R29/R30/R33/R35/R39/R40 fault definitions known before this freeze are
development data and are permanently excluded from v2 held-out scoring.

## Gate S: paired complete semantic envelope

For reference and candidate arms, every scheduled call has one ordered key
`(call_index, round_index, request_id)`, one exact surfaced token, and one
complete-vocabulary contiguous little-endian FP32 sidecar.  The verifier reads
and hashes the sidecar bytes itself.  Cardinality, order, shape, dtype, byte
length, finiteness, and token identity are mandatory.

The frozen policy is either:

- `exact`: all FP32 sidecar bytes are identical; or
- `declared_tolerance`: both maximum absolute difference and relative L2 are at
  or below thresholds frozen in `preregistration.json` before fault design.

Tokens remain exact in both modes.  Missing, duplicate, malformed, nonfinite,
or extra rows are invalid rather than passes.

This is a conventional paired semantic baseline.  It must be reported
separately from structural ForkAudit gates.

## Gate A: paired synchronized allocator envelope

Each arm records exactly H0/H1/H4/H6/H7 after device synchronization and after
peak reset at H0.  Current allocated and peak allocated bytes must match the
paired reference exactly at every phase.  Each arm must restore H7 current
allocated bytes exactly to its own H0 value.  Duplicate/missing phases,
unsynchronized rows, negative values, or a peak below current are invalid.

The denominator is the registered framework allocator only.  It is not NVML,
process memory, admission capacity, or evidence that auditing saves memory.

## Gate C: hybrid atomic-version coherence

Every scheduled state-changing call is wrapped by the auditor:

1. independently read synchronized live KV/GDN state before the call;
2. execute exactly one model call;
3. independently read synchronized live KV/GDN state after the call;
4. persist the surfaced token and complete FP32 logits; and
5. derive and seal a receipt from the call key, input, live pre/post snapshots,
   and logit artifact.

The call result cannot supply the state snapshot used by the gate.  Each live
snapshot binds request ID, logical KV length, KV and GDN content digests, KV and
GDN versions, KV and GDN commit epochs, observation ID, source kind, and
synchronization status.  For a call appending `q>0` tokens:

- post KV length equals pre KV length plus `q`;
- KV and GDN versions each advance exactly once;
- KV and GDN commit epochs each advance exactly once and are equal after the
  call;
- the next call for the same request begins from the exact preceding post
  snapshot; and
- the ordered receipt set exactly equals the frozen schedule.

The rule is uniform over every call.  Fault-ID branches, payload sentinels, and
post-outcome thresholds are forbidden.  Content equality is not required:
digests are bound to prove which live states were observed, while monotone
length/version/epoch relations decide atomic coherence.

## Held-out ordering

The complete detector/replay source, schemas, thresholds, clean gate, and this
contract must be hash-frozen before a fresh isolated fault designer is spawned.
The designer may receive only `designer_snapshot/`.  The detector cannot change
after fault bytes are frozen.  Every valid, escaped, ineligible, invalid, and
clean-false-positive row must be retained.  A later campaign may be called
“method-v2 held-out, designer--executor separated,” not a natural-defect recall
study.

