# Authoritative method-v3 contract

## Fixed authority

`authoritative_config.json` is the sole method authority.  Its file SHA-256,
the preregistration SHA-256, designer-snapshot manifest SHA-256, atomic-policy
SHA-256, schedule SHA-256, and method-core manifest SHA-256 are compiled into
`executed_source/v3_constants.py`.  Public verification accepts no configuration
argument and loads fixed package-relative paths.

The frozen geometry is Qwen3.5-35B-A3B revision
`59d61f3ce65a6d9863b86d2e96597125219dc754`, two requests, and exactly sixteen
round-major calls: request A then request B for each of eight rounds.  Round
zero consumes 32 tokens per request; rounds one through seven consume one token
per request.  Every call exposes exactly 248,320 little-endian FP32 logits.

## Trusted capture boundary

The wrapper owns synchronization and reads bound live tensor objects.  It
computes canonical component digests from sorted tensor role, dtype, shape,
stride, device, byte length, and contiguous bytes.  Logical KV length, KV/GDN
versions, and KV/GDN commit epochs are read from bound scalar tensors after the
same synchronization.  A model-call return value or callback cannot provide a
state snapshot.  Formal mode requires the torch/CUDA backend; a CPU backend is
test-only and rejected by the formal executor.

This remains trusted producer instrumentation, not adversarial attestation.
The exact tensor bindings, observer code, CUDA synchronization, filesystem,
hash implementation, and reference lane are in the trusted computing base.

## Exact disk inventory

For every frozen case and each `reference`, `clean`, and `mutant` lane, the
verifier requires exactly:

- `lane-binding.json`;
- `allocator.json`;
- `receipts/call-000.json` through `receipts/call-015.json`; and
- `logits/call-000.f32le` through `logits/call-015.f32le`.

Files must be regular non-symlinks beneath the fixed verification root.  Any
missing, extra, renamed, duplicate, truncated, nonfinite, or hash-mismatched
artifact is invalid.  The verifier reads bytes from disk; it does not consume
preparsed receipts or caller mappings.

## Receipt and identity binding

Each sealed receipt binds campaign ID, run ID, lane, case ID, assigned GPU UUID,
call key, exact schedule and method hashes, model revision, atomic-policy hash,
input token count, token, full-logit descriptor, and independently observed live
pre/post snapshots.  Observation IDs and synchronization-event IDs are globally
unique across the complete campaign, including allocator endpoints.

## Gates

Semantic comparison requires exact call order/cardinality, exact tokens, and
byte-exact full-vocabulary FP32 logits.

Atomic coherence requires, for each call consuming `q` tokens: KV length
advances by `q`; KV and GDN versions each advance once; KV and GDN commit epochs
each advance once and agree before and after; and each request's next pre-state
equals its preceding post-state.

Paired structural comparison additionally requires every candidate pre/post KV
and GDN content digest, length, version, and epoch to equal the corresponding
reference state.  A reference change paired with an unchanged, stale, or
rolled-back candidate therefore fails even when outputs match.

Allocator comparison uses exactly synchronized H0/H1/H4/H6/H7 endpoints bound
to campaign, run, lane, case, and device.  Peak bytes never decrease after the
H0 reset; current and peak bytes match reference at every phase; and H7 current
bytes equal the lane's own H0 value.

## One-shot boundary

A later sealed formal configuration must contain the only output root and eight
specified H20 UUIDs.  The executor accepts no output/configuration arguments.
Before work, it verifies an otherwise empty exact eight-H20 node, source and
configuration hashes, creates a campaign-global O_EXCL lock and a config-hash
O_EXCL lock under the fixed campaign parent, then precreates eight pending
terminal records.  Changing the output root or formal configuration cannot
bypass the campaign-global lock.  One idempotent finalizer handles SIGINT,
SIGTERM, normal exit, and caught exceptions, kills registered process groups,
and retains a terminal or pending record for every case.  Source/configuration
hashes are checked before and after execution.

Local tests and historical examples are development evidence only.  The
scientific campaign remains HOLD until a fresh independent audit approves this
freeze and a fresh isolated designer later seals a new case set.

