# R39 dual-producer repeat design

## Question

Can the frozen R33 six-capture result be reproduced by two fresh producer
processes, each with fresh out-of-process receivers, when both are constrained
by one independently derived slot census frozen before either producer starts?

This is a repeatability check against accidental one-off producer/capture
error. It is not an adversarial attestation protocol.

## Frozen execution

The bundle launches the unchanged R33 H20 producer twice, serially, on one
selected GPU. Both invocations use the same frozen Qwen3.5-35B-A3B revision,
input receipts, R33 preregistration, source ledgers, two policy cells, and
three capture phases. A fresh top-level Python process is created for each
invocation. Each invocation creates one fresh spawn receiver per policy cell,
so a valid run has two producer PIDs, four observer PIDs, and four independent
observer-session commitments.

Before either invocation, the unchanged R39 census generator derives 180
opaque slots from model geometry and schedule alone. Producer manifests and
producer rows are unavailable to that derivation. Each of the two results must
first pass the unchanged R39 census audit and the unchanged R33 lifecycle
replay independently.

## Cross-producer equality

For each of 2 policies x 3 captures x 180 slots, the verifier requires:

- the same opaque slot ID and exact five-field semantic coordinate, also equal
  to the preexecution census;
- exact equality of the stable receiver-derived tensor descriptor;
- exact equality of the SHA-256 digest over the tensor's contiguous byte
  payload; and
- exact equality of all 16,110 receiver-derived relation labels in that
  capture.

The totals are 1,080 semantic coordinates, 1,080 stable descriptors, 1,080
content digests, and 96,660 unordered relation labels per producer. Numeric
tolerance is zero and there is no canonical-semantic fallback.

`storage_token` and `view_token` are receiver-local HMAC identities and must be
different/unrelated across sessions; the verifier does not compare those token
strings. It independently recomputes and compares the relation labels derived
from them.

## Why byte-exact content equality is preregistered

Before designing this prospective repeat, two already completed independent
formal executions were compared read-only:

- R33 raw SHA-256
  `50d39cfcea072fb770da539d90abeddcd8a40802b88f4f95315001333c09e974`;
- R39 fresh R33 raw SHA-256
  `40ad93bfc340efe71d614240529267f0855d97fb1ebfe8657e23007af2b6c51a`.

They closed exactly on 1,080/1,080 semantic coordinates, 1,080/1,080 stable
descriptors, 1,080/1,080 bytewise content digests, all six relation-vector
digests, and 96,660/96,660 relation labels. Therefore the prospective protocol
freezes byte-exact equality. A mismatch is a failed result; it cannot trigger a
post-hoc tolerance or fallback.

## Claim boundary

If the formal run passes, it supports only this statement: under one frozen
Qwen3.5/H20/PyTorch stack and one frozen input/protocol, two fresh serial
producer executions with fresh out-of-process receivers exactly reproduced all
1,080 semantic slot observations, bytewise content digests, stable descriptors,
and 96,660 independently reconstructed relation labels per execution.

This reduces the risk of accidental producer-side enumeration, semantic
binding, or capture error. It does not eliminate trust in the producer, resist
a malicious producer, attest OS/driver/runtime state, prove that a correct slot
ID names the intended tensor against adversarial substitution, or establish
cross-stack generality.

