# R40 held-out fault method freeze v7

Status: `HOLD_PENDING_FRESH_AUDIT_AND_EXTERNAL_OPERATOR_BINDING`.

This is a method-only, fail-closed freeze. It contains no fault set, formal
configuration, operator binding, snapshot-release record, or GPU result. Nothing in
this directory authorizes a scientific claim or a formal run.

V7 fixes all five release blockers found by the independent v6 audit. The signed
operator payload and the signed execution contract both bind the canonical runner
manifest bytes and an independently recomputed closed-tree inventory. Every argv
index and every allowed environment entry has one exact type. A separately signed,
exact runtime expectation binds the interpreter, fixed probe, Python identity,
torch package root, device case, and complete staged runtime tree; the isolated
probe derives its arguments only from that verified expectation. Lifecycle success
is derived from eight real `Popen` handles and their wait/kill receipts, never from
a caller-provided gate. The full receipt and its digest are committed into every
terminal. Terminal signals require exact integers, so `2.0` and `15.0` are rejected.

The independent operator must publish a canonical binding signed by the private key
corresponding to `OPERATOR_TRUST_ROOT.json`. The package contains only the public
verification key and accepts no verifier callback, key override, or self-asserted
approval tuple. The snapshot remains sealed; do not give it to a designer until a
fresh independent audit and the external binding both exist.

Declared boundaries are unchanged: malicious concurrent same-user mutation and
power-loss/crash atomicity across a multi-file publication are outside this local
method freeze. Normal exceptions are rollback-safe and all formal artifacts are
no-replace. The host kernel and dynamic loader remain part of the fixed trusted
stack; v7 does not claim hostile-runtime attestation.
