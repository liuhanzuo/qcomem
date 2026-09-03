# R40 held-out fault method freeze v6

Status: `HOLD_PENDING_FRESH_AUDIT_AND_EXTERNAL_OPERATOR_BINDING`.

This is a method-only, fail-closed freeze. It contains no fault set, formal
configuration, operator binding, snapshot-release record, or GPU result. Nothing in
this directory authorizes a scientific claim or a formal run.

V6 fixes the release-blocking v5 interfaces: the operator key is fixed rather than
caller-selected, signed bindings are canonical, both snapshot commitments are
recomputed from the closed tree, signal handling is exercised in real subprocesses,
success has one exact lifecycle-gate schema, terminal trees are closed-world, torch
provenance is obtained in an isolated child, command paths are typed and
inventory-bound, and retained-parent publication rolls back linked outputs on a
late identity failure.

The independent operator must publish a canonical binding signed by the private key
corresponding to `OPERATOR_TRUST_ROOT.json`. The package contains only the public
verification key and accepts no verifier callback, key override, or self-asserted
approval tuple. The snapshot remains sealed; do not give it to a designer until a
fresh independent audit and the external binding both exist.
