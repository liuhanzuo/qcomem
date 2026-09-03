# R40 v4 real-binder verifier

Status: **local mechanism PASS; formal H20 integration HOLD; GPU not run**.

Unlike v1--v3, this package does not mutate an off-path candidate dictionary.
`ActualBindingVerifier` freezes source content, descriptors, and storage identity
before `original_build`; validates the actual returned request group; and, only
after `original_phase` returns, compares the real
`gdn_phase_witness.storage_witness.rows` with the current live objects and the
frozen lifecycle expectations.

Local tests place faults in the real builder/group mapping: coherent
cross-layer swap, request/base alias, same-geometry one-way mapping error, and
stale post-rebind mapping. All fail closed. A clean real group and returned
phase rows pass; serializer-row tampering fails.

Run locally:

```bash
python3 -m unittest discover -s tests -v
python3 scripts/static_audit.py
```

There is deliberately no formal launcher. Remaining integration gates are
listed in `acceptance.json`; no Qwen/H20 result is claimed.
