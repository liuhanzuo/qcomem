# R40 held-out method v4 freeze

Status: **HOLD_PENDING_FRESH_AUDIT**. This directory contains no fault set, formal
execution configuration, or GPU result. It does not authorize a scientific claim.

V4 is a fail-closed supervisory layer over the immutable v3 archive. The formal
campaign must externally pin both the audited v4 archive and source ledger; neither
hash is accepted from code inside that archive. `v4_guard.py` implements the pure
validators used before any one-shot execution. The future launcher must import these
validators from the externally verified archive and must not offer path overrides.

The trusted producer boundary remains explicit: CUDA tensors and the physical GPU
UUID are captured by the audited runner. This is software provenance, not hardware
attestation.

