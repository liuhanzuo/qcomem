# R40 V33 single-link canonical-input successor

V33 is an operations-only, non-overwriting successor to V32.  V32 reached the
H20 container but stopped before scientific execution because the canonical V6
archive on the shared filesystem had more than one hard link.  The frozen
stager correctly rejected it.  V32 is therefore an infrastructure/preflight
failure and is not scientific evidence.

V33 preserves the complete V32 scientific payload, including the strict
rank-artifact counter value of eight, and changes no producer, model, data,
schedule, cell, threshold, or acceptance rule.  Before staging, the formal QS
command copies the already SHA-256-approved V6 archive byte-for-byte into the
new V33 scratch directory.  It then requires an exact regular file with one
link and the same approved SHA-256.  The canonical-file gate is not relaxed.
V33 also uses fresh stage, scratch, and result paths.

## Inherited V32 scientific change

This is a non-overwriting successor to V31. V31 completed all eight rank-local scientific shards, but its terminal finalizer rejected the rank artifacts because the frozen contract expected seven primary memory calls at ownership-artifact publication while the actual frozen schedule had observed eight. V31 remains inadmissible and is not reused as scientific evidence.

V32 preserves the V31 selected ownership-witness cell, producer, model, data, schedules, and all scientific settings. Its only non-identity runtime change is the post-science finalizer contract: the rank-artifact counter is the strict integer `8`; `7`, `9`, and booleans are rejected. The end-of-run global-absence counter remains the separately preregistered strict integer `12`.

The verifier freezes all 60 persistent GDN coordinates before the actual production builder. It requires 480 exact request-to-persistent aliases at setup, then observes every actual cached forward. After request 0's first transition it requires 60 fresh/private endpoints while the other 420 request rows remain exact aliases. After generation it requires all 480 request rows to be private and peer/base-disjoint.

Each of 64 functional rebind calls records 60 exact edges with pre/post tensor object IDs, storage keys, full descriptors, and content SHA-256 values. The unchanged production phase serializer is checked against all 540 rows at setup, post-transition, and post-generation, and each serialized artifact is reread from its published rank path.

Local tests are mechanism and packaging checks only. The package remains `HOLD_PENDING_FRESH_AUDIT_AND_H20` until one fresh terminally closed eight-rank H20 run and post-run audit pass.
