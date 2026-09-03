# R40 V31 borrowed-transition live-binding successor

This is a non-overwriting successor to the terminally successful V29 materialized-cell run. It changes the selected ownership-witness cell only to `N=8`, `vllm-q16-shared-document-reuse`, and `borrow-immutable-base-functional-rebind`.

The verifier freezes all 60 persistent GDN coordinates before the actual production builder. It requires 480 exact request-to-persistent aliases at setup, then observes every actual cached forward. After request 0's first transition it requires 60 fresh/private endpoints while the other 420 request rows remain exact aliases. After generation it requires all 480 request rows to be private and peer/base-disjoint.

Each of 64 functional rebind calls records 60 exact edges with pre/post tensor object IDs, storage keys, full descriptors, and content SHA-256 values. The unchanged production phase serializer is checked against all 540 rows at setup, post-transition, and post-generation, and each serialized artifact is reread from its published rank path.

Local tests are mechanism checks only. The package remains `HOLD_PENDING_FRESH_AUDIT_AND_H20` until one terminally closed eight-rank H20 run and post-run audit pass.
