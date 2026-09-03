# R30 post-discovery D clean C: local evidence mirror

This directory mirrors the small audit surface of the immutable Tidal run at
`/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/r30-postdiscovery-d-clean-20260825c`.
The two 993,280-byte FP32 sidecars and the 637,893-byte raw result remain in
Tidal and are bound here by the raw and final SHA-256 ledgers.  The remote
`final-artifacts.sha256` was verified entry by entry with `sha256sum -c` after
the candidate-import-free replay completed.

Attempts A and B are retained remotely as operational-invalid integration
evidence.  A failed before model load because the imported R29 adapter expected
a UUID-valued CUDA selector.  B executed one model step but failed before any
token/logit value or clean verdict was emitted because the runner referenced a
digest helper from the wrong RR2 module.  C is therefore explicitly
postexecution development evidence, with both amendments preserved.

C supports only that the real single-token alias repair works at the tested
request-0 boundary on one frozen model/runtime/GPU: 30 convolution clones on
the first transition, zero on repeat, 60 completed-request tensors disjoint
from base and peer, byte-exact full FP32 logits and terminal state versus the
materialized control, and exact cleanup.  It is not preregistered science, a
held-out fault result, a multi-step/order regression, a detection rate, a
cross-runtime result, or a manuscript-ready claim.
