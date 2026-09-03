# Public v3 evaluation contract

The system executes two resident requests for eight calls each.  Calls are
round-major: request A then request B in every round.  The first call for each
request consumes 32 tokens; each of the next seven calls consumes one token.

Every candidate is compared with a reference under three always-on gates:

1. Exact surfaced tokens and exact bytes for every complete 248,320-element
   little-endian FP32 logit vector at identical call cardinality and order.
2. Per-call hybrid state coherence from independently read live pre/post KV and
   GDN tensors.  Logical length advances by input-token count; component
   versions and commit epochs advance once; component epochs agree; request
   continuity holds; and candidate pre/post state equals the corresponding
   reference pre/post content and counters.
3. Synchronized allocator current/peak values at H0/H1/H4/H6/H7, with exact
   paired equality, nondecreasing peak after reset, and exact H7-to-H0 current
   restoration.

The reference lane, observer tensor bindings, synchronization, receipt and
sidecar writer, verifier, hashing, timeout, and terminal ledger are outside the
mutation surface.  A case that cannot execute without modifying those trusted
surfaces is retained as ineligible or invalid, not replaced.

