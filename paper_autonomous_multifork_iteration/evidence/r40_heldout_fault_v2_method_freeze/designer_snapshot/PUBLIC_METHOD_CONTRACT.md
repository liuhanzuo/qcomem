# Public method-v2 contract

The system executes a fixed schedule of state-changing calls over two resident
requests.  Each call consumes a positive number of input tokens and updates a
hybrid state with a KV component and a GDN component.  The evaluator applies
all three gates below to every eligible case; the designer cannot select or
change the gates.

## Paired semantic gate

A reference arm and candidate arm must expose the same ordered call keys and
the same surfaced token at every call.  Every call persists the complete
vocabulary-sized, contiguous, little-endian FP32 logit vector.  The registered
policy is byte-exact comparison, with zero numerical tolerance.  Missing,
extra, duplicate, reordered, malformed, truncated, nonfinite, or hash-mismatched
records cannot pass.

## Paired allocator gate

After synchronization and a peak reset, reference and candidate arms record
current and peak allocated bytes at H0, H1, H4, H6, and H7.  Both quantities
must match the reference exactly at every endpoint.  Each arm's H7 current
allocation must equal its own H0 current allocation.

## Hybrid atomic-version gate

Every scheduled call is bound to independently reread, synchronized live state
immediately before and after that call.  A live snapshot contains request ID,
logical KV length, KV and GDN content digests, component versions, component
commit epochs, an observation ID, and provenance.  For an input of `q > 0`
tokens:

- logical KV length advances by exactly `q`;
- KV and GDN versions each advance by exactly one;
- KV and GDN commit epochs each advance by exactly one;
- the two post-call commit epochs are equal; and
- the next pre-state for the same request exactly equals its preceding
  post-state.

The call result cannot provide the state read by this gate.  The complete call
schedule and sealed receipt cardinality must match exactly.

## Evaluation boundary

The observer, synchronization primitive, receipt writer, sidecar writer,
hashing implementation, and reference lane are trusted evaluation
infrastructure and are outside the mutation surface.  A case that cannot reach
the registered model call without violating those boundaries is retained as
ineligible or invalid; it is never silently replaced.

