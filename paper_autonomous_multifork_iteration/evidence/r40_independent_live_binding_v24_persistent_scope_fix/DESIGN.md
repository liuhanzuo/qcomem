# V24 persistent-build-scope repair design

V24 repairs producer endpoints; it does not weaken `ActualBindingVerifier`.
The exact global-freshness, descriptor, offset, interval, storage-size, peer
isolation, and persistent-source predicates remain byte-identical to v22.

## Controlled runtime seam

R39 installs its runtime hooks before loading the immutable runner and checks
the exact bound updater functions inside every GDN forward. Replacing the
native cache functions or rebinding a layer method would invalidate the frozen
dispatch receipt before the R40 callback. The compatible overlay seam is the
rank's backbone call boundary:

```text
single token: fresh compact conv pre-rebind
              -> unchanged in-place causal-conv route
all tokens:   unchanged native recurrent update
              -> fresh compact recurrent post-rebind
              -> immutable runner callback / allocator sample
```

The hook pair is installed once from the rank-wide runtime loader and removed
by the combined restore path. The resident builder is wrapped for every call
only to validate 30 linear layers and mark every request as covered. This keeps
primary updater identities and route counts unchanged while removing the
selected-witness-only bypass.

## Exact persistent-prefill exception

V23 showed that the immutable `_convert_persistent` path performs one backbone
call with a new, necessarily unmarked document cache before the resident-group
builder can mark requests. V24 wraps that exact immutable conversion function
with a one-shot identity state machine. The state binds the exact backbone and
document, admits exactly one pre/post pair on the same cache, and accepts
completion only if `_convert_persistent` returns that exact cache. The call is
not included in resident-request counters and receives no compact-rebind
mutation. Every unmarked cache outside this state, including a multi-token
cache, fails closed. The wrapper identity is restored with the other rank-wide
hooks.

## Setup descriptor canonicalization

V22 reported that the prebuild request-base view used an authorized
noncanonical stride `[33546240,1,8192]`, while the first real convolution
endpoint used canonical compact stride `[32768,4,1]`; all other descriptor
fields matched. Canonicalizing persistent state inside the outer builder would
invalidate the immutable runner's prebuild persistent guard. Cloning an already
returned request would instead add a derived lineage edge and make the final
endpoint different from the direct persistent-rooted destination.

V24 scopes a replacement of the immutable builder's global
`_prepare_request_gdn_base` to the materialize policy. The request tensor is
still an exact persistent alias when the helper creates its final direct
contiguous clone, and the unchanged outer `PassiveCloneLineageMode` records
exactly one persistent-rooted edge to that live destination. Borrow delegates
the original helper. Runtime receipts require exactly 60 direct compact
materializations per materialized request, no compact materialization for
borrow, and exact helper restoration.

## Measurement boundary

The final live cache still contains one compact convolution and one compact
recurrent tensor per coordinate. The repair adds transient clones before the
immutable allocator endpoint, and the R40 historical verifier intentionally
retains superseded endpoints. Old allocator values are therefore invalid and
must be rerun. No result is accepted until semantic exactness, dispatch
receipts, cleanup, and terminal closure all pass in the same formal execution.
