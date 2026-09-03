# V26 construction-step receipt repair

V26 preserves the v25 producer endpoint repair, exact persistent-build scope,
and abort-aware call accounting. It does not weaken `ActualBindingVerifier`;
global freshness, descriptor, offset, interval, storage-size, peer-isolation,
persistent-source, lineage, allocator, semantic, and terminal predicates remain
unchanged.

## Two-step mixed-policy request construction

The immutable builder first calls `_prepare_request_gdn_base` with the borrowed
policy for every fresh or reused request. `_request_with_gdn_policy` calls the
same helper a second time only when the final policy is materialized. Thus
construction-step counters and final-policy counters have different meanings:

```text
borrowed_setup_calls_delegated == requests_wrapped
materialized_setup_calls_canonicalized == materialized_requests_returned
materialized_setup_states_canonicalized == 60 * materialized_requests_returned
borrowed_requests_returned + materialized_requests_returned == requests_wrapped
```

Both final policies must be nonempty. The focused regression executes one
borrow-final request and one borrow-then-materialize request. The full regression
reproduces max-N priming, four-arm warmup, the formal 3-by-4 memory/witness
matrix, and rank-dependent fault groups.

## Exception-aware cached-call accounting

Expected mutant controls can raise inside the backbone after the pre-hook has
observed a valid wrapped cached call. The inherited post-hook uses
`always_call=True`. A null exception-path output increments
`cached_calls_aborted_before_postprocess` without recurrent rebind. Normal
returns keep the exact 30-state rebind. The receipt requires:

```text
cached_calls_postprocessed + cached_calls_aborted_before_postprocess
    == multi_token_cached_calls_observed + single_token_cached_calls_observed
recurrent_states_post_rebound == 30 * cached_calls_postprocessed
```

## Failure diagnostics and unchanged measurement boundary

The rank gate evaluates named exact predicates. On failure it prints the sorted
failed names and the full canonical receipt counters in one deterministic JSON
diagnostic. It creates no new artifact and does not modify the successful path,
finalizer, terminal tree, or measurement boundary.

The pre-hook still gives every cached single-token call a fresh compact
convolution target before the unchanged in-place route. The post-hook gives
every successful cached call fresh compact recurrent endpoints before the
immutable callback/allocator endpoint. All old allocator values remain invalid
until a new terminally closed formal execution passes.
