# V25 mixed-policy and abort-aware receipt repair

V25 preserves the v24 producer endpoint repair and exact persistent-build
scope. It does not weaken `ActualBindingVerifier`; global freshness,
descriptor, offset, interval, storage-size, peer-isolation, persistent-source,
lineage, allocator, semantic, and terminal predicates remain unchanged.

## Mixed-policy request construction

The formal runner uses both GDN policies. `requests_wrapped` is therefore the
sum of borrowed and materialized returned requests, not a borrowed-only count.
V25 adds `borrowed_requests_returned` at the resident-group wrapper and freezes
three exact equalities:

```text
borrowed_setup_calls_delegated == borrowed_requests_returned
materialized_setup_calls_canonicalized == materialized_requests_returned
borrowed_requests_returned + materialized_requests_returned == requests_wrapped
```

The independent materialized-state equality remains
`materialized_setup_states_canonicalized == 60 * materialized_requests_returned`.
Both policies must be nonempty. The regression reproduces the full warmup,
formal 3-by-4 memory/witness matrix, and rank-dependent fault group counts.

## Exception-aware cached-call accounting

Expected mutant controls can raise inside the backbone after the pre-hook has
observed a valid wrapped cached call. V25 registers the existing post-hook with
`always_call=True`. When PyTorch supplies a null output for that exception
path, the hook increments `cached_calls_aborted_before_postprocess` and returns
without cloning recurrent state. On normal return it performs the unchanged
30-state recurrent post-rebind. The receipt requires:

```text
cached_calls_postprocessed + cached_calls_aborted_before_postprocess
    == multi_token_cached_calls_observed + single_token_cached_calls_observed
recurrent_states_post_rebound == 30 * cached_calls_postprocessed
```

Thus an abort is neither mistaken for a successful postprocess nor left
unaccounted. A regression verifies that the original backbone exception is
preserved.

## Unchanged runtime seam and measurement boundary

The pre-hook still gives every successful cached single-token call a fresh
compact convolution target before the unchanged in-place causal-convolution
route. The post-hook still gives every successful cached call fresh compact
recurrent endpoints before the immutable callback/allocator endpoint. The
exact `_convert_persistent` identity scope still bypasses only its one document
prefill cache. All old allocator values remain invalid until a new terminally
closed formal execution passes.
