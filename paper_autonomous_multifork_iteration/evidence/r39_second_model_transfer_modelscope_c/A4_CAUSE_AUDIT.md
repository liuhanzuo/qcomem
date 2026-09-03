# A4 valid-negative cause audit

Audited immutable inputs:

- A4 aggregate SHA-256:
  `33f9acb87baaf15fd62e74e39cd5c57260f626554be35542c122317dffdfc4da`.
- A4 static preregistration SHA-256:
  `ccd1ef0bcd9dc98c0d8fc871326162bfc0bcb6cf2e6beb7599af04433b023cb0`.
- A4 split-adapter source SHA-256:
  `5901f153fcfcabbfab63f756a3c19a04ace56b4985fc02421f2dde4118a7373c`.

## What the records establish

For every one of eight ranks, `cross_arm_exact` is true at both `N=1` and
`N=2`.  Frozen identity, prefix immutability, private ownership, cross-arm
equivalence, and cross-N consistency all replay as full.  Therefore A4 did not
fail because the persistent fork escaped, aliased mutable request state, or
disagreed with independently materialized requests.

The negative comes from the independent one-shot full-model oracle.  In the
registered `N=1`, request-0 trajectories, all eight ranks exceed the frozen
0.005 relative-L2 tolerance.  Across the 16 registered steps the range is
0.01580442957741941--0.08068267932341613.  Fifteen of 16 step top-1 values
match; rank 7 step 1 does not.  Those clean oracle failures create the 16
reported clean false positives and make positive runtime transfer
unauthorized.

## What the records do not establish

A4 compares a cached, two-chunk manual split path against a standard
one-shot recomputation, but it does not include either of the following:

1. official one-shot wrapper versus a manual one-shot traversal of the same
   layers; or
2. official one-shot wrapper versus the official full-model DynamicCache
   document/query chunking path.

Consequently the frozen evidence cannot attribute the discrepancy uniquely to
manual mask construction, the split boundary, or the numerically distinct
recurrent cached/chunk kernel path.  Calling any one of those the proven A4
root cause would overstate the evidence.

## R39 repair boundary

The old A4 source is not modified.  R39 subclasses its tested generic state,
packing, cloning, and cache utilities in a new file.  The subclass mirrors the
official Transformers 5.14.1 hybrid-mask mapping for both
`qwen3_5_moe_text` and dense `qwen3_5_text`, passes an explicit active full
attention layer index when split suffix caches leave earlier layers empty, and
records mask routing.  The two missing reference controls above are frozen as
preconditions for reference authority.

