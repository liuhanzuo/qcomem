# Pointer-free storage-witness specification

Version: `qcomem-pointer-free-storage-overlap-reviewer-spec-v2`

The witness intentionally excludes allocator addresses.  A producer assigns a
normalized `storage-NNNN` identifier when a backing storage is first observed
within a phase and emits tensor shape, stride, storage offset, dtype, device,
storage size, logical tensor size, and a content digest.

For an element size `e`, storage offset `o`, shape `s`, and stride `t`, the
replay computes:

```text
minimum = maximum = o
for (size, step) in zip(s, t):
    displacement = (size - 1) * step
    minimum += min(displacement, 0)
    maximum += max(displacement, 0)
byte_start = minimum * e
byte_end_exclusive = (maximum + 1) * e
```

The interval must be non-negative and contained within `storage_nbytes`.  The
declared `tensor_nbytes` must equal `product(shape) * e`.  The same normalized
storage ID cannot be reused with a different device or storage size.

Two views overlap exactly when their normalized storage IDs are equal and the
half-open intervals intersect:

```text
left.start < right.end and right.start < left.end
```

Exact aliasing additionally requires equal interval, shape, stride, storage
offset, dtype, device, storage size, tensor size, and content digest.  Bounding
intervals are conservative for exotic interleaved strides: they can produce a
false positive overlap but not a false negative shared byte.

For every state coordinate, `shared-base` setup and incomplete shared requests
require exact aliasing to the corresponding persistent state.  Every required
disjointness relation is stronger: the replay evaluates the complete Cartesian
product of tensor rows across the two owners, irrespective of layer, state
family, or state index.  Thus materialized or completed requests must be
all-pairs disjoint from every persistent row, and every peer pair not jointly
incomplete under `shared-base` must be all-pairs disjoint.  At the final phase
every request must be all-pairs disjoint from the base and every peer.
Binding-token receipts independently require unchanged tokens for incomplete
requests and changed binding and storage tokens for completed requests.

Phase capture IDs must be unique, while request and persistent lifecycle-guard
IDs must remain stable across all three phases.  Persistent binding and content
digests must also remain unchanged.  Thus reuse of the same allocator address
cannot establish continuity, and an absolute pointer field is rejected as a
schema violation.

`test_storage_witness.py` covers exact aliasing, partial overlap, adjacent
half-open offsets, negative strides, conflicting normalized-ID reuse, absolute
pointer rejection, capture/guard reuse relations, and nonmatching-coordinate
request/base and request/peer alias adversaries.  The full replay applies the
same all-pairs checks to all 96 timelines and 288 phase artifacts and reports
the mechanically evaluated comparison counts.
