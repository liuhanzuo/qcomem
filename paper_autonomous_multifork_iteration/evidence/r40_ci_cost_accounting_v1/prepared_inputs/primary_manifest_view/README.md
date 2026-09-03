# Anonymous RR2 W-run evidence and offline replay

This Round-4 package binds the exact RR2 W-run raw evidence, receipts,
preregistration, executed source closure, and reviewer-safe configuration.  It
does not modify or replace `upstream/forkaudit-summary.json`; that registered
aggregate remains byte-identical with SHA-256
`8700901ad7423d215e9e9e81a709e976f43963752e1b9f3d64441412b390d2bc`.

Run the complete offline replay from this directory:

```sh
./replay/run_replay.sh
```

The command verifies the 536-entry raw SHA-256 ledger, reconstructs all 96
factorial configurations and 96 three-phase ownership timelines, recomputes
the eight FP32 oracle values from binary sidecars, reconstructs the nine
matched-clean/mutant pairs, regenerates the validation-dashboard inputs plus
the allocator and mutant tables, and
runs adversarial unit tests for the pointer-free overlap algorithm.  No network
or GPU is required.  Before any replay, the command also verifies the complete
package file set, byte sizes, SHA-256 values, and the externally checkable
`MANIFEST.sha256` sidecar.  The v2 manifest records the preceding immutable v1
manifest as its parent; the functional revision fixes the Python 3.9 test
entry point and binds this verifier without rewriting upstream evidence.

## Schema correction without rewriting history

The original v1 aggregate used the field name `oracle_max_relative_l2` for the
preregistered tolerance `0.005`.  The replay emits
`derived/derived_summary_v2.json`, where:

- `oracle_relative_l2_tolerance` is `0.005`; and
- `oracle_max_relative_l2` is the observed maximum
  `0.0017432502481433169`.

The derived summary includes the parent aggregate's raw SHA-256 and explicit
invariants showing that the two values are distinct and that the observed
maximum is below the tolerance.

## Pointer-free storage witness

Each tensor view is represented by a normalized first-appearance storage ID,
shape, stride, storage offset, dtype, device, storage size, and content digest.
Its conservative byte interval is computed by accumulating the minimum and
maximum displacement of every strided dimension and converting the resulting
element bounds to a half-open byte interval `[start, end)`.  Two views overlap
iff their normalized storage IDs match and their half-open intervals intersect.
Exact aliasing additionally requires identical geometry, storage metadata, and
content digest.  Absolute addresses are rejected by the strict row schema.
Whenever the ownership contract requires disjointness, replay compares the
complete Cartesian product of tensor rows across owners rather than only rows
at matching coordinates.  The derived storage validation records separate
request/base and request/peer all-pairs comparison counts; adversarial tests
inject aliases at deliberately nonmatching coordinates for both relations.

This bounding-interval method may conservatively flag overlap for exotic
interleaved views, but cannot overlook a byte shared by two views.  Timeline
capture IDs must be unique, while request and persistent lifetime-guard IDs
must remain stable across setup, transition, and generation; therefore address
reuse cannot by itself establish storage continuity.

## Provenance and anonymity

All raw evidence came from the immutable W-run directory referenced by the
Round-3 submission and was checked against its original detached receipts.
The exact executed source was recovered from the source path embedded in that
run and verified against `upstream/preregistration/code.sha256`.  The original
command is represented only by its byte SHA-256 because it contains private
mount paths.  Compiled Python caches and a test log containing absolute paths
are deliberately outside this anonymous reviewer release.
