# Attempt C root-cause audit (internal, postexecution)

## Disposition

Attempt C exposed an implementation defect in the one-token GDN transition,
not merely an over-strict audit predicate.  A completed request can retain the
same mutable convolution-state tensor, storage, and byte range as both the
persistent base and an unadvanced peer.  The clean rejection is therefore a
valid ownership alarm, while the whole attempt remains operationally invalid
and ineligible for manuscript use.

This is a postexecution diagnosis.  It does not change the frozen C
preregistration, does not rehabilitate any C fault result, and is not itself a
new held-out experiment.

## Bound observations from C

- All three clean lanes reached the frozen one-token semantic horizon and
  cleaned up exactly, but each stopped at the third mandatory receipt.
- The authenticated rejection was identical on all three ranks:
  `gdn_completed_binding_rebound`, with the message
  `completed request[0] did not out-of-place rebind layer:0/conv/state:0`.
- The first two receipts passed.  Aggregation then failed closed on
  `rank 0/clean clean completion`; no aggregate summary or completion marker
  exists.
- Three rank JSON files and nine 993,280-byte FP32 sidecars were present.  The
  remote `raw-artifacts.sha256` ledger verified all twelve entries before the
  pod disappeared.  Its bytes were not mirrored locally, so no per-file hash
  is reconstructed or guessed here.

## Why request, base, and peer share one range

The request builder seeds `copy.deepcopy` with every tensor object in the
persistent cache (`gpu/qcomem_vllm_paged_multifork_resident.py`,
`_seed_tensor_memo`, lines 276--291).  Consequently, each request initially
reuses the exact same GDN tensor objects.  The borrowed-base constructor then
requires `_storage_key(request_tensor) == _storage_key(source_tensor)` for all
60 GDN tensors and records 60 exact base aliases (lines 488--540).  With two
requests constructed independently from the same persistent cache, this gives,
coordinate by coordinate:

```
request[0] tensor is persistent tensor
request[1] tensor is persistent tensor
therefore request[0], persistent, and request[1] are the same tensor object
```

The storage key is `(device, untyped_storage.data_ptr(),
untyped_storage.nbytes())` (lines 76--78).  Object identity also preserves
shape, stride, storage offset, and hence the byte interval computed by the
witness from `(shape, stride, storage_offset, element_size)`.

For a numeric descriptor reference, the hash-bound RR2 witness from the same
model/checkpoint and source closure records layer-0 convolution state at borrowed
setup as:

| field | persistent | request 0 |
|---|---:|---:|
| normalized storage ID | `storage-0000` | `storage-0000` |
| shape | `[1, 8192, 4]` | `[1, 8192, 4]` |
| stride | `[33546240, 1, 8192]` | `[33546240, 1, 8192]` |
| storage offset | `0` | `0` |
| dtype | `torch.bfloat16` | `torch.bfloat16` |
| storage bytes | `65536` | `65536` |
| covered byte range | `[0, 65536)` | `[0, 65536)` |

Reference file:
`evidence/round_04_rr2_package/upstream/raw/rank-0/N-1/arm-4d56c6884bfcef13/witness/phase-setup_pre_transition.json`,
SHA-256
`0d1561048e6fd74f76a65d24ca29bbea46016383679a78b336f572e8b5478156`.
This prior descriptor is used only to make the same-stack geometry concrete;
it is not mislabeled as a C-local post-transition snapshot.  C failed before
that post-transition snapshot could be serialized.

## The one-token path bypasses the functional cache seam

The exact Transformers 5.14.1 source installed in the frozen runtime was read
on the node before it disappeared.  In `Qwen3_5GatedDeltaNet.forward`, a cached
`seq_len == 1` call reads
`cache_params.layers[layer_idx].conv_states[0]` and sends it directly to
`causal_conv1d_update`; the source comment states that this per-step kernel
updates convolution state in place.  This branch does not call
`cache_params.update_conv_state`.  The recurrent state is different: its final
state is still written through `cache_params.update_recurrent_state`.

The local adapter in `gpu/qcomem_qwen35_native_cache.py` only replaces the
cache-layer `update_conv_state` and `update_recurrent_state` methods.  It thus
works for the multi-token convolution branch, which calls
`update_conv_state`, and for recurrent state, but it cannot intercept the
direct single-token convolution mutation.

The relevant source bindings are:

- resident request builder:
  `546efd59e2833034bc2e24d4cc0e6077f5a408275e359af43cd96f7f71cad16e`
- functional cache adapter:
  `2ede63c74e4799316cc179cd3900f1e26e8dc284da326233376b2ed4c79d3a84`
- storage witness:
  `57c0dfe457abf165f058faac57200173f8e75f874cd3220510e4ac676a9fc520`
- C executor:
  `b5efd926cef0dc6505a8d710af453c38499784491f807ef63a5b3d9bfd63d360`

These files are in the imported RR2 code ledger bound by execution-input v3.

## Disjointness conclusion

The witness defines overlap as equal normalized storage ID plus intersecting
byte intervals.  At post-transition, the shared-base contract requires every
completed request tensor to be disjoint from the persistent base and every
peer, including an unadvanced peer.  At the failing layer-0 convolution
coordinate, the one-token path neither replaces the cache mapping nor changes
the tensor object; it mutates that shared object in place.  Therefore:

```
request[0] range == persistent range == request[1] range
request[0] vs persistent: overlapping, not disjoint
request[0] vs request[1]: overlapping, not disjoint
```

The exact C-local normalized storage IDs were not emitted because the binding
guard rejects before the storage snapshot is returned.  Nevertheless, the
combination of (1) exact tensor-object reuse at setup, (2) the observed lack of
rebind at the failing coordinate, and (3) the installed in-place single-token
kernel path establishes a real cross-owner alias.  Relaxing the predicate
would hide the bug; it would not repair ownership.

## Honest implementation repair

Keep the scientific predicate unchanged.  Before a cached one-token model call
on a request borrowing the persistent GDN base, clone and rebind each of the 30
convolution-state tensors for that request only.  The operation must be
fault-ID-blind and conditional on the request still aliasing the borrowed base;
already-private convolution states must be a no-op.  The clone must preserve
shape, stride-visible semantics, dtype, device, and exact content while using
storage disjoint from the persistent base and all peers.  The existing
functional recurrent-state update then supplies the other 30 post-transition
rebinds.

This repair changes ownership timing only: convolution state is borrowed until
immediately before its first in-place single-token transition, then becomes
request-private.  Eagerly materializing all 60 states at request construction
would change the intended borrowed-base contract and is not the proposed fix.

`gpu/r30_clean_single_token_gdn_diagnostic.py` (SHA-256
`c0bfd465c78643d711b2b645f9573f9ba0b33a95a000059159fc7f2a8c731435`)
was prepared after discovery to capture pointer-free before/after descriptors.
Its capture primitives were later source-bound and used in the replacement
trial's explicitly postexecution-development clean regression; the standalone
diagnostic entry point was not run.  Neither use rehabilitates attempt C or
turns the repair evidence into a held-out result.
