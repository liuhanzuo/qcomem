# RW-D5 I component dtype — one-GPU debug-only preregistration

This bundle authorizes only a single-GPU, sequential Prefix Cache then HYPIC
component inventory on Trial 1879097. It is not formal evidence and cannot
produce a paper number. The full 16-cell launcher is included only for review
and must not be executed.

Freeze F is an invalid 0/16 runtime attempt; its exact receipt is included.
Debug A and H were both retired before GPU submission. A lacked exact terminal
component validation. H fixed that but incorrectly treated JSON object insertion
order as semantic: producer canonicalization sorts nested keys, so a correct
HYPIC receipt would fail after disk round-trip. Their identities and blockers
are recorded in the two retirement receipts; neither is authorized.

Official HYPIC commit `98147c01909004e66d98bcb18b886927d41b0ee5`
constructs convolution and convolution-tail buffers with
`cache_params.dtype.conv`, and temporal and transition buffers with
`cache_params.dtype.temporal`. Defaults are BF16 and FP32, respectively. The
debug run must observe and validate live tensors, not promote source facts into
runtime results.

The debug hook activates only when `FORKAUDIT_RWD5_DTYPE_DEBUG_PATH` is set.
It records each live component's dtype, element size, shape, stride, device,
contiguity, allocator capacity, cache classes, exact dtype environment, and
official commit, then returns before formal authority or Store receipt output.

The fail-closed validator requires:

- exact cache class (`MambaRadixCache` or `PICache`) and `MambaPool`;
- exact environment BF16 convolution and FP32 SSM;
- exact semantic component key set independent of JSON object ordering:
  Prefix `{conv[0], temporal}` and HYPIC additionally
  `{transition, conv_tails[0]}`;
- BF16/2-byte conv/tails and FP32/4-byte temporal/transition;
- exact recurrent-layer/allocator-slot axes, tensor ranks, positive shapes and
  strides, independently recomputed C-contiguity, and `cuda:0` device;
- HYPIC transition/tail topology consistent with temporal/conv tensors.

The client validates immediately after collection. The launcher then re-reads
the producer's canonical JSON from disk via a separate `dtype_debug_validate`
stage. Only two passed validation receipts plus an empty
`formal-receipts-disabled` directory permit `COMPLETED_DEBUG_ONLY`.

Local pre-freeze checks:

- focused tests: 59/59;
- inherited same-protocol tests: 10/10; combined: 69/69;
- real producer `_atomic_json(sort_keys=True)` disk round-trip passes for both
  valid Prefix and valid HYPIC inventories;
- wrong/missing/extra/layout/identity/environment debug negatives fail closed;
- mixed component formal producer/replay and byte totals pass (Prefix 160 bytes,
  HYPIC 448 bytes in the toy fixture);
- Python compilation and both launcher Bash syntax checks pass;
- GPU executions from this bundle at freeze time: zero.

Only `launch_hypic_component_dtype_debug_1gpu.sh` is authorized after exact
manifest verification. Do not run `launch_hypic_retained_state_bytes_8gpu.sh`.
