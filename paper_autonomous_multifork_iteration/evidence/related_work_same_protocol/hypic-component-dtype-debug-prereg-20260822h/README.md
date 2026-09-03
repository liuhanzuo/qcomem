# RW-D5 H component dtype — one-GPU debug-only preregistration

This bundle authorizes only a single-GPU, sequential Prefix Cache then HYPIC
component inventory on Trial 1879097. It is not a formal experiment and cannot
produce a paper number. The full 16-cell launcher is included for source/hash
review only and must not be executed from this bundle.

Freeze F is retired after its live run failed before 0/16 raw/store outputs.
The exact invalid-attempt receipt is included. The earlier debug preregistration
A is also retired because its terminal checked only status/schema and could
accept a forged component mapping. `retired-debug-a.json` records that defect
and its immutable identity. Neither retired bundle is authorized.

Official HYPIC commit `98147c01909004e66d98bcb18b886927d41b0ee5`
constructs convolution and convolution-tail buffers with
`cache_params.dtype.conv`, and temporal and transition buffers with
`cache_params.dtype.temporal`. Its defaults are BF16 for convolution and FP32
for temporal state. The debug run must observe and validate the actual live
tensors rather than treating those source facts as runtime results.

The debug hook activates only when `FORKAUDIT_RWD5_DTYPE_DEBUG_PATH` is set.
After the frozen rank-0 workload enters the cache, it records each component's
dtype, element size, shape, stride, device, contiguity, allocator capacity,
cache classes, exact dtype environment, and official commit, then returns
before formal authority or Store receipt generation.

The client and launcher both fail closed. For each mode the client requires:

- exact cache identity (`MambaRadixCache` for Prefix, `PICache` for HYPIC) and
  `MambaPool`;
- exact environment `SGLANG_MAMBA_CONV_DTYPE=bfloat16` and
  `SGLANG_MAMBA_SSM_DTYPE=float32`;
- Prefix keys exactly `conv[0]`, `temporal`;
- HYPIC keys exactly `conv[0]`, `temporal`, `transition`, `conv_tails[0]`;
- BF16/2-byte convolution and tails; FP32/4-byte temporal and transition;
- exact recurrent-layer and allocator-slot axes, positive shapes/strides,
  independent C-contiguity, expected tensor ranks, and `cuda:0` device;
- HYPIC transition and tail topology consistent with the live temporal and
  convolution tensors.

After each run the launcher independently re-reads the live inventory through
`dtype_debug_validate`. Only two passed validation receipts plus an empty
`formal-receipts-disabled` directory permit `COMPLETED_DEBUG_ONLY`. Wrong dtype,
missing component, extra component, forged stride/shape, cache-class, commit,
or dtype-environment drift all prevent the terminal marker.

Local checks before this preregistration:

- focused tests: 58/58;
- inherited same-protocol tests: 10/10; combined: 68/68;
- mixed BF16-conv/FP32-temporal positive mock receipts pass;
- toy Prefix and HYPIC unique payload totals replay at 160 and 448 bytes using
  each component's own element size;
- wrong/missing/extra/layout/identity/environment debug terminal negatives fail;
- producer and blind replay component-dtype tamper tests fail closed;
- Python compilation and both launcher Bash syntax checks pass;
- GPU executions from this bundle at freeze time: zero.

Only `launch_hypic_component_dtype_debug_1gpu.sh` is authorized after verifying
this exact manifest. Do not run `launch_hypic_retained_state_bytes_8gpu.sh`.
