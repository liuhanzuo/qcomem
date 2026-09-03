# RW-D5 G component dtype — one-GPU debug-only preregistration A

This bundle authorizes only a single-GPU, sequential Prefix Cache then HYPIC
component inventory on Trial 1879097. It is not a formal experiment and cannot
produce a paper number. The full 16-cell launcher is included for source/hash
review only and must not be executed from this bundle.

Freeze F is retired after its live run failed before 0/16 raw/store outputs.
The exact invalid-attempt receipt is included. Official HYPIC commit
`98147c01909004e66d98bcb18b886927d41b0ee5` constructs convolution and
convolution-tail buffers with `cache_params.dtype.conv`, and temporal and
transition buffers with `cache_params.dtype.temporal`. Its defaults are BF16
for convolution and FP32 for temporal state. The debug run must observe the
actual live tensors rather than treating those source facts as runtime results.

The debug hook activates only when `FORKAUDIT_RWD5_DTYPE_DEBUG_PATH` is set.
After the frozen rank-0 workload enters the cache, it records each component's
dtype, element size, shape, stride, device, and contiguity, then returns before
formal authority or Store receipt generation. The debug runner verifies that
no file appears under `formal-receipts-disabled`, flushes between modes, kills
the server before reusing the GPU, and labels every output non-paper evidence.

Expected source-derived mapping to be checked, not assumed:

- Prefix: `conv[0]` BF16 and `temporal` FP32; transition/tails absent;
- HYPIC: `conv[0]` BF16, `temporal` FP32, `transition` FP32, and
  `conv_tails[0]` BF16.

Local checks before this preregistration:

- producer and replay use storage-contract v3 and forbid legacy `dtype`;
- mixed BF16-conv/FP32-temporal positive mock receipts pass;
- toy Prefix and HYPIC unique payload totals replay at 160 and 448 bytes,
  respectively, using each component's own element size;
- producer rejects wrong temporal, transition, and convolution-tail dtypes;
- blind replay rejects same-element-size forged component dtypes;
- focused tests: 51/51; inherited same-protocol tests: 10/10; combined: 61/61;
- both shell launchers pass `bash -n` and Python sources compile;
- GPU executions from this bundle at freeze time: zero.

Only `launch_hypic_component_dtype_debug_1gpu.sh` is authorized after verifying
this exact manifest. Do not run `launch_hypic_retained_state_bytes_8gpu.sh`.
