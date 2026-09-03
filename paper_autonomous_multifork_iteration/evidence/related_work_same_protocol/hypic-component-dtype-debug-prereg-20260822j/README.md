# RW-D5 J component dtype — one-GPU debug-only preregistration

This bundle authorizes only a single-GPU, sequential Prefix Cache then HYPIC
component inventory on Trial 1879097. It is not formal evidence and cannot
produce a paper number. The included 16-cell formal launcher must not run.

Historical failures are preserved rather than interpreted as scientific
results. Formal freeze F was invalid at 0/16 because it assumed a unified Mamba
dtype. Debug A and H were retired before GPU use. Debug I passed independent
static audit but its live launcher failed under `set -u` before starting a
server: one `local` command expanded `${mode}` before that shell had assigned
`mode`. Exact retirement and invalid-attempt receipts are included.

J preserves I's independently audited component contract and makes only the
launcher declaration repair. Every local whose later value depends on an
earlier declaration is now assigned on a separate command. The analogous
`wait_ready` declaration in the frozen formal launcher is repaired as well,
although formal execution remains forbidden. A real extracted
`run_debug_mode` function smoke executes both approved modes under
`set -Eeuo pipefail` and crosses the formerly failing declaration without
starting a model or server.

The debug client remains fail closed on:

- official commit and exact `MambaRadixCache`/`PICache` plus `MambaPool` class;
- exact BF16 convolution and FP32 SSM runtime environment;
- semantic component key sets after producer canonical JSON round-trip;
- BF16/2-byte conv/tails and FP32/4-byte temporal/transition;
- recurrent-layer/allocator-slot axes, ranks, positive shapes/strides,
  independently recomputed C-contiguity, `cuda:0`, and HYPIC topology.

The launcher independently re-reads both live inventories with
`dtype_debug_validate`. Only two passed validation receipts and an empty
`formal-receipts-disabled` directory permit `COMPLETED_DEBUG_ONLY`.

Local pre-freeze checks:

- focused tests: 60/60; inherited same-protocol tests: 10/10 (70/70 combined);
- exact `run_debug_mode` function smoke passes for both modes under nounset;
- all audited canonical-JSON positive and wrong/missing/extra/layout/identity
  negative checks remain passing;
- Python compilation and both Bash launchers' syntax checks pass;
- GPU executions from J before freeze: zero.

Only `launch_hypic_component_dtype_debug_1gpu.sh` is authorized after exact
manifest verification. Do not run `launch_hypic_retained_state_bytes_8gpu.sh`.
