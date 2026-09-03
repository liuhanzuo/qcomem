# R40 primary compiled-dispatch v7 (`20260827g`)

This is a new, non-overwriting repair package derived from the frozen v6
audit. It does not modify v6, the audit, the paper, the immutable RR2 runner,
or its original launcher. No QS/SSH/GPU operation was used to build or test
this package, and no formal result is claimed here.

## Evidence boundary

V7 targets an honest-process, trusted pinned PyTorch/CUDA boundary. For every
registered primary `unified_attention` call it:

1. creates an exact rank/cell/local call ID;
2. resolves the one expected `kernel_unified_attention` cache bundle and full
   compile configuration immediately before the original Triton launcher;
3. calls that original launcher;
4. emits no success evidence on exception; and
5. seals a receipt only after normal return on the same assigned CUDA device
   and stream.

The sealed digest covers the call ID and shape, selected artifact ID, complete
selected configuration, exact autotuner `selected_kwargs` plus
`num_warps`/`num_ctas`/`num_stages` (or the exact statement that no autotuner
was observed), device/stream identity, and both post-return predicates.

This is not driver/device binary attestation and is not malicious-runtime
resistance. It establishes an exact selected-launcher-to-successful-return
receipt only within the declared trusted runtime. GDN remains explicitly eager
and keeps the frozen mutually exclusive route accounting from v6.

## Additional closure

- The actual vLLM unified entry point and Triton kernel source, Triton
  `CompiledKernel.run` getter, Triton `Autotuner.run`, all Transformers GDN
  routes, and qcomem cache-rebind routes are source-hashed.
- A real frozen-environment CPU/no-CUDA preflight installs, instance-checks,
  and restores every hook before any shard. It records Python and exact
  torch/Transformers/vLLM/Triton versions and fails if CUDA initializes.
- Each rank is bound to a non-overwriting proxy launch-identity file whose PID
  survives `exec`, the immutable eight-row H20 assignment receipt, one UUID,
  the model/weight/code/protocol ledgers, immutable runner and launcher, and
  the runtime preflight manifest.
- Schemas are exact. Duplicate or unreferenced table rows, a decoy kernel,
  fallback calls, missing/extra calls, wrong ranks, missing ranks, altered
  source/artifact bytes, and incomplete aggregation fail closed.
- The outer supervisor records separate stdout/stderr, an allowlisted
  environment/version record, a non-overwriting failure receipt and partial
  terminal ledger on failure, or a terminal ledger followed by `COMPLETE` on
  success.

## Local gates

The source tree passes 24 CPU tests with zero skips, Python compilation, Bash
syntax checks, all dependency and frozen-fixture ledgers, and five permanent
regressions for the counterexamples accepted by v6. See
`local-detached-validation.json` after archive generation.

## Formal execution remains HOLD

The launcher is deliberately unusable without the exact explicit gate:

```bash
R40_H20_EXECUTION_AUTHORIZED=yes bash \
  /mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r40_primary_compiled_dispatch_v7_20260827g/paper_autonomous_multifork_iteration/evidence/r40_primary_compiled_dispatch_v7/executed_source/r39_primary_formal_h20.sh
```

Its fixed one-shot result root is
`.../runs/qcomem/r40-primary-compiled-dispatch-v7-20260827g`. A failed or
successful root is never reused. Do not run it until the fresh archive has an
independent PASS audit and the user separately authorizes the formal run.
