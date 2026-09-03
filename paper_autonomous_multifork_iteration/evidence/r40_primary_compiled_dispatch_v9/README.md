# R40 primary compiled-dispatch v9 (`20260827i`)

This is a new, non-overwriting metadata-closure package derived from frozen
v8. It modifies neither v8 nor earlier evidence, the paper, the immutable RR2
runner, or its original launcher. No QS/SSH/GPU operation was used to build or
test this package, and no formal result is claimed here.

V7's frozen archive contained one `__pycache__` directory and 13 `.pyc` files
inside the detached primary code. Its formal script necessarily rejects any
such member before launch, so v7 could not complete. V8 excluded every
`__pycache__`, `.pyc`, and `.pyo` archive member, requires the detached code
directory and rejects contamination before creating the one-shot result root,
then repeats that check after Python preflight. V9 preserves that repair and
corrects v8's stale `V7 targets` README label to `V9 targets`; all non-bytecode
RR2 scientific source and input bytes remain unchanged.

## Evidence boundary

V9 targets an honest-process, trusted pinned PyTorch/CUDA boundary. For every
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
- A fresh extraction must contain zero `__pycache__`, `.pyc`, or `.pyo`
  members anywhere. The primary-code check runs once before `RESULT_ROOT`
  creation and again immediately before the immutable launcher.

## Local gates

The source tree passes 24 CPU tests with zero skips, Python AST compilation,
Bash syntax, the dependency ledger, and the focused source-fixture ledger. A
fresh normalized archive extraction repeats those checks and also passes the
archive-side focused-fixture ledger. The archive contains only safe normalized
regular-file/directory members and zero Python-bytecode artifacts. A static
formal-preflight trace proves the authorization gate precedes detached-code
presence and bytecode gates, which in turn precede one-shot result-root
creation, while the repeated post-preflight gate still precedes the immutable
launcher. See `local-detached-validation.json`.

## Formal execution remains HOLD

The launcher remains unusable without the exact explicit gate and a fresh
independent PASS audit:

```bash
R40_H20_EXECUTION_AUTHORIZED=yes bash \
  /mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r40_primary_compiled_dispatch_v9_20260827i/paper_autonomous_multifork_iteration/evidence/r40_primary_compiled_dispatch_v9/executed_source/r39_primary_formal_h20.sh
```

Its fixed one-shot result root is
`.../runs/qcomem/r40-primary-compiled-dispatch-v9-20260827i`. A failed or
successful root is never reused. Package construction does not itself grant
launch authority; the operator must deliberately set the exact gate only
after the fresh archive receives an independent PASS audit.
