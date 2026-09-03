# Round-39 compiled-dispatch receipts (`20260826g`)

This non-overwriting correction addresses one bounded gap: the audited adapter
recorded the Python vLLM attention callable but not each selected Triton
artifact/configuration. It also records the *actual* GDN route without claiming
that the frozen eager GDN fallback is compiled.

Version `g` preserves the scientific hook, count closure, and acceptance gates
from `f`; it only makes the archive's detached test replay self-contained. The
archive now carries the three immutable R29 fixtures consumed by the formal
tests and `r29-fixtures.sha256` binds their exact bytes.

## Why `20260826e` failed closed

R29 imports `audit_qwen35_functional_stack_plan`, but it does not execute the
manual `qcomem_qwen35_gdn_functional.dispatch_qwen35_decoder_layer` router.
Document prefill and both request arms execute Transformers
`Qwen3_5MoeGatedDeltaNet.forward`; qcomem changes only the native cache writes
to functional tensor rebinding. The MoE modeling file defines its own GDN class
and `torch_chunk_gated_delta_rule`, rather than aliasing the similarly named
base-Qwen implementation. The frozen Transformers 5.14.1 environment has no
FLA/causal-conv fast path, so the multi-token calls select that MoE-local
`torch_chunk_gated_delta_rule`. Hooking the unused manual router therefore
correctly emitted zero GDN calls and `20260826e` produced no eligible receipt.

The earlier expected count of 390 was also wrong. `_build_document_cache` is
inside `run_pair`, so all six pairs rebuild a 30-layer GDN document cache:

- document prefill: `6 pairs * 30 = 180` GDN calls;
- request cells: `6 pairs * 2 arms * 30 = 360` GDN calls;
- total: `540` GDN calls.

Versions `f` and `g` increase the exact gate to 540 and validate the two phases
separately; it does not reduce a count to make an observation pass.

## What is intercepted

`executed_source/r39_compiled_dispatch_receipts.py` installs hooks before the
functional stack, qcomem native-cache module, or Transformers Qwen3.5 module
has been imported by the entry point. It then intercepts:

- `vllm.v1.attention.ops.triton_unified_attention.unified_attention`;
- Triton 3.6's `CompiledKernel.run` property and `Autotuner.run`;
- the actual Transformers `Qwen3_5MoeGatedDeltaNet.forward`;
- the actually selected MoE-local `torch_chunk_gated_delta_rule`; and
- qcomem's functional conv/recurrent cache-update functions.

Every attention call must launch exactly one compiled kernel and bind exactly
one complete run-local Triton artifact directory, including PTX, cubin,
compiler hash, and `num_warps`/`num_ctas`/`num_stages`. Every GDN call must
observe exactly one eager chunk-rule invocation and exactly one functional conv
plus recurrent rebind. The Transformers wrapper, core GDN, selected chunk
function, and qcomem cache adapter sources are SHA-bound and copied into the
run output before detached replay.

`r39_verify_formal_binding.py` binds the receipt to the valid R29 result and
semantic sidecar. In addition to exact totals, it requires the exact sequence:
for each of six pairs, 30 document-prefill rows at sequence length 4033 with no
previous cache state, followed by two 30-row request-cell blocks at sequence
length 16 with previous state. Every block must visit all 30 linear layer
indices in config order.

The GDN receipt proves dispatch/source selection only. It does not attest the
underlying ATen, cuBLAS, cuDNN, or CUDA operators and does not describe the
eager fallback as a compiled artifact.

## Local checks

```bash
cd /Users/liuhanzuo/MacLLM-Bench
python3 -m py_compile \
  paper_autonomous_multifork_iteration/evidence/r39_compiled_dispatch/executed_source/*.py
python3 -m unittest discover \
  -s paper_autonomous_multifork_iteration/evidence/r39_compiled_dispatch/executed_source \
  -p 'test_r39_*.py' -v
bash -n \
  paper_autonomous_multifork_iteration/evidence/r39_compiled_dispatch/executed_source/r39_launch_h20.sh \
  paper_autonomous_multifork_iteration/evidence/r39_compiled_dispatch/executed_source/r39_formal_h20.sh
```

The seven unit tests include archived-R29 closure, separate 180/360 GDN phase
closure, missing attention/GDN rejection, run-local source snapshots, and
runtime/cache-source substitutions.

The formal-test class verifies `r29-fixtures.sha256` before executing any of
its three tests. A fresh detached extraction therefore fails immediately if the
R29 design, formal result, semantic sidecar, or fixture ledger is absent or
byte-substituted.

An import-only preflight also passed under the frozen H20 Python environment on
the already allocated Pod, without executing a model forward and without
initializing CUDA. A tiny CPU `Qwen3_5MoeGatedDeltaNet` instance selected the
intercepted MoE-local chunk callable; the qcomem installer bound both
intercepted cache-rebind methods; importing the functional stack afterward kept
that binding. The remote report is
`.../indep-bench_debug/r39-compiled-dispatch-20260826f-preflight3.json`
(SHA-256
`68737895b691a918abf4813e1da70e410cc73060eb2fdf4e6fa531e95c1abe5d`).
It records the selected MoE Transformers source hash
`688d9a8f2830d6729cd2945563f38b710100c086565b97c27c94c96bd9716b9f`
and qcomem native-cache source hash
`2ede63c74e4799316cc179cd3900f1e26e8dc284da326233376b2ed4c79d3a84`.

## Exact H20 launch (not run during package construction)

Stage the immutable package at
`/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r39_compiled_dispatch_20260826g`, then use the already allocated Pod:

```bash
printf '%s\n' \
  'bash /mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r39_compiled_dispatch_20260826g/paper_autonomous_multifork_iteration/evidence/r39_compiled_dispatch/executed_source/r39_formal_h20.sh' \
  'exit' | qs exec -i qs-252052-1907355-ai-1452425-master-0
```

The formal launcher uses only GPU 0, writes the fresh output root
`.../runs/qcomem/r39-compiled-dispatch-20260826g`, and creates, stops, evicts,
or deletes no QS resource. Do not launch it concurrently with an all-eight-GPU
formal experiment.

## Formal H20 result

The `20260826g` formal run terminated naturally and passed detached replay.
The downloaded full archive is
`formal_h20/r39-compiled-dispatch-20260826g-full.tar.gz` (SHA-256
`67c614b63e910194b8e4e18e2e1d94ce88d37d4aa68db6d1012852a573bae575`).
Its 35-entry terminal ledger rechecks locally. The formal aggregate, receipt,
and replay SHA-256 values are respectively
`348ce8d6714dfaee428479093e05cbeaee4bd89144c34d2d8b71834baf735be0`,
`bd7469a6de97a375c6f86f95a4a52072d36ad966d216134d8b1df46dc945d966`,
and
`56ad985c99baf354214bbf1aad838bef52314bf6fae3582be38a482352527318`.

The receipt closes exactly 120 expected unified-attention calls over one
selected, fully hashed Triton artifact/configuration. It separately closes 180
document-prefill and 360 request-cell GDN calls over the selected eager
`torch_chunk_gated_delta_rule` and qcomem functional cache-rebind sources. All
seven bound negative controls reject. This is evidence for the exact R29 H20
execution only: it does not establish a compiled GDN binary, underlying
ATen/CUDA operator identity, malicious-producer resistance, or cross-stack
generality.
