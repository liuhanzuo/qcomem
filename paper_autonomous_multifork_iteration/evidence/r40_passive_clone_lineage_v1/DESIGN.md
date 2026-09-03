# Passive materialized-clone lineage design

## Decision

`TorchDispatchMode` is viable for the frozen Qwen3.5 builder's ownership
witness.  The frozen builder seeds `copy.deepcopy` with every persistent tensor
object (`qcomem_vllm_paged_multifork_resident.py`, lines 276--291), so the
request-side base initially holds the exact persistent Python tensor handle.
For the materialized policy, `_prepare_request_gdn_base` directly executes
`cloned = request_tensor.clone()` and installs that returned object at the
current semantic coordinate (lines 488--533).  The outer builder performs this
once for each request (lines 575--609).

An observer can therefore be installed around the unchanged real builder:

1. Before the builder call, independently enumerate `(layer, family,
   state_index)` from the persistent object and hold each tensor strongly.
2. Enter a `TorchDispatchMode` and execute the unchanged builder.  At
   `aten.clone.default` (or a future explicit `aten.copy_.default`), record the
   actual source and destination tensor objects, their storage descriptors and
   byte intervals.  Retain strong handles so Python IDs cannot be recycled.
3. After the builder returns, independently enumerate the live request tensors.
   Each materialized destination must be the output of exactly one recorded
   edge rooted at the persistent tensor at the same semantic coordinate.  The
   current frozen path additionally requires a direct `aten.clone.default`
   edge.  Borrowed bases must remain exact aliases and emit no such edge.
4. Recheck that every persistent object, descriptor, storage interval and byte
   digest is unchanged.

This resolves the v6 same-content/same-geometry blocker without modifying any
scientific tensor value.  If coordinate A actually clones equally shaped and
equally valued source B, the operator edge says B and the post-build
coordinate-A lookup fails.  No builder-emitted audit row or mapping is used.

## Why the alternatives do not solve this blocker

- `saved_tensors_hooks` observes tensors retained for autograd.  The formal
  builder runs under inference mode, so it is not a provenance source for these
  eager ownership clones.
- CUDA allocator events identify allocation sites and destination addresses,
  but do not identify the semantic source of a clone when two sources have the
  same size.
- Content, shape, dtype, stride and source/request storage separation remain
  insufficient when two persistent sources share content and geometry.
- Monkey-patching `Tensor.clone` is higher-level and more invasive than
  observing the dispatcher; it is also easier for alternate call surfaces to
  bypass.

## Real runner integration surface

The ownership witness builds the persistent cache, captures a persistent guard,
calls `build_resident_request_group`, then captures its request guard in the
frozen runner (lines 3407--3460).  A future formal hook can place the registry
and dispatch mode exactly around that one builder call.  It must remain absent
from `_run_clean_memory_cell` (lines 3163--3235), so Python observer state and
strong references do not contaminate primary allocator endpoints.

## Boundaries and remaining gates

- The mechanism verifies construction provenance for the eagerly executed
  materialized/borrowed GDN base clones.  It does not attest that the persistent
  cache itself has correct semantics, defeat a malicious PyTorch dispatcher, or
  establish runtime independence.
- Source tensor object identities must be unique by semantic coordinate.  If
  the persistent cache itself aliases one object across coordinates, this
  observer fails closed because provenance is intrinsically ambiguous.
- The frozen helper is synchronous and single-threaded.  `TorchDispatchMode` is
  thread-local; a future asynchronous/worker-thread builder needs explicit mode
  propagation or must be held.
- The CPU E2E validates PyTorch 2.8.0.  Before any formal evidence, the frozen
  PyTorch 2.11.0+cu129 H20 environment must pass a no-paper, no-primary-memory
  smoke gate showing `.clone()` is observed as `aten.clone.default`, source and
  output handles are the actual call/return objects, CUDA BF16 values are
  unchanged, and all expected `N*60` edges are captured.
- This v1 is a mechanism prototype, not formal Qwen/H20 evidence.  It creates no
  launcher and changes neither v6 nor the manuscript.
