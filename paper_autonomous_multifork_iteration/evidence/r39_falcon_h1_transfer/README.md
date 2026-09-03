# R39 Falcon-H1 bounded transfer package

This directory freezes a not-yet-run, fail-closed experiment for one additional
architecture/runtime configuration. Its only possible positive conclusion is
bounded exact relational transfer for `tiiuae/Falcon-H1-0.5B-Base` at Hugging
Face revision `59fb76e8c5d3fc7441b062be638e1ba0afd5c687`, under the registered
Transformers 5.14.1 H20 environment and the official Falcon-H1 naive path.

The package was constructed before Falcon GPU execution. It contains no Falcon
scientific result and does not authorize a claim of runtime independence,
compiled-path support, performance, memory savings, long-context quality, or
model-family generality.

## Frozen cell

- Eight independent ranks, each bound to a distinct H20 UUID and a distinct
  PG-19 train book.
- The original PG-19 UTF-8 objects are tokenized with the frozen Falcon
  `tokenizer.json` (`tokenizers==0.22.2`, no special tokens). Fixed token windows
  are document `[197,261)`, query 0 `[477,485)`, and query 1 `[509,517)`.
  No out-of-vocabulary filtering or token reselection is permitted; every
  frozen ID is asserted to lie in `[0,32784)`.
- Thirty-six hybrid Falcon-H1 layers, split at depth 18. Every layer contributes
  four mutable families: KV key, KV value, convolution state, and Mamba2
  recurrent state. A complete full-model receipt therefore has 144 rows.
- Fanout `N in {1,2}`, two independent queries, two greedy steps, and an exact
  chunk schedule of `[64,8,1]`.
- Candidate arms are deep materialization and persistent lossless Q16 fork.
  Q16 is a bit-preserving representation in this experiment; it is not a claim
  that KV and recurrent families receive scientifically distinct quantization.
- The independent reference is official `AutoModelForCausalLM` plus official
  `DynamicCache`. It cannot import the candidate adapter, candidate runner, or
  immutable Q16 dependency, and it cannot dynamically import or execute them.

All semantic gates use zero tolerance: generated token IDs, complete
full-vocabulary CPU FP32 logit bytes, and every one of the 144 family content
hashes must match at every request and step. Persistent base contents must be
unchanged; mutable children must be disjoint from the base and from every peer;
family membership and shapes must be complete and correctly bound.

## Runtime and model authority

The canonical model identity is the Hugging Face commit above. Acquisition is
from the official public ModelScope repository at full commit
`a475c769e108fd1dc6cfe41e342305d36431ef20`; the seven scientific load files,
including `model.safetensors` and `tokenizer.json`, are byte-identical across
the two frozen sources. The weight SHA-256 is
`865a1e864b3fe6495ec37256e1fdec8cd1d254b607eab29141e7263791172ce6`.

Every acquisition attempt starts in a new empty temporary file, requires a
complete HTTPS response with status 200 and exact `Content-Length`, verifies
the frozen size and SHA-256, and only then atomically renames the file. Range
requests, append/resume, symlinks, pre-existing output roots, and replacement
of any authority object are rejected.

The runtime source bytes are also frozen:

- `modeling_falcon_h1.py`: `e90bf774524e9b66284ad1c5528c35339271a187f58f16ba2d45c97f4bc6b5bd`
- `cache_utils.py`: `ee7902fbd031ed332b5e26d07756a33f09b5c90a435b8363b9330876dc33ce0e`
- `masking_utils.py`: `5f48e428ea02d1b6008acb45c147fcdb4eba89deea69627744662aa05da1b9f2`

`AutoModelForCausalLM.from_pretrained` must finish before the package checks
that the official module-level `is_fast_path_available` flag exists and is a
boolean. It then records the pre-force value and sets the flag to `False`
before any forward call. No Mamba, causal-convolution, or Flash dependency is
installed by this package. `USE_HUB_KERNELS=NO` is exported before the first
Transformers import, so Hub wrapper/kernel loading is also disabled. Attention
is fixed to `eager`.

## Package map

- `preregistration/static-preregistration.json` is the scientific freeze.
- `preregistration/{huggingface-tree,modelscope-tree,cross-source-equivalence}.json`
  bind both official model sources.
- `preregistration/pg19-tokenized-inputs.json` binds raw-source hashes,
  tokenizer identity, deterministic windows, and Falcon token IDs.
- `preregistration/source-manifest.json` hashes all executable and documentary
  package inputs plus the immutable Q16 dependency.
- `preregistration/freeze.json` binds the static and source manifests, official
  runtime sources, and fresh nonoverwriting remote paths.
- `executed_source/run_r39_falcon_reference.py` is the isolated official
  reference producer. It must run with Python isolated mode (`-I`) so the
  candidate directory is absent from `sys.path`.
- `executed_source/run_r39_falcon_candidate.py` is the candidate producer.
- `executed_source/replay_r39_falcon_transfer.py` is the torch-free detached
  verifier.
- `executed_source/launch_r39_falcon_transfer_8gpu.sh` is the formal launcher;
  it does not provision or manage a pod.

The launcher runs package/source/static/freeze checks, acquires and seals one
fresh model snapshot, runs all eight reference processes to completion, and
only then launches the eight candidate processes. It closes model and package
authorities again, performs detached replay, writes a complete artifact ledger,
and creates `COMPLETE` only after the terminal checksum ledger.

## Frozen controls

Exactly five injected controls must detect their registered first predicate:

1. mutable-cache alias -> `PRIVATE_MUTABLE_STORAGE`;
2. state-family omission -> `STATE_FAMILY_COMPLETENESS`;
3. position/current-length drift -> `POSITION_CANONICAL`;
4. family relabel -> `STATE_FAMILY_BINDING`;
5. reference importing candidate code -> `REFERENCE_IMPLEMENTATION_INDEPENDENT`.

A separate prefix-content mutation detector must trip
`PERSISTENT_PREFIX_IMMUTABLE` without changing the storage identity.

See `PROTOCOL.md` for the full scientific contract and `SCOPE_AUDIT.md` for the
claim boundary and fail-closed interpretation.
