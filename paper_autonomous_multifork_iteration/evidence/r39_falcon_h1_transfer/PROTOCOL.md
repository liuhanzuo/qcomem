# Frozen protocol: R39 Falcon-H1 hybrid-state transfer

## 1. Question and decision rule

The experiment asks one bounded question: under the single frozen
Falcon-H1/Transformers/H20 configuration below, does the persistent lossless-Q16
fork preserve the exact semantics and mutable-state isolation of a deep
materialized fork?

A rank passes only if all registered identity, geometry, acquisition, dispatch,
input, ownership, immutability, control, token, logit, and 144-family state gates
pass. The aggregate passes only if all eight ranks pass. There is no partial,
mean, approximate, or threshold-based success. A missing object, incompatible
API, nonzero numeric difference, incomplete receipt, or authority drift makes
the formal cell invalid.

## 2. Frozen identities

- Model: `tiiuae/Falcon-H1-0.5B-Base`.
- Canonical Hugging Face revision:
  `59fb76e8c5d3fc7441b062be638e1ba0afd5c687`.
- Official ModelScope acquisition revision:
  `a475c769e108fd1dc6cfe41e342305d36431ef20`.
- Model weight SHA-256:
  `865a1e864b3fe6495ec37256e1fdec8cd1d254b607eab29141e7263791172ce6`.
- Tokenizer SHA-256:
  `605c664925653e3fbf2f35ea063847db441ba5b7a6af04378880409c3ab311fc`.
- Runtime: registered image label `vllm-cu129-v1`, Transformers 5.14.1,
  BF16 model weights, eager attention, official `DynamicCache`, one process and
  one CUDA stream per rank, and eight distinct H20 UUIDs.
- Trust-remote-code is false. The package installs no optional Mamba,
  causal-convolution, or Flash dependency. `USE_HUB_KERNELS=NO` must be set
  before any Transformers import, disabling Hub-loaded wrapper kernels.

The three official Transformers source files and their exact SHA-256 values are
frozen in the static registration and checked by both producers. Model
construction happens before the official Falcon-H1 module exposes its global
fast-path flag. Therefore each producer must complete `from_pretrained`, assert
that `is_fast_path_available` exists and is boolean, record its value, set it to
`False`, and only then issue any forward. The selected Mamba route is the
official `FalconH1Mixer.torch_forward`; the package does not replace that code.

## 3. Model geometry and state families

The registered model has 36 `hybrid` layers, hidden size 1024, model vocabulary
32784, and tokenizer vocabulary 32768. Every layer must expose self-attention,
Mamba2, and feed-forward components. The split depth is exactly 18.

At sequence length `S`, every active layer has exactly these four families:

| Family | Frozen shape | Frozen dtype |
|---|---:|---:|
| KV key | `[1,2,S,64]` | BF16 |
| KV value | `[1,2,S,64]` | BF16 |
| convolution state | `[1,1792,4]` | BF16 |
| Mamba2 recurrent state | `[1,24,64,128]` | FP32 |

Each family is bound to a `(layer_index, family, state_index)` identity and a
content SHA-256. Lower and suffix receipts must compose into the exact ordered
Cartesian product of 36 layers and four families: 144 rows, with no omission,
duplicate, extra active layer, or relabel.

For a suffix cache, layer 0 is empty while layers 18--35 contain history.
Consequently `create_causal_mask` must receive `layer_idx=18` for the suffix and
`layer_idx=0` for the lower half. The package includes a focused 64-token
prefill plus 8-token continuation test requiring a suffix mask width of 72 and
byte-identical output to the same direct mask call.

## 4. Frozen inputs and schedule

Ranks 0--7 use eight distinct PG-19 train objects whose full bytes and SHA-256
are frozen. Each object is decoded as strict UTF-8 and processed by the official
Falcon tokenizer JSON with `tokenizers==0.22.2`, `add_special_tokens=False`, and
no external normalization. Windows are selected only after full tokenization:

- document tokens `[197,261)` (64 tokens);
- request 0 tokens `[477,485)` (8 tokens);
- request 1 tokens `[509,517)` (8 tokens).

No OOV filtering, clipping, modulo transform, or alternate-window reselection
is allowed. The static builder independently asserts every ID is in
`[0,32784)` and binds each sequence as little-endian int64 bytes.

Every request follows the identical chunk schedule `[64,8,1]`: document
prefill, one query chunk, then one-token continuations. Two greedy steps are
recorded. Fanout is `N=1` and `N=2`; request ordering and interleave are fixed.

## 5. Candidate arms

The deep-materialized arm independently constructs a complete exact lower
prefix state for each child. The persistent arm constructs one exact lower
state, packs it with lossless 16-bit representation, and forks private mutable
children. This experiment uses 16 bits for residual, attention, and linear
categories with group size 64. Because the representation is lossless here,
the cell does not claim differential quantization behavior among KV,
convolution, or recurrent families.

Before and after all descendants run, the persistent base must have identical
full content hashes. Every mutable tensor view in a child must be disjoint from
every base view, and all child pairs must be disjoint. For `N=2`, all peer pairs
must actually be compared; an empty or vacuous comparison fails.

## 6. Independent official reference

The reference producer uses only official `AutoModelForCausalLM` and official
`DynamicCache` under the same `[64,8,1]` chunks. Its source is AST-audited to
forbid imports of the adapter, candidate runner, and Q16 dependency. Calls to
`__import__`, `exec`, `eval`, `importlib.import_module`, and
`importlib.util.spec_from_file_location` are also forbidden. The launcher gives
the reference an empty `PYTHONPATH` and Python isolated mode (`-I`). The
reference additionally requires that the candidate, adapter, and Q16 module
names are unavailable through import discovery. The launcher waits for all
eight reference processes to finish and only then starts any candidate process.

At both greedy steps of both requests, the reference and candidate each write:

- the generated token ID;
- the complete `[1,32784]` CPU little-endian FP32 logit vector and content hash;
- the ordered 144-row state-family receipt and every family content hash.

Sidecars require gap-free exact byte coverage, finite values, exact SHA-256,
and independently recomputed argmax.

## 7. Exact gates

For every rank, fanout, arm, request, and step, detached replay requires:

- exact token equality;
- exact full FP32 logit bytes (registered max-absolute and relative-L2
  tolerances are both zero);
- exact equality of all 144 family rows, including shape, dtype, binding, and
  content SHA-256;
- exact deep-materialized versus persistent trajectories;
- exact candidate versus independent official trajectories;
- exact request-0 trajectory across `N=1` and `N=2` within each arm;
- persistent-base content immutability and complete private mutable ownership.

## 8. Controls

Five frozen injected controls must pass clean state and fail the registered
first predicate after mutation: cache alias, family omission, position drift,
family relabel, and reference candidate-import. A sixth, separately reported
detector flips one prefix-content bit in a fresh packed state and must detect
the content change while storage identity stays fixed. The latter validates the
base-content detector; it is not counted among the five frozen controls.

## 9. Acquisition, closure, and nonoverwriting paths

The frozen ModelScope tree must exactly equal the live tree at the pinned full
commit. Each file uses a fresh zero-byte temporary per attempt, official HTTPS
hosts only, response status 200, exact `Content-Length`, no `Content-Range`, and
exact frozen size/SHA-256 before atomic rename. Neither partial reuse nor a
pre-existing model root is accepted. The completed model tree and directories
are made read-only and attested before and after execution.

The stage, model, and run paths in `freeze.json` are versioned and required to
be absent before use. Source, static, freeze, model, and GPU-assignment
authorities are captured before execution; source/static/freeze/model
authorities are reverified at the terminal. Detached replay precedes an artifact
ledger, the terminal checksum ledger, and finally `COMPLETE`.

## 10. Claim boundary

Even a fully positive aggregate authorizes only the bounded exact relational
result defined in Section 1. It does not authorize runtime independence,
compiled/fused dispatch, other Falcon revisions, other architectures, latency,
throughput, capacity, memory-saving, quality, production scheduling, or
continuous-batching claims.
