# R39 Falcon-H1 scope and evidence audit

## Status at freeze

This package is a preregistered experiment, not a result. At freeze time no
Falcon GPU cell had been executed and no Falcon scientific output had been
inspected. `freeze.json` records both facts. Failure or incompatibility must be
reported as a failed/invalid formal cell; nothing in this package permits
changing the model, tokens, paths, thresholds, state census, controls, runtime
route, or reference after viewing output.

## What is genuinely new in this cell

The cell moves the exact-fork audit to a hybrid Falcon-H1 architecture in which
every decoder layer has attention KV state, convolution state, and Mamba2
recurrent state. It therefore tests a materially different registered state
census: 36 layers, four families per layer, 144 complete family rows, and a
depth-18 split. The inputs are freshly derived from the same class of PG-19 raw
objects using the frozen Falcon tokenizer; no token IDs from another model are
reused.

The candidate is compared with an independent official full-model execution at
the same chunk boundaries. This is stricter than a token-only oracle: the
full-vocabulary FP32 logit bytes and all 144 state-family contents must be exact
at each step.

## Reused authority and why it is bounded

The only reused implementation is the immutable, hash-bound generic
`qcomem_torch.py` dependency for lossless Q16 packing, cache cloning, and state
containers. Falcon-specific execution, geometry, masking, dispatch, and state
census live in the new adapter and are source-manifest bound. The independent
reference cannot import either component. Reuse therefore does not make the
reference circular, but it also does not authorize claims about a new
compression algorithm: Q16 is explicitly lossless in this cell.

## Trusted producer boundary

The formal evidence still trusts the registered Python producers and the
frozen official Transformers source bytes to serialize honest tensors. The
package reduces this trust with independent producers, an empty reference
`PYTHONPATH`, AST import/dynamic-execution bans, full sidecar coverage,
per-family content hashes, pre/terminal authority closure, injected controls,
and detached torch-free replay. It does not provide a hardware instruction
trace, compiled-kernel binary fingerprint, remote attestation, or a second
independent serializer. Those absences are explicit in receipts and remain a
limitation.

## Runtime boundary

This is intentionally one runtime cell: registered image `vllm-cu129-v1`,
Transformers 5.14.1, H20, BF16 weights, eager attention, and the official
Falcon-H1 naive Mamba path. The package verifies exact official source hashes
and forces `is_fast_path_available=False` only after model construction and
before any forward. It sets `USE_HUB_KERNELS=NO` before the first Transformers
import and runs the reference in Python isolated mode. It neither installs nor evaluates optional Mamba,
causal-convolution, Flash, compiled, or fused paths. A positive result is not
evidence of runtime independence.

## Input boundary

The corpus slice is fixed and small by design: eight distinct PG-19 train
books, one 64-token document and two 8-token queries per book, followed by two
greedy steps. Raw source hashes, strict UTF-8 decoding, tokenizer revision/hash,
tokenizers library version, and token windows are all frozen. This establishes
deterministic bounded semantic equivalence, not long-context quality,
representative language coverage, robustness to held-out faults, or benchmark
effectiveness.

## Ownership and fault boundary

The positive path checks base immutability, child/base disjointness, all child
pair disjointness, family completeness/binding, and exact trajectories.
Constructed controls show that the registered predicates detect aliasing,
omission, position drift, relabeling, candidate import into the reference, and
prefix-content mutation. They are detector controls, not naturally occurring
or held-out system faults. The cell therefore must not be described as broad
fault-tolerance evidence.

## Claims explicitly prohibited

Regardless of outcome, this package cannot support claims of:

- runtime-independent, stack-independent, or model-family-wide correctness;
- optional, fused, Flash, Mamba-kernel, causal-convolution, or compiled-path
  equivalence;
- latency, throughput, capacity, memory reduction, or production efficiency;
- long-context task quality or preservation of any benchmark score;
- scheduler, concurrency, continuous-batching, or multi-tenant correctness;
- robustness to organic, adversarial, or held-out faults;
- cryptographic proof that an untrusted producer or GPU executed the claimed
  computation.

If all gates pass, the defensible statement is only: under this frozen
Falcon-H1/Transformers/H20 naive-path configuration and registered short PG-19
workload, the persistent lossless-Q16 and deep-materialized forks are exactly
equal to each other and to the independent official execution on tokens, full
FP32 logits, and all registered mutable state families, while satisfying the
registered ownership and immutability predicates.
