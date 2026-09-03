# R30 expanded captured-boundary oracle sweep

This is a new-output-only experiment.  It does not overwrite or reinterpret
the eight primary attention rows or four prior GDN rows.

Before candidate execution, the producer, NumPy reference, launcher, targeted
test, deterministic input manifest, final preregistration, and source ledger
are SHA-256 pinned.  Input preparation tokenizes two fixed PG-19 train windows
but does not load or execute the candidate model.  Candidate execution is
restricted to `CUDA_VISIBLE_DEVICES=3`; replay runs with an empty CUDA device
set and imports NumPy but no Torch, Transformers, vLLM, or qcomem module.

The fixed coverage is 2 inputs, all 10 full-attention layers, 12 of 30 GDN
layers, and 8 query positions per captured call: 20 attention operator rows
(160 query positions) and 24 GDN operator rows (192 token transitions).  Every
row has one clean decision and one preassigned seeded wrong-operator decision.

The claim boundary is intentionally narrow: agreement is conditional on the
captured post-RoPE attention inputs and post-native-q/k-normalization recurrent
inputs.  The experiment does not establish honest capture, independent
end-to-end semantics, upstream projections or convolution, downstream logits,
compiled dispatch completeness, unseen-fault coverage, or another stack.
