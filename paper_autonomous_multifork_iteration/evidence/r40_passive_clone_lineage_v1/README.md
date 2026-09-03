# R40 passive clone-lineage v1

Status: **CPU mechanism PASS; formal GPU/H20 not run; manuscript support false.**

This non-overwriting package addresses the fundamental discriminability blocker
recorded by `r40_independent_live_binding_v6_clean_formal`.  It observes the
unchanged frozen production helper at the PyTorch dispatcher boundary, rather
than trusting a producer-emitted mapping or changing tensor content.

The test suite dynamically compiles and executes the exact frozen
`_seed_tensor_memo` and `_prepare_request_gdn_base` functions from the archived
Qwen3.5 builder.  Its hard case gives all 60 persistent coordinates identical
BF16 content and geometry, then makes coordinate A actually clone source B
before assignment.
The production helper completes, while passive lineage fails on the observed
source coordinate.  Correct materialization, exact borrowed aliasing and
source/value non-mutation pass.

Run locally with:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

See `DESIGN.md`, `mechanism-assessment.json`, `cpu-e2e-result.json` and
`source-code.sha256` for scope and frozen bindings.  No QS, GPU, launcher,
existing evidence package or paper file was touched.
