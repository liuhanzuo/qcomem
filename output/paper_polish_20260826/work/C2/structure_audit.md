# C2 structure audit

## Paper identity

This paper uses CoMem to measure a training-free trade-off between fixed-k retrieval and cached-depth reconstruction on a frozen Qwen3-8B backbone, finding that the short-read systems benefit comes from retrieval while deeper cached read-out incurs a fidelity cost.

## Claim thread

Two long-context scaling problems -> CoMem mechanism -> matched j=0 versus j=12 direction estimand -> depth-varying attribution study and its residual context-set limitation -> serving resource account -> bounded negative conclusion.

## High-impact issues

1. The abstract and introduction repeat the same “retrieval benefit, depth cost” conclusion at high density.
2. “Depth independently costs fidelity” and “pure-depth arm” overstate the attribution because the manuscript later acknowledges an uncontrolled lower-band context-set difference.
3. The contribution list mixes mechanism, measurement protocol, negative result, and systems accounting without a clear priority.
4. Several paragraphs carry long chains of caveats that can be reordered into claim -> evidence -> boundary.
5. The conclusion is accurate but can more directly distinguish measured read/prefill quantities from unmeasured end-to-end value.

## Planned repair

Tighten the abstract and introduction, conservatively narrow causal-attribution wording, clarify the three estimands, and sharpen the conclusion. Preserve all registered cells, intervals, operation counts, and appendix provenance.
