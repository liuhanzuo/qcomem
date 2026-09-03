# A32 structure audit

## Paper identity

This paper studies whether a length-associated signed confidence--accuracy gap in a fixed LLaDA-8B-Instruct evaluation bed reflects a causal effect of content-free prompt length, then tests how the read transfers to MATH500 under judge uncertainty.

## Claim thread

Problem (confidence may drift from accuracy with prompt length) -> observational GSM8K contrast -> exact confidence/accuracy decomposition -> within-item randomized content-free-length check -> judge-audited second-bed sensitivity -> bounded conclusion and limitations.

## High-impact issues

1. The abstract contains nearly the entire result ledger in one paragraph, obscuring the primary distinction between association and the within-item causal check.
2. The introduction repeats definitions and uses colloquial or promotional phrasing (for example, “high confidence is being bought” and “honest null”) where a direct measurement statement is clearer.
3. The second-bed section repeats the same reporting verdict several times before advancing the argument.
4. The conclusion restates multiple caveats in separate sentences instead of synthesizing the supported result, the unsupported transport claim, and the next measurements.
5. The appendix is evidence-dense but appropriately separated; the main text should refer to it without reproducing its full provenance vocabulary.

## Planned repair

Prioritize the title/abstract, introduction, second-bed synthesis, and conclusion. Preserve every result, interval, design qualifier, judge convention, and reporting status. Do not reorganize tables or evidence lineages in this pass.
