# R43 figures4papers revision record

Date: 2026-09-02

## Scope

This revision replaces the R42 dashboard-like opening figure with a
vector-first protocol geometry and adds a separate quantization-scope figure.
The figures were drawn with deterministic Matplotlib primitives following the
local `scientific-figure-making` (figures4papers) guidance.  The discarded
generic ImageGen draft is not used by the manuscript.

## Scientific boundaries encoded in the figures

- Figure 1 presents the Write--Retain--Read lifecycle, including the separate
  document-residual bypass, request-local lower-layer replay, ordered merge at
  depth `j`, suffix execution, and a subordinate ForkAudit validation rail.
- Figure 2 uses the exact 40-layer Qwen3.5 geometry at `j=7`: L0--L6 form the
  lower prefix, L3 is its only full-attention layer, and L7--L39 form the
  33-layer online suffix (24 GDN and 9 attention layers).
- The frozen retained-state policy is residual Q4 / L3 attention KV Q4 / lower
  GDN convolution-recurrent state Q8, with lower-layer cache bits
  `[8,8,8,4,8,8,8]`.
- Only retained per-document tensor state is packed.  Model weights and forward
  computation remain BF16; Q16 is the unpacked BF16 reference.
- Each 64-value group stores packed unsigned codes and BF16 scale/bias.  Both
  components feed Read-side dequantization to BF16 request state.
- Neither figure claims weight quantization, lower total VRAM, or TTFT speedup.

## Files

- Renderer: `figures/qcomem_figures_r43.py`
- Figure 1: `figures/qcomem_pipeline_r43.{pdf,svg,png}`
- Figure 2: `figures/qcomem_quantization_map_r43.{pdf,svg,png}`
- Manuscript: `main_r43_figures4papers.tex`

## Deterministic asset hashes

Two consecutive renderer executions produced identical hashes.

- `qcomem_pipeline_r43.pdf`: `e8988c69e196c6bce9e3c0b2e824056ba8bf9e0af1e110097ca24f9029c0b7b6`
- `qcomem_pipeline_r43.svg`: `509d54ab51eccdf609455c8852b465e054851c70df23bc049950281c53efb0a3`
- `qcomem_pipeline_r43.png`: `f422edbf7e7c89da0e8b1b7dc285fb1a85da8d53986fb08dba27f8e1aceaf828`
- `qcomem_quantization_map_r43.pdf`: `498c207092ce2c78beca2216e75dca7217835c776e9d095a5e38b93595ca8c00`
- `qcomem_quantization_map_r43.svg`: `4acc7e18a1055e4ab828b984aede7077b502d97b066960c321baf1f1424da6c9`
- `qcomem_quantization_map_r43.png`: `201e2ea30595daab9ce3d9d80c6745ddbe02f859c8affb0ef986710ae4fe59e7`

## QA

- Compiled with `latexmk` under the repository ICLR 2026 style.
- Inspected the affected manuscript pages in color and grayscale at final
  5.5-inch placement.
- Confirmed vector inclusion and embedded fonts.
- Confirmed no overfull boxes, undefined references, or undefined citations in
  the final log.  Historical underfull warnings in appendix tables and the
  bibliography remain unchanged.
