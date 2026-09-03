# C2 semantic lock

- CoMem is training-free on frozen Qwen3-8B; document chunks are written to one residual activation tensor per chunk at depth j, retrieved with BM25, and read by upper layers.
- Fixed-k read length is independent of document length, but external hidden/text/index storage grows with context; no system-level constant-memory claim.
- Matched read-depth direction: j=12 is less accurate than j=0 under the stated qa1 cells and conventions; the 4k +27 pp magnitude is selected and reader-convention-dependent.
- Depth sweep is pointwise, no multiplicity correction except the stated Bonferroni load-bearing comparison, downward overall but not strictly monotone.
- Five-arm E--A contrast varies depth in the computation graph while freezing named encoding/relocation factors, but retains an uncontrolled lower-band context-set difference; it is not fully deconfounded causality.
- Relocation endpoint contrasts are 0/100 discordant with an upper bound near 3 pp, not proof of no computational effect.
- E1 measured depth-memory difference is 0.3 GB at a 6.7k-token read; read/prefill break-even is about 26 reads/document; decode is excluded.
- No superiority over unimplemented prior systems, no end-to-end latency benefit, no trained-cache-compressibility claim, and no multi-hop qa2 result.
- Preserve all models, tasks, sample sizes, intervals, p-values, table cells, figure paths, citations, labels, and registered statuses.
