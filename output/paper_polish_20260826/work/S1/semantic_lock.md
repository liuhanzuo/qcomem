# S1 Semantic Lock

- Supports: ALPS, ELSA-4096, and ProxSparse.
- Recovery: identical 624,951,296-token (reported as 625M) joint sparse-weight plus rank-16 SLoRB recovery.
- Avg-9 ranges: 2.675 pp before recovery and 0.118 pp after recovery.
- Sampling reference: 0.52 pp marginal per-checkpoint binomial SE; it is not a pairwise/joint resolution threshold or equivalence margin.
- Per-item predictions for the three recovered endpoints were not retained; joint uncertainty is unidentified.
- Archived 5B SparseForge checkpoint: Avg-9 58.47, branch active; no folded/reprojected exact-2:4 export or pre/post-export measurement exists.
- Component effects of curvature scoring, annealing, and SLoRB are not isolated under matched budgets.
- Archived 5B corpus and manifest are unavailable/unverifiable in the accessible workspace.
- The 5B-versus-AST Avg-9 margin is +0.528 pp with checkpoint-z p=0.47 and is not robust to task removal.
- All equations, table cells, task counts, citation keys, labels, code identifiers, hashes, and appendix provenance records are locked.
