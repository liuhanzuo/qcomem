# P7 Semantic Lock

- Models: Qwen3-4B (12 probed blocks of 36) and Qwen3-1.7B (11 of 28).
- Quantization: single-block 4-bit weight-only HQQ, group size 64; all other blocks and LM head remain full precision.
- Predictor: standardized deployment-minus-calibration shift in activation tail-mass ratio, computed only on the full-precision model.
- Primary corrected labels: pad-masked degradation.
- Point associations: Spearman rho about +0.559 on 4B and about +0.12 on 1.7B.
- Depth relation: turnover-depth rho about 0.91 on 4B; masked partials about +0.27 and +0.01 with wide intervals spanning zero.
- Robustness records, LOO ranges, AUROC values, p-values, hashes, and provenance strings are locked.
- Three-host equality is pipeline determinism with effective run sample one, not independent scientific replication.
- Only two of eight preregistered cells were run; no population, universal trigger, or deployment-gain claim is supported.
- Tables, figures, equations, citation keys, labels, and appendix records are locked.
