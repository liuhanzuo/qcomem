# Anonymous GDN transition-oracle replay

This package replays the four preregistered recurrent-core rows using the
candidate-import-free NumPy FP32 reference.  It contains the 40 original NPY
sidecars, the exact preregistration, the executed reference source, and the
registered result.  The capture manifest differs from the registered GPU
capture in only one reviewer-safety transformation: five private module paths
are replaced by `anonymous-environment/<module>`.  `PROVENANCE.json` binds the
registered raw capture, result, source, and terminal ledgers.

Run from this directory:

```sh
./run_replay.sh
```

The command first verifies the complete package manifest, then independently
recomputes all four clean rows and four seeded faults.  Because IEEE-FP32
`einsum` reduction order can differ slightly across NumPy/BLAS builds, replay
requires identical Boolean decisions and checks each reported scalar within a
declared portability allowance of `1e-6`; this is far below the preregistered
clean output tolerance `0.005`.  It reports recorded and replayed maxima rather
than rewriting the registered result.

Boundary: the oracle starts from captured post-native-q/k-normalization arrays.
It does not validate upstream projections, q/k normalization, causal
convolution, gated RMS normalization, output projection, or end-to-end logits.
