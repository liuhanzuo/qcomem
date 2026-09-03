# Anonymous GDN transition-oracle replay

This package replays four recurrent-core rows retained from the original plan
using the candidate-import-free NumPy FP32 reference.  The successful run
followed an invalid clean control and changed the q/k capture boundary and
query scale.  Because the retained amendment bytes have no mechanically
verifiable immutable pre-execution binding, this package treats the result as
amended, post-hoc bounded captured-input evidence rather than predeclared
evidence.  `gdn_transition_oracle_preregistration_execution_diff.json` records
the exact field diff, chronology, and classification basis.

The package contains the 40 original NPY sidecars, the exact original
preregistration, the executed reference source, and the registered result.
The capture manifest differs from the registered GPU
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
declared portability allowance of `1e-6`; this is far below the retained
original-plan clean output tolerance `0.005`.  It reports recorded and replayed maxima rather
than rewriting the registered result.

Boundary: the oracle starts from captured post-native-q/k-normalization arrays.
It does not validate upstream projections, q/k normalization, causal
convolution, gated RMS normalization, output projection, or end-to-end logits.
