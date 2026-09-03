# Public baseline / remote candidate reconciliation

Audit date: 2026-08-22. This is a claim-level attribution audit, not a scientific validation of either manuscript.

## Immutable identities

| object | SHA-256 | consequence |
|---|---|---|
| public baseline and `pre_revision_01/paper.tex` | `d53d4efb6476df3ed35475a881d259f0e5cdd9e9acfa6abc165437682a825d72` | authority for this revision |
| remote candidate `paper.tex` | `e9d9a84ccd3072c3bc9eadcb462f1070a1c84305494bce7e5e6296598d0e3b3b` | separate candidate; never silent public evidence |

The source diff has 21 hunks affecting the abstract, figures/captions, theorem wording/proofs, empirical claims, scope, and appendices. A file-name match therefore cannot establish common code revision, configuration, execution, or intended claim.

## Remote package checks

`CONSISTENCY_JDIR=remote_snapshot/code python3 remote_snapshot/consistency_r790.py` passed `154 pass, 0 fail`. It verifies that its hard-coded numerical literals agree with remote JSON. It does **not** hash-link those JSON files or scripts to `d53d...`, reproduce a run, validate input data, or audit whether the remote source prose is identical to public prose. `hyperparam_audit_r838.py` returned 27 pass / 1 fail: its remote runtime claim is inconsistent for the two MNIST arms. The remote CIFAR data path (`remote_snapshot/early_traj_value_r201/data`) is absent locally.

## Claim-level decision table

| public claim family / concrete values | remote candidate material | can support public value/claim? | decision |
|---|---|---|---|
| T1 fixed-quadratic product and squared-step bound | remote has altered T1 wording and candidate scripts | No | A mathematical derivation is not a version link; public theorem is retained only at its stated formal scope. |
| T2 full-norm strong-convex tail return and conditional directional rate | remote changes T2 wording and adds different directional assertions | No | Remote assertions cannot establish the public theorem or its prerequisites. |
| T3 affine product; any nonlinear GD/SGD budget values | `pilot15`, `pilot14`, `pilot16`, `verify_t3_ou` JSON and checker | No | JSON literals are internally consistent only for `e9d...`; nonlinear transfer text is withheld. |
| synthetic figure coordinates, $R^2$, exposure, six-seed values | `pilot2/3/14/..._out.json` and remote checker | No | Same-looking names do not establish that the public figure was generated from these bytes. |
| real MNIST controlled-corruption results | `pilot17_mnist_fashion.py`, `pilot18_smallconflict.py`, outputs | No | No public script/output/config/pre-registration link. |
| CIFAR-10N values (damage, recovery, late injection) | `pilot19_cifar10n.py` plus full JSON | No | Candidate has a distinct source and unavailable local data path; not public evidence. |
| EQRA-loss positive endpoint | `eqra_loss_salvage.py` plus full JSON | No | Candidate-specific method, inputs, and source; it cannot validate public claims. |
| EQRA P3 clean-block false-positive result | `eqra_loss_p3_precision.py` plus full JSON | No | It is a useful risk signal for planning only, not public evidence. |
| remote package has files whose manifest/checker can be read | remote manifest and read-only checks | Yes, remote-only artifact fact | Registered as REM-CODE-001/C-REMOTE-001; no scientific transfer. |

## Closure rule

To change any “No” to “Yes,” the authors must provide, for each public numerical claim: (1) a public-version code revision or a cryptographic ancestry proof; (2) exact configuration, seeds, data/split hashes, and pre-registration; (3) output hash and extraction rule; and (4) a claim-level audit against the post-revision manuscript. A deliberate new public manuscript revision may cite a separately registered remote experiment, but it cannot retroactively make it baseline evidence.
