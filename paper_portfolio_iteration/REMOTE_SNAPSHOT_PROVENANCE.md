# Remote snapshot provenance

Copied read-only on 2026-08-22 from SSH host `47.84.140.142`, beneath:

`/newcpfs/user/qixuan1/01_p5_share/run/iclr27_theory_k3_20260806_r1/paper`

| Local project | Remote source | Files | Public/remote `paper.tex` relation |
|---|---|---:|---|
| `a11-correlated-majority-vote` | `A11_correlated_mv_earlystop/audit_pack` | 113 | Exact match: `0a09788c045e0c3f12a08e2aba5c799872a476641755ff5ca67bbe53b729acb6` |
| `a2-erase-late-absorb-early` | `A2_lrphase` | 194 | Mismatch: public `d53d4efb...`, remote `e9d9a84c...`; evidence attribution requires reconciliation |
| `a2-subgroup-mix-ranking` | `A2_subgroup_mix_ranking_author_candidate_r1915` | 97 | Exact match: `258defa2a33a776fc377f920ef753dc5136e4882fc296c6e517e79e54f7e2d0c` |

Remote and local regular-file counts matched after transfer (113/194/97). Transfer
used an SSH tar stream after SCP proved inefficient for many small files. The
remote source was not modified.

An exact manuscript hash match establishes version identity, not scientific
validity. Each result still requires configuration, seed, independent-unit,
metric, and output-path checks before it can support a claim.
