# R30C result — internal only

Status: **FAIL**. This post-run pointer is not part of the frozen R30C bundle or
its `SHA256SUMS`. It is not eligible for automatic manuscript use.

## Execution identity

- QS job/trial: `249885` / `1899487`
- Pod: `qs-249885-1899487-ai-1443683-master-0`
- Physical device: GPU1, NVIDIA H20-3e
- Frozen bundle SHA-256:
  `0b2de62302670bd7a3acddf65f9aeb51ad097ac3b7b2678f22d8c8b1c5d66b47`
- Static input-manifest SHA-256:
  `e03efe750df430ff7ea14de60a8607d5f5542e2818a9f9a3b79bd351456ef517`
- Static gate: PASS
- Formal candidates executed: exactly 1; RC=1
- Detached replays executed: exactly 1; RC=1

## Frozen-gate outcome

Both the candidate-integrated analysis and detached replay stopped at:

```text
fresh native allocations were not all zeroed
```

- Event 18: 11 allocated blocks, 5 zeroed, 6 missing from the zeroing receipt.
- Event 32: 5 allocated blocks, 2 zeroed, 3 missing from the zeroing receipt.

Other internal scheduler receipts were present but must not be reframed as a
passing result: ragged A/B admission at event 18 (1,105/2,209 prompt tokens),
cached A/B overlap at event 24, B/C turnover at event 32, free order A/C/B,
zero simultaneously-live block overlaps, and one-request sequential controls.

The output gates also fail independently of the first gate:

| Role | Native tokens | Sequential tokens | Max abs error | Mean abs error | Within 0.005 |
|---|---|---|---:|---:|---|
| A | `[5028, 279]` | `[5028, 279]` | 1.3378868 | 0.2797908 | no |
| B | `[780, 638, 25633, 571, 11]` | same | 2.7770004 | 0.5851390 | no |
| C | `[353, 557]` | `[353, 3738]` | 0.1939602 | 0.0146888 | no |

All six dense FP32 sidecars have the exact 248,320-entry model-vocabulary
dimension at every step. Their total size is 17,879,808 bytes. The post-hoc
read-only recomputation matched the serialized comparisons, but this does not
turn the failed detached replay into a pass.

## Remote immutable evidence

- Stage:
  `/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r30_native_batching_20260825c`
- Run:
  `/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/r30-native-batching-20260825c`
- FAIL archive:
  `/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r30_native_batching_20260825c/r30-native-batching-20260825c-FAIL-full.tgz`
- Archive: 8,435,620 bytes, 43 entries, SHA-256
  `afe9107112547b87764c28728fbf4aec1df8171c606d845560bc3f5d6459338b`
- Terminal ledger: 35 files, SHA-256
  `edd25090f8aad0a19f442e957521abfd48cbb8d6c2f7a5b558ee75e124fe0d10`
- Scheduler trace: 139 rows, SHA-256
  `e734eb7d49385137e5f4bb80f59e4e291bbf0664753a7f9227d9f78f3d4190bc`
- Outputs JSON SHA-256:
  `9f80be20c88547cfa800f3c9af2af1650fed2f2c1647831245d8b1b52b4a06ff`
- Post-hoc failure summary SHA-256:
  `94be1efa7369e861d17deda99cd616115f2cc21ca7a72fef759f4e1ada35d9d0`
- Formal stderr SHA-256:
  `62e1c4fedcbee9920d626d1700387af72745506c4f6ab5878047efc55fb62fbd`
- Detached replay stderr SHA-256:
  `ed9d964a6feeba6a62742353b98d27acf111c8a1230047d3493f5721f17bd749`

Phase rows / schedule rows / maximum scheduler membership were: warmup 9/2/1,
native 51/12/2, sequential A 20/5/1, sequential B 38/11/1, and sequential C
20/5/1. GPU1 returned to 0 MiB and 0% after execution. The node was not stopped
or otherwise modified by this lane. R30B was preserved unchanged.
