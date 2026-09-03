# QS UCP preview: R40 live-binding v18 formal H20

Status: **AWAITING USER CONFIRMATION; NO TRIAL CREATED**.

Fresh-audit verdict: `GO_TO_FORMAL_H20`. The approval report SHA-256 is
`f178c6a211352a97744b172e84c3b84dd13b76e4d29d6561dc8d5b474b189562`.

## Resolved Trial parameters

| Parameter | Resolved value |
|---|---|
| Inherit | Job `253976`, Trial `1911962` (successful v11 route) |
| Name | `liuhanzuo-qcomem-r40-v18-live-binding-8h20-20260828a` |
| Image | `artifactory.devops.xiaohongshu.com/media/redaccel:0.9.1-gpu` |
| Job type | `PytorchJob` |
| Queue | `408` / `RL_main` |
| Cloud | `6` / Alibaba Cloud Shanghai |
| Cluster | `53` / H20-141G bare-metal N zone |
| Resource package | `183`: 8 GPU, 170 CPU, 1800 GiB |
| Workers | `1 × 8 GPU` |
| Priority / restart / overuse | `0` / `0` / `true` |
| Mount | `/mnt/tidal-alsh-hilab/dataset/diandian` |
| Tags | inherited, no update |
| Command | atomically require a new scratch root, then `sleep infinity` for the frozen staged launch |

Cluster 53 had 128/960 H20 GPUs free at preview time. Historical Trials
`1911962`, `1905906`, and `1915001` are terminated and have no reusable Pod, so
a new Trial is required.

## Creation command

```bash
qs training create \
  --from-job-id 253976 \
  --from-trial-id 1911962 \
  --name "liuhanzuo-qcomem-r40-v18-live-binding-8h20-20260828a" \
  --image "artifactory.devops.xiaohongshu.com/media/redaccel:0.9.1-gpu" \
  --command "bash -lc 'umask 077; test ! -e /mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_debug/qcomem-r40-v18-live-binding-8h20-20260828a; mkdir -- /mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_debug/qcomem-r40-v18-live-binding-8h20-20260828a; cd -- /mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_debug/qcomem-r40-v18-live-binding-8h20-20260828a; exec sleep infinity'" \
  --queue-id 408 \
  --cloud-id 6 \
  --cluster-id 53 \
  --resource-package-id 183 \
  --job-type PytorchJob \
  --worker-num 1 \
  --priority 0 \
  --restart-num 0 \
  --env "QCOMEM_R40_NODE=1,QCOMEM_R40_SCOPE=TRUSTED_BINDING_LIVE_BINDING_V18_FORMAL" \
  --description "R40 v18 live-binding formal 8xH20; fresh-audit-approved; nonoverwrite." \
  --overuse \
  --mount-path "/mnt/tidal-alsh-hilab/dataset/diandian" \
  --yes
```

## Frozen inputs and no-overwrite transfer

The exact new input root is
`/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/r40-v18-fresh-audit-20260828a-inputs`.
It was verified absent. A running QS Code instance `qscode-2196` can access the
shared mount. After confirmation, the operator will atomically create this new
directory, upload exactly two files sequentially with `qs code upload`, verify
two regular single-link files and no other nodes, then make both files `0444`
and the directory `0555`:

| File | Bytes | Approved SHA-256 |
|---|---:|---|
| `r39-primary-compiled-dispatch-20260827f.tar.gz` | 752689 | `306daba7b79b045a306f0b22d6434143dd568cf1f3b6af7114ad1a4ebe1d6f82` |
| `r40-independent-live-binding-v18-self-contained-stage-20260828a.tar.gz` | 89893 | `a775458bd171bf365193800b886bc5140c5caf6fcb375959d1e2b5a119431475` |

`qs code upload` has no no-clobber flag. Therefore any upload interruption,
403, or hash failure forbids overwriting/retrying this directory; recovery must
use a fresh `...-inputs-b` root and be recorded as infrastructure handling.

## Formal runtime gates

After the new Pod is Running, it must reverify both shared-input hashes before
staging. The exact fixed clean-stage path is
`/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r40_v18_clean_20260828a`.
The fixed stage, result root, and one-shot marker must all be absent.

The staged launcher will explicitly unset both self-test flags and pass:

- `R40_H20_EXECUTION_AUTHORIZED=yes`;
- `R40_V18_FRESH_AUDIT_APPROVED=yes`;
- source-ledger approval
  `c28418d468aeb2b2269643584e28881d121c018ae2c11a45364a0a255e86e308`;
- overlay approval
  `a775458bd171bf365193800b886bc5140c5caf6fcb375959d1e2b5a119431475`;
- v6 approval
  `306daba7b79b045a306f0b22d6434143dd568cf1f3b6af7114ad1a4ebe1d6f82`.

Before science, the frozen launcher requires 162/162 tests with zero skips,
PyTorch `2.11.0+cu129`, a CUDA smoke, and exactly eight distinct H20 GPUs bound
by UUID. A pre-science failure is infrastructure/preflight, not a scientific
negative. A result becomes paper evidence only after formal completion and a
separate post-run audit.
