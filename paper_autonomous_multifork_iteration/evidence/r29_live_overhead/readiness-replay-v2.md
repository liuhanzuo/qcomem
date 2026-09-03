# Round 29 live-overhead offline replay-v2 readiness

Status: the postexecution replay-only repair is locally source-closed and
frozen. It has not been uploaded or executed. It authorizes no GPU work and no
change to the completed formal-b result or its raw audit artifacts.

## Immutable formal-b inputs

The one scientific execution is the existing formal-b run at
`r29-live-overhead-20260825b`. Its formal result SHA-256 is
`3ccf86e2233b560f003d965fdae05a8e3b0773e15976a05c8d70af881338bc22`.
The runner labeled that result completed, scientifically valid, and formally
eligible after one discarded warmup and five measured pairs. The formal stdout
and stderr hashes are respectively
`717c93f37b5e2e2b7313b694606ee629bb87fb7330258d74d13546d88ba6e76a`
and
`78b2e4662b8fbf3729516e7ad2fe00d1c0de64d18d0455abc8a9714589081a8c`.
The formal launcher log hash is
`0ea7cf1d885ad4976e0150df9dce6ffc37f00cf7db07e24b2b2e886ad0493c46`.

Before replay v2, the 13 formal-result/raw-audit entries are frozen by a ledger
whose raw SHA-256 is
`995a7cdfb0f502e4a0b4d603e59482f479d400c46e090ac7ff6908ca4d4b14fe`.
That exact ledger and all of its bound bytes must verify before replay v2.
The only existing stages are `00-started`, `01-preflight-passed`, and
`02-formal-complete`.

## Retained replay-v1 failure

The first offline replay used source
`43eb9deaf45352ff30d8aada2a38ae2e820c19e2c88a7a11dd486815781fc5cb`
and failed with `RuntimeError: request GDN receipt failed`. Its stdout is empty
with SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
its stderr SHA-256 is
`eee313d85c24b5f23f21e633a1973890d388e04f77cc8fd3b0de0b38d427ce97`.
It produced no replay JSON, stage 03, or terminal completed marker. Those logs
must remain unchanged.

The failure is a verifier schema bug. The receipt legally stores a raw
request-GDN witness. That raw object has 60 ordered rows and a `rows_sha256`,
but it does not have `rebound_tensor_count`; the latter is a derived replay
summary field. Replay v1 incorrectly required the derived field directly on
the raw object. The complete disclosure, including the fact that the formal
numeric outcome was already read before repair, is frozen in
`postexecution-replay-only-amendment-v2.json` at SHA-256
`faf773740b5a3f920a6192f9bb4cfad15aba9e5b9a8994d14ed6fa01bf223f17`.

## Replay-v2 correction and gates

Replay v2 is candidate-import-free for request-GDN interpretation. It locally
validates the exact raw record and row schemas, canonical Qwen 30-layer order,
60-row cardinality, canonical JSON row digest, lowercase SHA-256 tokens, and
the rebound/unchanged binding and storage-token relations. Only after those
checks does it derive the expected 60 rebound tensors. It does not import or
call the live candidate witness replay implementation.

The focused fixture now has the actual raw schema and deliberately omits the
derived count. Tamper gates cover an unrehashed row change, a reordered but
rehashed witness, an invalid rebound storage-token relation with a recomputed
row hash, a malformed token with a recomputed row hash, and an extra derived
field inserted into the raw schema. The complete fake formal replay still
passes.

Local checks completed before freeze:

- all nine focused tests pass;
- replay and test sources pass `py_compile`;
- the replay-only launcher passes `bash -n`;
- all nine entries in `source-code-replay-v2.sha256` pass `sha256sum -c`;
- the executable source ledger raw SHA-256 is
  `043efb9139ae579669efda3984071036c46df45971e00ad3692c803ad171d141`.

Frozen source hashes are:

- unchanged runner: `d4d6e04ba07e90438472b72412a6cb302c431e0626f495cae8ce18b2344c825b`;
- repaired replay: `53e70a1f1af5989c5e3a2a18d7097b37fcc83fa9821c18a54d05953328d7c54d`;
- focused test: `b39a203793f686a0f18c7a1c2944f9e0e8dae4ce292f06dc7f5e302d1d8c38e6`;
- unchanged formal launcher: `dd8a248d09c496dc572b27b5ddafee552a1de56cdbe3773a835dd7ffeb4381c5`;
- new replay-only launcher: `c528fca28ecc509dc31a48383a56cbb3ea6515b22f6896b9b4504e71a15524d9`.

## Upload set and exact CPU-only command

Stage a new immutable package at
`/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r29_live_overhead_replay_v2_20260825c`.
Upload exactly the repaired replay, focused test, replay-only launcher,
postexecution amendment, replay-v2 source ledger, this readiness record, and
the unchanged files referenced by the source ledger. Do not overwrite the
formal-b staged package.

Place the already-created 13-entry raw/audit ledger, without changing its
bytes, at
`$RUN_DIR/receipts/raw-audit-pre-replay-v2.sha256`. Then run only:

```bash
R29_PACKAGE_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r29_live_overhead_replay_v2_20260825c \
UPSTREAM_CODE_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_gpu_forkaudit_lifecycle_transfer_20260819c/gpu \
ENV_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/envs/vllm-cu129-v1 \
RUN_DIR=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/r29-live-overhead-20260825b \
PRE_REPLAY_V2_LEDGER=/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/r29-live-overhead-20260825b/receipts/raw-audit-pre-replay-v2.sha256 \
bash /mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r29_live_overhead_replay_v2_20260825c/gpu/launch_r29_live_overhead_replay_v2.sh
```

The launcher sets `CUDA_VISIBLE_DEVICES` to the empty string for compilation,
tests, and replay. It never invokes the formal runner. It writes only new
v2-named replay/log/stage/terminal-ledger files and fails if any target already
exists. A replay-v2 failure must be retained and does not authorize a GPU
rerun. No overhead result is citable until replay v2 passes and the new
terminal ledger is independently hash-closed.
