# Round 29 true-concurrency readiness v3

Status: dependency-closed, hash-consistent, and ready for an independently
approved second one-H20 attempt. No second GPU execution has occurred.

The immutable scientific design remains
`5c9fc301ec63e2702d097b9dbe9c68758164c653c6c7b53fedad290428a9a96`.
The first v2 attempt (QS Job 249885 / Trial 1898483) completed model work and
wrote two sidecars, but failed before committing `formal-result.json` because
PyTorch's private `_CUuuid` object was placed directly in the hardware JSON.
It produced only stages `00-started` and `01-preflight-passed`; no formal
result, independent replay, or terminal ledger exists. The traceback and the
two sidecar size/SHA records were disclosed, but no overlap or lifecycle
outcome was read. That attempt remains an infrastructure failure and is not
formal scientific evidence. Full partial-disclosure metadata is frozen in
`pre-second-execution-amendment-v3.json`.

The v3 runner converts the device UUID to `None` or `str`, normalizes the
Torch/CUDA version scalars, and strict-JSON serializes the complete formal
envelope before atomic write. The model calls, inputs, serialized/concurrent
treatments, CUDA-stream overlap gate, cancel/scrub/replacement schedule,
oracles, success rules, resource request, and claim boundary are unchanged.

Post-amendment readiness checks:

- `sha256sum -c source-code-v3.sha256` passes all nine entries;
- all three Python sources compile;
- the launcher passes `bash -n`;
- all seven focused tests pass, including a complete formal-shaped payload
  containing a mocked `_CUuuid`-like object;
- the CPU/mock lifecycle and positive-overlap arithmetic gate pass;
- the selected `7620...` lifecycle source ledger remains the frozen upstream
  dependency closure.

Executable ledger raw SHA:
`4afc29d3d154f66518710a2b3f00a8a4f31d13da2de0e0c425ed91d807489405`.
Only `source-code-v3.sha256` is executable for the second attempt; v1 and v2
are retained as history.

The second attempt must use the fresh directory
`/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/r29-true-concurrent-lifecycle-20260825b`.
Resource isolation is unchanged: one exclusive H20, one process, no server
port, no collective, read-only shared model/data assets, and private output,
Triton-cache, and TorchInductor-cache paths. Expected wall time remains about
eight minutes.
