# R39 dual-producer formal H20 result

Status: **PASS**. The formal launcher process (`PID 36206`) exited naturally. The outer run reached stages `00` through `08`, and both producer runs reached their internal stages `00` through `06`. No QS resource was stopped, killed, or evicted during collection.

## Frozen archive

- Remote run tree: `/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/r39-dual-producer-repeat-20260826a`
- Remote archive: `/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/r39-dual-producer-repeat-20260826a-formal-complete.tar.gz`
- Local archive: `r39-dual-producer-repeat-20260826a-formal-complete.tar.gz`
- Archive size: `567094` bytes
- Archive SHA-256: `85bf3cfd3e960c6d9d91d5907ba0a7e1fe62d2838711eb250883f1133d2ab5cd`
- Remote and local archive mode: `0444`
- Archive publication: a same-filesystem temporary archive was published through a non-overwriting hard link after confirming the final remote targets were absent. The local copy was decoded to a fresh partial name, checked against the remote size and digest, made read-only, and moved with no-overwrite semantics.
- Transfer: marker-delimited base64 from the QS pod. The decoded local size and SHA-256 exactly matched the remote values.
- Tree completeness: the live terminal tree and archive both contained `87` files and `36` directories (`123` tar members). The archive has one top-level directory, `r39-dual-producer-repeat-20260826a`, and zero absolute-path or `..` members.

The archive SHA-256 is the binding for the complete tree, including generated runtime-cache files, the terminal ledgers themselves, and `stages/08_complete`.

## Fresh-extraction verification

The downloaded archive was unpacked into a newly created temporary directory before verification. All ledger paths were resolved relative to that extracted copy.

| Ledger | Rows verified | Ledger SHA-256 | Result |
|---|---:|---|---|
| Outer `receipts/terminal-files.sha256` | 63/63 | `21a0189f227e418957f9e9015bca982ec1cb459a5daf848bc90b57b55fb9beb2` | PASS |
| Producer A `receipts/terminal-files.sha256` | 17/17 | `d42baabec0713cc90a854350d99dcff9ea20ae519465a1f9c65a968d2e3a462d` | PASS |
| Producer B `receipts/terminal-files.sha256` | 17/17 | `80fd968721fe3f6fe1def1ad072796454507e6286425a654d5996f9ece444297` | PASS |

The final dual summary SHA-256 is `e58571615119f51a73c7a826f087c5d406664c8e88a6ca6312db7229f0ba152e`, with `passed=true`.

## Scientific closure

- Producer PIDs were distinct: A `36666`, B `37721`.
- Observer PIDs were four-way distinct and disjoint across producers: A `{36949, 37280}`, B `{38059, 38330}`.
- The four observer-session commitments were also distinct: `1cdb8a9a4924027415bddf368e24ca6a801fb5ec34bdfb33a4a6da3215a67134`, `e4c7ebda1859c93fe84bc438a3dd2859dac3eecd6abb15033f834323acf32028`, `2d7f1c9f3a67698de0da571bc00633551c9874f8b0f0329deea9158186b6d69f`, and `c0c86d0720ae1b76ad63461705cac82e0b4d726cfe8474bc612ce388c5593f77`.
- All captures reported process separation through `torch-cuda-ipc-reduction`.
- Each producer supplied six matched captures with 180 unique slots per capture: `1080/1080` semantic sets, content SHA-256 values, and stable descriptors matched exactly across producers.
- Each producer supplied exactly `96660` independently reconstructed pairwise relations (`6 x 16110`); all six relation-vector digests matched across producers. Numeric tolerance was zero and semantic fallback was disabled.
- Raw result SHA-256 values were distinct and receipt-bound: A `891f640a64d28061b14d6e65c97d32f74c08b39d4e5fc1985a3ac856ec6b4824`, B `eb6208d380b982f13c323f023fd0653fb1eead9a5a589b6c25911098b5917978`.
- Both R33 replays passed with the frozen protocol and exact `1080` row / `96660` relation counts. Replay-file SHA-256 values were A `b7798699587912ecf260b5d29e8dbdc9b52a22df1633cacc78bfb29a9e2b7cba` and B `2d65136fdbc5542281bb4c1cc755bcd451ff50f1ba70baaa92538f1c59ead125`.
- The R39 preexecution census contained 180 slots. Its file SHA-256 was `88df8c35bbc876a62196f5b2de9795280e290d7d5d0f1cf953ab5847dfb76a52`, and its independently recomputed slot-semantic SHA-256 was `31d788fd9e39f2a8431edf695d732c46fffa750e27569484d199224decedf65a`. Both producer audits were census-bound and passed `1080` row / `96660` relation observations without using producer manifests or rows as the expectation.
- The census file was completed at `2026-08-26 23:16:40.232-23:16:40.263 +08:00`, the freeze marker at `23:16:40.333`, and producer A started at `23:16:40.608`; producer B started only after producer A completed.

Frozen provenance hashes are: preregistration `fe3583907cd0cfadb4045509d3a103ab64052452e00184edcf72835742510b72`, source ledger `24296bf0902d9687c7029654f6d8e1406582d33f548464e07c6e8d63a9a190b1`, and slot protocol `6d062e9a40e9ac51384031b2e01b0d549680f3549dc8be07e6f27edc4733e37b`.

## Claim boundary

The supported claim is limited to the following frozen statement:

> Under one frozen Qwen3.5/H20/PyTorch stack and one frozen input/protocol, two fresh serial producer executions with fresh out-of-process receivers exactly reproduced all 1,080 semantic slot observations, bytewise content digests, stable descriptors, and 96,660 independently reconstructed relation labels per execution.

The result does **not** support malicious-producer resistance or trusted-producer elimination; OS, driver, allocator, or runtime attestation; proof against adversarial live-tensor substitution under a correct opaque slot ID; independent model re-execution by the receiver; cross-model, cross-runtime, cross-hardware, or statistical generality; continuous-batching serving behavior; or absence of transient writes between paused captures.

## Sole caveat

The outer terminal ledger was written at `2026-08-26 23:19:38.811 +08:00`, followed by `stages/08_complete` at `23:19:38.827`. Consequently, `stages/08_complete` is present in the complete archive but is not one of the outer ledger's 63 rows. The outer ledger does cover `stages/07_scientific_execution_complete` and all ledger-scoped scientific outputs. This is a final-marker ledger-closure caveat, not a failure of the reported scientific result; the archive-level SHA-256 binds the marker and the entire archived tree.
