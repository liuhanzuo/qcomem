# RW-D5 HYPIC retained-state bytes — frozen preregistration U

U is a minimal fail-closed recovery from T's pre-science asset-authority
rejection. It authorizes only Prefix Cache and official HYPIC
`transition_rope_recompute`: two modes over the same eight frozen workloads,
16 affected cells total. Full Recompute, CoMem, RR2, GDN, serving controls, and
all other arms remain forbidden. U is a stopped preregistration, not a result.
No GPU may run until a fresh independent audit returns GREEN on the exact U
manifest and STOP identities.

## Exact T invalid attempt

T ran its audited safe wrapper on Job 247574 / Trial 1879456. All 63 frozen
manifest rows passed, then the wrapper exited 1 with the exact terminal error
`ERROR: asset post-repair stat drift: model-artifacts.sha256`. No server was
started, no T run directory or instrumented checkout existed, and no raw,
Store, or formal output was produced. Read-only checks before the pod's later
platform termination found all 14 weight and all 9 artifact ledger entries
byte-exact, no writable top-level model file, no matching Python/SGLang
process, and eight GPUs at zero MiB with no compute process.

All three repaired files remained regular, root-owned, mode 0444, exact size,
and exact SHA, but their inode, device, mtime, and ctime changed together:

- `model-artifacts.sha256`:
  `444|0|0|778|10509881|1048686|1787380591|1787380591`, SHA `d7842468...`;
- `preprocessor_config.json`:
  `444|0|0|390|10509879|1048686|1787380591|1787380591`, SHA `27225450...`;
- `video_preprocessor_config.json`:
  `444|0|0|385|10509880|1048686|1787380591|1787380591`, SHA `7768af27...`.

This is a node-local `/tmp` rematerialization, not content or permission drift.
The platform later marked Trial 1879456 `Terminated` and the shell observed
exit 137, after the wrapper had already rejected the attempt with exit 1; that
later event is not the invalidation cause. The exact available facts and the
absence of a transcript SHA are recorded without invention in
`invalid-formal-t-job247574-trial1879456.json`.

The first intended U node, Job 247665 / Trial 1879665, never produced a pod,
remained `Uncommit`, and was terminated with zero execution. Future U execution
authority is instead Job 247668 / Trial 1879689. Neither node identity is
confused with T's Job 247574 / Trial 1879456.

## U asset identity contract

Three recovery designs were compared:

1. Keep old-node inode/device/time as authority. Rejected: it deterministically
   rejects a byte-identical node-local rematerialization.
2. Check only three file SHAs. Rejected: it omits kind, ownership, permission,
   size, the remaining 14+9 payloads, and within-preflight replacement races.
3. Use stable cross-node semantics plus two exact same-preflight physical
   snapshots. Selected: smallest fail-closed repair with no scientific change.

The stable cross-node identity is: regular non-symlink, exact SHA and size,
mode 0444, uid/gid 0 for all three repaired files, exact raw ledger SHAs, and
successful `sha256sum -c` for all 14+9 entries. Inode, device, mtime, and ctime
are recorded observations, never old-node authority.

After all stable asset and ledger checks, the frozen snapshot helper opens each
file with `O_NOFOLLOW`, compares lstat/fstat identity, hashes through the open
descriptor, and proves the descriptor and path did not change. The wrapper
takes this pre-snapshot, performs every root, safe-cwd, manifest/STOP, and exact
Python import-origin probe, then takes a second snapshot immediately before
publication and exec. The two canonical snapshots must be identical. No U
preflight directory, pass marker, copied authority, run directory, or
instrumented repository is created before that equality. The static builder
revalidates the published observation against the live model view and binds it
into preregistration.

## Scientific and lifecycle boundary

U preserves T/Q's science exactly. HYPIC's observed raw allocator multiset and
slot-3 duplicate anomaly remain explicit. Store MiB is only the overlap-aware
union of exact target-entry-owned physical tensor ranges; metadata, allocator
multiplicity, pool capacity, NVML, and process allocation are excluded. The
terminal contract still removes exact target entries, returns selected slots
to the unique full physical domain, and requires the original duplicate
fingerprint to remain unchanged. No global allocator-correctness or runtime-
safety claim is made.

Cleanup remains TERM, bounded poll, KILL survivors, reap, then explicit PID and
process-group absence. Before completion, all eight frozen GPU UUIDs must show
no compute applications and zero MiB, and no matching server, scheduler, or
client process may remain. The only authorized external entry is
`code/launch_hypic_retained_state_bytes_safe_u.sh`; the internal launcher is
forbidden as a direct entry point.

Frozen U validation before STOP:

- focused RW-D5 tests: 85/85;
- inherited same-protocol tests: 10/10;
- Bash syntax and Python compilation pass;
- exact T invalid/recovery-node separation passes;
- cross-node physical identity variation is accepted while same-preflight
  replacement, mode/SHA/size/owner, manifest, ambient, cwd/import, lifecycle,
  process, and GPU tamper regressions fail closed;
- scientific producer, client, blind replay, formal helper, and instrumentation
  bytes are unchanged from audited T;
- GPU submissions by U before STOP: zero;
- `main.tex` and paper tables changed by this workstream: no.

If any frozen byte changes, retire U and refreeze. T's invalid attempt and all
debug-only runs are not paper Store results.
