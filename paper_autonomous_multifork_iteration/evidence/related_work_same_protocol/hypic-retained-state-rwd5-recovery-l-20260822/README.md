# RW-D5 operational recovery L

The first K attempt is invalid before scientific output: static preregistration
stopped after stages 00/01 because three top-level model-view files were mode
0644. No server receipt, raw receipt, Store receipt, or formal cell exists; all
eight GPUs were terminally 0 MiB. K's scientific code and frozen design remain
unchanged.

The only repair already performed was `chmod 0444` on exactly
`model-artifacts.sha256`, `preprocessor_config.json`, and
`video_preprocessor_config.json`. Full before/after SHA, mode, uid/gid, size,
inode, device, mtime, and ctime are frozen in
`asset-mode-repair-receipt.json`. Bytes, hashes, sizes, inodes, devices, and
mtimes are identical; only mode and ctime changed. No top-level writable model
file remains.

`launch_recovery_l.sh` does not chmod or edit anything. It fail-closes on the
exact repaired state, verifies both the immutable K science manifest and this L
recovery manifest, requires fresh L output/instrumented/recovery paths, copies
the repair receipt, and then `exec`s the unchanged K launcher with explicit L
path overrides. Any identity/stat drift stops before K can start a server.

This L package is operational authority, not a new science freeze and not a
result. It must receive independent GREEN before use. Only K's 16 affected
Prefix/HYPIC cells are authorized.
