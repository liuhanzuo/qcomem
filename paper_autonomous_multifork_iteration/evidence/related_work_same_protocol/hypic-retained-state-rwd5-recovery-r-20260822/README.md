# RW-D5 Recovery R — controlled launch authority for scientific freeze Q

R changes no scientific code, model bytes, workload, Store denominator,
allocator-anomaly contract, or experimental arm from frozen Q. It only replaces
Q's unsafe direct launch command, which could inherit ambient override variables.
Q's exact manifest is `b56f3552...8274` and its STOP hash is `21ffb897...d8df`.

The R wrapper starts from a clean outer `env -i`, clears shell/Python/Git and
launcher overrides, independently verifies exact Q and R manifests plus Q STOP,
and pins every Q launcher input: Python, official repository, Q code/manifest,
J dtype mirror, D allocator mirror and its three provenance companions, model
and both model ledgers, validation data, fresh R run directory, and fresh R
instrumented worktree. The new paths are:

- run: `.../hypic-retained-state-rwd5-20260822r`;
- instrumented repository: `/tmp/HYPIC-98147c0-rwd5-store-r`;
- recovery receipt: `.../hypic-retained-state-rwd5-recovery-r-20260822`.

Before publishing a pass marker, R checks the three exact read-only model asset
modes, verifies no writable top-level model file remains, validates `/` as a
root-owned non-writable safe directory with no import shadows, changes the real
cwd to `/`, rejects both `''` and `'/'` from frozen Python `sys.path`, and proves
the focused test and `sglang` imports resolve exactly to Q and official HYPIC.
Only then does it write the frozen-authority receipt and immediately exec Q.
Probe failure creates no recovery receipt directory or false pass marker.

`asset-mode-repair-receipt.json` correctly retains Job 247512 / Trial 1879097:
that is the actual earlier chmod-only authority for the shared model view, not
the D allocator debug or the R execution identity. It records no model-byte or
scientific-result change.

Local Recovery R regressions before STOP:

- wrapper/publication tests: 7/7;
- malicious ambient values cannot change the parsed final exec environment;
- malicious caller cwd cannot shadow Q tests or official `sglang`;
- bad root stat/import shadows fail before publication;
- all new D allocator authority variables are exact and pinned;
- stale `recovery-P`, `fresh P`, `K_LAUNCHER`, `freeze-k`, and `store-k`
  identities are forbidden in the wrapper;
- Bash syntax and JSON checks pass;
- GPU submissions from R: zero.

R remains STOPPED until its exact manifest receives fresh independent GREEN.
