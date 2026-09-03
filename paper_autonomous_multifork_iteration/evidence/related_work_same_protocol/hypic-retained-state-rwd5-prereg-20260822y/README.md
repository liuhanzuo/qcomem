# RW-D5 HYPIC retained-state bytes — frozen preregistration Y

Y closes X's final static-builder manifest TOCTOU. It authorizes only Prefix
Cache and official HYPIC `transition_rope_recompute`, two modes over the same
eight frozen workloads (16 affected cells). All other arms remain forbidden.
Y is not a result package and requires fresh independent GREEN before staging
or GPU execution.

## Exact correction

X correctly derived the runtime manifest count instead of hard-coding it, but
independent audit found that the builder still read the manifest pathname three
times: once for the expected SHA, once for the count, and once for `sha256sum
-c`. Atomic replacement between reads could bind the external SHA to the old
bytes while deriving the count and member expectations from replacement bytes.
X was retired before staging or GPU; exact identity and blocker are frozen in
`retired-freeze-x.json`.

Y opens the manifest once with `O_NOFOLLOW`, captures one stable-FD byte stream,
computes the externally expected SHA from those bytes, parses canonical unique
rows and count from those same bytes, and uses those captured row digests for
full member verification. The preregistration records the capture SHA, member
count, and same-capture row-verification result. A deterministic executable
regression performs a real `os.replace()` after capture: the builder continues
to use the original captured authority and never accepts the replacement count
or row set. Duplicate and malformed rows remain fail closed.

Y has 72 manifest members because it adds the X retirement receipt. The exact
manifest count, wrapper expectation, recorded count, and emitted authority
description remain one closure.

## Unchanged platform and science

The sole execution authority remains the unexecuted Job247699/Trial1880085,
with fresh Y freeze/preflight/run/instrumented paths. Platform identity must be
unique in `/proc/1/environ`: Job247699, Trial1880085, and the immutable exact
scope `ROUND27_HYPIC_STORE_FORMAL_W`. Caller ambient values are not authority;
mismatch fails before publication and all 16 cells.

V/T/Q scientific files are byte-identical. Producer, client, replay,
instrumentation, modes, workloads, dtypes, Store denominator, target-entry
ownership, raw allocator-anomaly boundary, cleanup, and terminal resource gates
are unchanged. Store MiB remains only the overlap-aware union of exact
target-entry-owned physical tensor ranges; no global allocator or runtime-safety
claim is made.

Frozen validation before STOP:

- focused tests: 90/90; inherited same-protocol: 10/10;
- real atomic manifest replacement after capture retains original SHA/count/rows;
- same-capture SHA, canonical count, row digests, wrapper count, recorded count,
  and emitted description are closed; duplicate rows reject;
- platform/PID-1, manifest/STOP/FD, asset 14+9, J/D mirror, malicious ambient/cwd,
  lifecycle, process, and eight-GPU zero regressions remain GREEN;
- scientific core equals V; W/X/Y GPU and staging count is zero;
- `main.tex` and tables changed: no.

Only `code/launch_hypic_retained_state_bytes_safe_y.sh` is authorized after
fresh audit GREEN. Direct internal launch is forbidden. Any byte edit after STOP
requires a new freeze and audit.
