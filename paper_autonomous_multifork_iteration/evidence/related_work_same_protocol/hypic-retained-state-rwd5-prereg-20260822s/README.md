# RW-D5 HYPIC retained-state bytes — frozen preregistration S

S authorizes only the affected Prefix Cache and official HYPIC
`transition_rope_recompute` Store-byte measurement: 2 modes × the same 8
frozen Qasper/2WikiMQA workloads = 16 cells. Full Recompute, CoMem, RR2, GDN,
serving controls, and every other method are forbidden. This is a stopped
preregistration, not a result package; GPU execution requires a fresh
independent GREEN review of this exact manifest. The only authorized external
entry point is the frozen `code/launch_hypic_retained_state_bytes_safe_s.sh`;
the internal eight-GPU launcher must not be invoked directly.

## Why P was invalid and what S preserves scientifically

Recovery P completed Prefix 8/8, but all eight HYPIC schedulers failed before
any HYPIC raw or Store receipt because the pre-snapshot Mamba free tensor
contained a duplicate entry. The entire Prefix-only partial output is excluded;
P supports no storage comparison and is not a scientific negative.

The audited D one-GPU debug-only observation on Job 247574 / Trial 1879456
captured the HYPIC post-prime allocator without mutation or formal receipt:
capacity 183, raw count 182, unique count 181, and exactly one duplicate excess
(slot 3 at raw positions 168 and 177). The unique missing physical domain is
exactly slots 14 and 15; they are the two distinct lock-free target segment
entries and are absent from the raw free list. The immutable live mirror is
included under `live-allocator-debug-d-trial-1879456/`; its local manifest is
`d57e3e5436f9b7b586a3788a1f3205d9ee3e4f6403496edc3205c2927a842f7e`
and its original remote artifact ledger is `112cc04d...`.

The frozen D launch plan itself records Trial 1879456 and the exact run
directory, but not Job 247574. `allocator-debug-d-provenance.json` therefore
states honestly that the job ID comes from the external execution/submission
receipt and post-run platform query supplied by the run coordinator. It binds
the frozen D manifest/plan, live mirror, remote ledger, and raw/validation/
terminal receipt hashes without pretending that the launch plan contains the
job ID.

Static inspection finds that `MambaSlotAllocator.free()` and the unused group
return path concatenate indices without duplicate protection, while PICache has
several free call sites. The post-prime snapshot cannot uniquely identify which
call sequence duplicated slot 3. S records that limitation and makes no exact
root-cause, global allocator-correctness, or runtime-safety claim. It preserves
Q's physical-byte denominator, two modes, and 16-cell scientific contract.

## Selected physical-byte contract

S never deduplicates or repairs the live allocator. It records and blindly
replays the raw free-slot order, raw/unique counts, duplicate slot/count/
positions, unique free domain, and derived unique allocated domain.

- Prefix remains fail-closed on any duplicate pre-free Mamba entry.
- HYPIC requires the unique allocated physical domain to equal exactly the two
  selected target-entry slots; the cache must contain exactly those two target
  segments with lock refs zero.
- Store MiB is only the overlap-aware union of backing tensor byte ranges
  selected by those exact target entries. Allocator multiplicity, metadata,
  pool capacity, NVML, and process allocation are excluded.
- Terminal verification requires both target entries to disappear, both
  selected slots to enter the unique full physical domain, and the original
  duplicate fingerprint to remain unchanged—no migration or growth.
- Any changed terminal anomaly fails the cell closed and cannot be reported as
  a Store result.

Component dtype authority from K remains unchanged: K/V, convolution, and
convolution tails are BF16/2B; temporal and transition are FP32/4B. The exact J
debug-only mirror remains frozen and independently validated.

## Integrity and lifecycle gates

Producer ranges are derived from full tensor metadata and exact selected slots;
blind replay distrusts producer ranges and totals and rederives dtype, element
size, shape, stride, pointer-relative identity, storage bounds, layer×slot
coverage, and overlap-aware unions. Scheduler lineage, exact external cell,
readiness, overlay, code/model/data/environment authority, lock refs, component
topology, and terminal cleanup remain closed. Cleanup is TERM → bounded poll →
KILL survivors → final reap → explicit PID and process-group absence after each
mode. Before the artifact ledger and again immediately before `COMPLETED`, all
eight frozen GPU UUIDs must report zero compute applications and zero MiB used,
and no matching SGLang server/scheduler or formal client may remain.

The safe wrapper validates the exact S manifest and STOP identities, fixes `/`
only after proving it root-owned and non-writable with no import shadows,
verifies exact frozen origins for both unittest modules and official SGLang,
pins all launcher inputs under `env -i`, and uses fresh S output paths. The
internal launcher separately requires the wrapper marker and runs both frozen
test suites before any GPU server or preregistered output.

Frozen S validation before STOP:

- focused RW-D5 tests: 81/81;
- inherited same-protocol tests: 10/10; combined 91/91;
- exact J dtype mirror and D allocator mirror validation pass;
- duplicate/missing/selected-slot/terminal migration-growth tamper regressions
  fail closed;
- Python compilation, Bash syntax, official commit/clean checks pass;
- stubborn-process-group, compute-process, nonzero-memory, matching-process,
  inherited-test-failure, malicious-ambient, and malicious-cwd regressions fail
  closed;
- GPU submissions from S before STOP: zero;
- `main.tex` and paper tables changed by this workstream: no.

If any listed byte changes, retire S and refreeze. Never use P or either
debug-only run as a paper Store number.
