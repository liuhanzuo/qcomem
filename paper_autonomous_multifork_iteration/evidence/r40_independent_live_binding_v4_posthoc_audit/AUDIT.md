# R40 v4 independent posthoc audit

## Status and scope

This is an **external, posthoc, read-only CPU audit** of the separately frozen
`r40_independent_live_binding_v4_real_binder` package. It is not part of v4,
does not alter v4, and is not an authorized formal run.

- Overall v4 formal status: **HOLD**.
- Bundled v4 mechanical gates: PASS (6/6 unit tests, 9/9 static checks, 10/10
  source-ledger rows) when independently replayed before this package was
  frozen.
- Strong independent storage/lifecycle claim: **FAIL** under three concrete
  counterexamples.
- GPU execution: not run.
- QuickSilver invocation: not run.
- Positive paper evidence: **false**. This package records limitations and
  cannot be cited as a successful H20 experiment or as support for a positive
  paper claim.

The external binding in `V4_SOURCE_FILES.sha256` fixes the exact ten v4 files
audited. It also verifies v4's `source-code.sha256` at SHA-256
`b79bbc84821ecacf09d939faadd2818ec5674df88601b97ca0bb3ffd1db4616d`.

## Audit verdict

The v4 hook is a real improvement over v1--v3: it instantiates
`ActualBindingVerifier` from the live persistent cache before calling
`original_build`, and its local fault tests mutate the group consumed by the
verifier rather than an off-path candidate dictionary. The phase hook also
waits for `original_phase` and consumes its returned GDN object.

Those facts support only a narrow result: selected-coordinate content,
shape, and dtype are compared against a pre-build persistent reference and
against current live CPU objects in the supplied synthetic tests. They do not
establish independent verification of normalized storage identity or retained
handle lifecycle.

The key implementation boundary is
`executed_source/r40_real_binding.py:82-106` in v4. Rows are indexed and checked
for content/shape/dtype, but their `storage_id` is never independently related
to the current tensor's storage. The only serialized storage assertion is that
a request row ID differs from its persistent row ID. Initial request state in
v4 lines 59 and 75--76 is only a `(device, data_ptr, nbytes)` tuple, not a
retained tensor/storage handle. Peer aliasing is checked only immediately after
build in lines 78--80; the frozen persistent reference is not rechecked during
phases.

Accordingly, the broad label “local mechanism PASS” is too strong. A precise
label is: **the six bundled tests pass, while formal integration and the strong
storage/lifecycle acceptance claim remain HOLD/FAIL respectively**.

## Independently replayed counterexamples

`probes/probe_v4_counterexamples.py` reimplements the synthetic fixture without
importing v4's tests. Before importing the verifier, it checks all ten external
v4 hashes and the v4 source-ledger hash. It creates CPU BF16 tensors only and
contains no file-write path.

Each counterexample represents a condition an independent storage/lifecycle
verifier should reject. The frozen v4 verifier returned normally, so the probe
records `ACCEPTED`.

1. `peer_alias_after_transition=ACCEPTED`: after the clean group passes build
   verification, all selected request-0 tensors are rebound to the matching
   live tensors of incomplete request 1. The truthful phase serialization is
   accepted because completed requests need only differ from their old address
   tuple; phase-time peer separation is not rechecked.
2. `forged_storage_id=ACCEPTED`: a selected request row's normalized-looking
   storage ID is replaced with `storage-9999` while its live tensor and content
   remain unchanged. The row is accepted because row storage identity is not
   connected to live storage identity.
3. `persistent_mutation_after_freeze=ACCEPTED`: the persistent recurrent source
   is modified after pre-build freeze. A fresh phase snapshot of the modified
   object is accepted because phase verification compares the row to current
   persistent content, not to the frozen source reference.

The exact text and JSON results are frozen in `raw/probe.stdout.txt` and
`raw/probe.results.json`. Three independent unit assertions in
`tests/test_counterexamples.py` require these outcomes to remain `ACCEPTED`;
a future repaired verifier will intentionally make those tests fail, signalling
that this posthoc artifact no longer describes that new source.

## Finding-by-finding assessment

### Pre-build source independence: partial PASS

The reference is created before `original_build` from the builder's `cache`
argument plus preregistered coordinates. It is not derived from the returned
group, manifest, candidate rows, or phase serializer.

However, v4 does not freeze or independently verify `plan`, `policy`,
`gdn_base_policy`, resident count, or returned group policy/audit metadata.
Thus “source plus materialized policy is frozen” is not yet true. Source
descriptors are limited to shape and dtype; stride, storage offset, tensor byte
range, and storage bytes are omitted.

### Real group fault path: narrow PASS

The four bundled fault tests mutate the actual synthetic group object observed
by `verify_built_group`; there is no `_candidate_items` detector. They do not,
however, call the actual Qwen resident-group builder or exercise
`install_real_binding_hook` end to end. The same-geometry test relies on
synthetic per-layer contents being distinct and supplies no gate for equal-bit
real source coordinates.

### Returned phase rows: content PASS, storage FAIL

The hook consumes `result[1]` after `original_phase` returns. It validates row
content, shape, and dtype against current live objects. It does not validate
the row's storage identity, complete descriptor, uniqueness, or exact schema
against live objects. Duplicate row keys are silently collapsed by a dict. The
hook also ignores `result[0]`, so the checked in-memory GDN is not independently
bound to the phase artifact bytes already written by the original function.

### Lifecycle: FAIL for the strong claim

The final hook checks the expected three phase names, and the hard-coded
completed request sets happen to match the current runner. The verifier does
not derive or cross-check them from the returned GDN; it does not retain initial
handles; it does not recheck peer/base/within-request relationships after
transition; and it does not recheck persistent immutability. CUDA allocator
address reuse would additionally make raw address tuples insufficient even
when no adversarial alias is introduced.

### Local gates and source binding: mechanical PASS

Before freeze, the auditor independently observed:

- v4 unit tests: 6/6 PASS;
- v4 static audit: 9/9 PASS and exact match to its stored JSON;
- v4 source ledger: 10/10 PASS;
- no v4 formal launcher;
- no GPU or QS execution.

The v4 static checks are primarily source-string existence/order checks. In
particular, primary-memory zero events are a hard-coded payload field rather
than an observed global counter. They are useful preparation checks, not formal
evidence.

## Formal blockers

Do not launch the current package on H20. The following must first be repaired,
frozen in a new non-overwrite package, and independently re-audited:

1. Retain initial tensor/storage handles through the final phase rather than
   retaining allocator address tuples only.
2. Independently reconstruct the live storage equality/overlap partition and
   compare it with normalized row storage IDs, device, stride, storage offset,
   tensor/storage byte counts, and byte intervals.
3. Recheck frozen persistent identity/content and all request-to-base,
   request-to-peer, and within-request ownership relationships at every phase.
4. Cross-check exact phase and completed-request sets from the original call,
   returned GDN, resident group, and preregistration. Validate N=8, selected
   policy, plan, cell/run/capture/guard identifiers, exact unique row schemas,
   and exact row counts.
5. Bind the verified in-memory GDN to `result[0]` and re-read phase-artifact
   bytes; prevent failed checks from leaving admissible orphan artifacts.
6. Add hook-level end-to-end tests that inject these faults through the wrapped
   builder/phase path, including post-generation, duplicate/missing rows,
   policy/N drift, restore behavior, and exact build/registration count.
7. Gate selected coordinates for semantic discriminability when same-geometry
   source values are bitwise equal, or explicitly narrow the provenance claim.
8. Audit CUDA BF16/non-contiguous hashing, allocator reuse, asynchronous
   execution, and Transformers 5.14.1 hook behavior; ensure verifier hashing
   does not perturb the ownership observation it is intended to validate.
9. Build a non-overwrite, explicitly authorized one-shot formal launcher while
   preserving the pinned runner/argv/model/PG19/result roots. Enforce exact
   world/rank 0--7 predicates and no post-outcome retry.
10. Add an independent finalizer that derives exact per-rank and aggregate
    counts, verifies one artifact per rank/no extras, and measures global
    primary-memory hook events instead of accepting a hard-coded zero.
11. Hash-bind every new source and dependency, use Linux `sha256sum` in formal
    staging, enforce read-only or terminal-byte-checked staging, and perform a
    terminal source rehash tied to the launch receipt.

Only after those gates pass should a fresh authorized 8-H20 execution be
considered. Even then, the claim remains limited to the preregistered six
coordinates in the single fixed N=8 materialized-policy witness cell unless
coverage is explicitly expanded.

## Replay

From this audit directory:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 probes/probe_v4_counterexamples.py --format text
PYTHONDONTWRITEBYTECODE=1 python3 probes/probe_v4_counterexamples.py --format json
bash -n scripts/replay.sh
bash scripts/replay.sh
```

`scripts/replay.sh` first checks the external v4 binding and the frozen audit
payload, reruns the tests, then diffs freshly generated text and JSON against
the frozen raw outputs. It sets `PYTHONDONTWRITEBYTECODE=1` and fails if any
`__pycache__` directory appears.

## Chain of custody

- `V4_SOURCE_FILES.sha256`: external binding of the ten audited v4 files.
- `MACHINE_ENVIRONMENT.json`: non-secret local CPU/software environment.
- `probes/probe_v4_counterexamples.py`: read-only independent probe source.
- `tests/test_counterexamples.py`: three required regression assertions.
- `raw/`: original test stdout and deterministic probe text/JSON.
- `SHA256SUMS`: terminal hash ledger over all primary audit payload files.
- `TERMINAL_RECEIPT.json`: terminal verdict and hashes of both ledgers and raw
  results. It is intentionally outside `SHA256SUMS`; its own SHA-256 is the
  externally reported terminal receipt identifier.

This audit must remain separate from positive experimental evidence and from
the paper/registry until a repaired package has passed a new independent audit.
