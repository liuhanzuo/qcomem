# R40 live-binding v17 fresh-audit HOLD report

Audit date: 2026-08-28 (Asia/Shanghai)  
Decision: **HOLD -- do not execute on H20 and do not cite as a positive result**

This report records the two blocking findings from a fresh audit of the frozen
v17 staging archive.  The findings were independently identified and then
reproduced from the exact local archive by the primary agent.  No QS job, CUDA
initialization, GPU execution, or manuscript edit was performed for this
audit.

## Audited object

- Package directory:
  `paper_autonomous_multifork_iteration/evidence/r40_independent_live_binding_v17_linux_stage`
- Overlay archive:
  `packages/r40-independent-live-binding-v17-linux-stage-20260827b.tar.gz`
- Overlay SHA-256:
  `ea2067b60b38576ce050da2e5fae4fa42da288843ea94aa789830d8d0cd73a43`
- Canonical v6 archive SHA-256:
  `306daba7b79b045a306f0b22d6434143dd568cf1f3b6af7114ad1a4ebe1d6f82`
- V17 source-ledger-file SHA-256, as recorded by the clean-stage receipt:
  `bc97a0a9e8ea61106c7f5edc246845b50066829677ff4ca4537b9c48186d596e`

The clean-stage builder itself succeeded and reported 260 canonical v6
members, 130 retained logical members, 130 exact AppleDouble exclusions, 39
overlay archive members, and a final tree of 138 regular files plus 37
directories with zero AppleDouble paths.

## Blocker 1: the exact clean stage is not self-contained

The exact staged formal preflight was run with the externally supplied
canonical v6 and v17 overlay archive paths, matching the launcher's intended
environment.  It discovered 86 tests: 85 passed and one errored, with zero
skips.  The error was:

```text
test_scientific_payload_is_byte_identical_to_v16 ... ERROR
FileNotFoundError: .../evidence/r40_independent_live_binding_v16_local_integration/preregistration.json
Ran 86 tests
FAILED (errors=1)
```

The cause is explicit in
`tests/test_linux_stage_contract.py:134--141`: the test reconstructs
`ROOT.parent / "r40_independent_live_binding_v16_local_integration"` and reads
the prior sibling package.  That sibling is not a member of the exact clean
stage.  Source-tree success therefore does not imply success of the exact
staged preflight that gates H20 execution.

## Blocker 2: launcher approval and consumption use two pathname reads

`scripts/build_formal_launcher.py:52--56` first evaluates
`sha(a.v6)`, whose implementation reads `a.v6.read_bytes()`, and subsequently
reopens the same pathname with `a.v6.read_text()` for transformation.  The
approved bytes and consumed bytes are therefore not one stable snapshot.  A
pathname-object change between those reads can make the builder consume bytes
other than the bytes whose digest passed the approval check.

The repair must consume a single stable, no-follow file snapshot (or an
equivalent descriptor-bound snapshot), verify its identity and digest, decode
those same bytes strictly, and transform only that verified byte string.

## Passing observations that do not override HOLD

- The exact archive hash, clean/exclusion ledgers, deterministic stage
  construction, and AppleDouble exclusion accounting closed as declared.
- The other 85 discovered clean-stage tests passed before the preflight
  rejected the package.
- The source-tree static report records its declared local checks as passing.

These observations are packaging/mechanism diagnostics only.  The two blockers
occur before the formal GPU gate and therefore forbid an H20 result from v17.

## Required successor gate

A successor version must be a new package rather than a mutation of v17.  It
must (1) validate scientific-payload equivalence using only members present in
the exact stage and (2) bind approval and transformation to one stable byte
snapshot.  It then requires a new independent clean-stage audit before any
authorized H20 execution.
