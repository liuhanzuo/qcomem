# R40 held-out fault method freeze v8

Status: `HOLD_PENDING_FRESH_AUDIT_AND_EXTERNAL_OPERATOR_BINDING`.

This is a method-only, fail-closed freeze. It contains no fault set, formal
configuration, operator binding, snapshot-release record, or GPU result. Nothing in
this directory authorizes a scientific claim or a formal run.

V8 retains the v6/v7 controls and fixes the two blockers in the independent v7
audit. First, `authorized_launcher_v8.py` is the only formal entrypoint and
completes binding verification, all archive/source/snapshot/runner/runtime anchors,
runtime expectation, isolated probe, eight-worker execution, wait, and finalization
in one ordered chain. `Lifecycle.spawn_workers()` accepts no caller specs: its
ordered V8F01--V8F08 argv/env/cwd values come only from an opaque plan derived from
the signed execution contract. Second, every typed worker command is semantically
cross-linked to the runtime expectation: `argv[0]` is the expected Python, cwd is
identical, and CUDA visibility/index resolves to the expected physical UUID. The
exact actual argv/env/cwd and all typed-token rows are hashed as one spawned-spec
commitment. That digest and the binding, contract, expectation, and probe-report
digests appear in the closed-world gate and every terminal.

The 36-test CPU-only suite retains all prior archive, snapshot, manifest, typed
token, lifecycle, signal, cleanup, no-replace, and isolated-probe regressions. It
also reproduces both v7 blockers, proves eight real worker successes, proves real
SIGINT/SIGTERM produce eight failure terminals and exits 130/143, and proves that
missing binding, an unsigned manifest mutation, and a byte-for-byte `/bin/sh`
transplant fail before subprocess creation and leave zero terminals. See
`COUNTEREXAMPLES.md` for single-test commands.

The independent operator must publish a canonical binding signed by the private key
corresponding to `OPERATOR_TRUST_ROOT.json`. The package contains only the public
verification key and accepts no verifier callback, key override, or self-asserted
approval tuple. The snapshot remains sealed; do not give it to a designer until a
fresh independent audit and the external binding both exist.

Declared boundaries are narrow: malicious concurrent same-user mutation and
power-loss/crash atomicity across a multi-file publication are outside this local
method freeze. Normal exceptions are rollback-safe and all formal artifacts are
no-replace. The host kernel and dynamic loader remain part of the fixed trusted
stack; v8 does not claim generic hostile-binary recognition, hostile-runtime
attestation, GPU evidence, or held-out fault evidence. Formal H20 execution remains
blocked after audit until the external binding, snapshot release, independent fault
design, and formal configuration exist.
