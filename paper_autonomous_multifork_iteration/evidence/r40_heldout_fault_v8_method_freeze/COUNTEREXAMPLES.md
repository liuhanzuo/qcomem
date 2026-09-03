# V8 blocker counterexamples

These are local CPU-only security regressions. They are not fault cases and do
not release the sealed designer snapshot.

1. **Caller-controlled workers (v7 blocker).** Run
   `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest -v test_v8.V8Tests.test_31_counterexample_caller_worker_specs_has_no_lifecycle_entry`.
   The test proves that `Lifecycle.spawn_workers` accepts only `self`, that an
   `AuthorizedPlan` cannot be caller-constructed or mutated through its public
   interface, and that only `authorized_launcher_v8.py` bears the formal-launcher
   marker.
2. **Signed syntax with semantic drift (v7 blocker).** Run
   `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest -v test_v8.V8Tests.test_32_counterexample_signed_semantic_drift_rejected_before_any_spawn`.
   Two independently signed mutants preserve typed-token syntax but replace
   either worker `argv[0]` or one worker's CUDA visibility. Both are rejected
   before `Popen` because the worker contract no longer agrees with the runtime
   expectation.

The complete 36-test suite also retains the v6/v7 archive, trust-root, snapshot,
manifest, typed-token, lifecycle, signal, cleanup, no-replace, isolated-probe,
and transplanted-shell regressions. In particular, missing binding, mutation of
the signed runner manifest, and a byte-for-byte `/bin/sh` transplant each leave
the terminal directory empty and invoke no subprocess.
