# R40 v7 clean-formal audit

Status: **HOLD; not formal-ready; no launcher; GPU not run**.

Choice B is preserved in `preregistration.json`: a future H20 clean run would
only verify clean reference→real-group→actual-serializer consistency and
lifecycle. LB01--LB04 remain local unit-test mechanism evidence and must never
be described as H20 faults, a production fault campaign, or formal sensitivity.

The implemented subset fixes three independent-audit counterexamples:

- completed request aliasing an incomplete peer;
- forged normalized serializer `storage_id`;
- persistent mutation after the pre-build freeze.

All twelve local tests pass. V7 additionally closes exact row schema/count/key,
full descriptor/interval, completed-set cross-check, and a fail-closed lineage
receipt interface. The complete audit nevertheless found a fundamental
same-geometry/same-content coordinate-discriminability blocker and additional
end-to-end artifact/lifecycle gates listed in `formal-blocker.json`. In response,
the incomplete entrypoint, finalizer, and launcher generator were removed.
No staged execution archive is emitted because that would misleadingly suggest
formal readiness.
