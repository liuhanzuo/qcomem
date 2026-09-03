# R40 v8 clean-formal audit

Status: **HOLD; not formal-ready; no launcher; GPU not run**.

Choice B is preserved in `preregistration.json`: a future H20 clean run would
only verify clean reference→real-group→actual-serializer consistency and
lifecycle. LB01--LB04 remain local unit-test mechanism evidence and must never
be described as H20 faults, a production fault campaign, or formal sensitivity.

The implemented subset fixes three independent-audit counterexamples:

- completed request aliasing an incomplete peer;
- forged normalized serializer `storage_id`;
- persistent mutation after the pre-build freeze.

All fifteen local tests pass. V8 connects the lineage interface to a local
`TorchDispatchMode` around the unchanged builder, retains strong source/dest
handles, and matches them to returned live destinations. Wrong-source,
missing, and duplicate clone edges fail. It also retains exact row schema/count/key,
full descriptor/interval and completed-set cross-check. Production PyTorch
2.11/CUDA and the independent shared result remain unverified, with additional
end-to-end artifact/lifecycle gates listed in `formal-blocker.json`. In response,
the incomplete entrypoint, finalizer, and launcher generator were removed.
No staged execution archive is emitted because that would misleadingly suggest
formal readiness.
