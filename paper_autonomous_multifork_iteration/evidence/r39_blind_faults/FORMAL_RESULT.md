# Round-39 PDF-only blind-fault result (`20260826g`)

The designer saw only the frozen paper PDF and preregistered 11 faults before
the executor was revealed. The formal G campaign retained that fixed set,
selectors, payloads, observers, thresholds, and schedule. Its summary and
605-entry terminal ledger have SHA-256 values
`6f8207ecfb7dd15a964fe1bff3cb42f245eb81c311aad41f01064efd26adeb32`
and
`d93868eeee8a20330fd1de50bf2fbfff11994a79026167bc37397e0ccfb2c197`.

Seven faults produced valid matched reference/clean/mutant pairs: BF01, BF03,
BF04, BF05, BF06, BF08, and BF10. Across the four preregistered comparison
observers (ForkAudit replay, output equality, persistent-base invariant, and
allocation assertions), at least one observer exposed six of those fixed
pairs. BF03 escaped all four observers. BF02, BF09, and BF11 were retained as
pre-execution ineligible under their exact frozen selectors; BF07 was retained
as operational-invalid after its reference lane because the requested BF16
byte view was non-contiguous.

The narrow wording matters: the ForkAudit replay observer itself rejected none
of the seven mutants. The six exposed pairs were found by output or allocation
comparisons. Therefore this campaign is fixed-fault boundary evidence, not a
ForkAudit `6/7` detection claim, recall estimate, accuracy estimate, or
population detection rate.

The locally retained metadata archive is
`formal_h20/r39-blind-faults-20260826g-metadata.tar.gz` (SHA-256
`d8ec32db7de7f4f14637357b2c2cf2fa3b3bccb591b277dd9291b42830a5baee`).
It contains 173 ledger-bound metadata/log files; the omitted full-vocabulary
FP32 sidecars remain bound by the complete remote terminal ledger. The remote
audit verified all 605 ledger entries, including 352 FP32 sidecars of 993,280
bytes each.
