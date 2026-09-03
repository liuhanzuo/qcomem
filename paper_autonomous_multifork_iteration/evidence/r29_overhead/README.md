# Round 29 local replay overhead

This package records one preregistered, CPU-only practicality measurement for
the frozen RR2 reviewer release.  After one unreported warmup, five complete
replays each verified the package manifest, reconstructed the registered
factorial and timeline evidence, recomputed the numerical sidecars, and ran
the storage-witness tests.  All measured replays exited zero.

The result is intentionally narrow: 850.95 MiB logical package footprint and a
102.71 s median warm-cache replay on the disclosed Apple M4 Pro host.  It is
not a live-capture overhead, cold-download time, service-latency result, or
device-memory measurement.

