# V33 minimal falsifiable design

- Operations-only change from V32: make a byte copy of the approved canonical
  V6 archive inside the fresh scratch directory, then require regular-file
  type, `nlink == 1`, and the unchanged approved SHA-256 before staging.
- The canonical archive gate remains strict; no link-count check is weakened.
- Fresh stage, scratch, and result paths prevent reuse of the failed V32
  attempt.
- Scientific payload and configuration are byte-identical to V32 except for
  run/package identity and fresh-path bindings required by non-overwrite.

- Controlled change from V31: terminal rank-artifact `primary_memory_calls_observed` must be the strict integer `8`; reject `7`, `9`, and booleans.
- Unchanged separate end-of-run global-absence contract: 12 primary memory calls and zero hook events per rank.

- Selected cell: shared-document KV, borrowed immutable GDN base, `N=8`.
- Setup: all 8 requests × 60 coordinates are exact persistent aliases; zero setup clones.
- Post-transition: request 0 has 60 fresh/private coordinates; requests 1–7 retain 420 exact aliases.
- Post-generation: all 480 request coordinates are fresh/private and peer/base-disjoint.
- Rebind provenance: 64 calls × 60 coordinates, with exact pre/post object, storage, descriptor, and content records.
- Serializer binding: actual production 540-row serializer output and published artifact at each of three phases.
- Boundary: selected cell, frozen coordinates/phases, honest same-process Python/PyTorch runtime only.
