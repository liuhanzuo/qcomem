# V31 minimal falsifiable design

- Selected cell: shared-document KV, borrowed immutable GDN base, `N=8`.
- Setup: all 8 requests × 60 coordinates are exact persistent aliases; zero setup clones.
- Post-transition: request 0 has 60 fresh/private coordinates; requests 1–7 retain 420 exact aliases.
- Post-generation: all 480 request coordinates are fresh/private and peer/base-disjoint.
- Rebind provenance: 64 calls × 60 coordinates, with exact pre/post object, storage, descriptor, and content records.
- Serializer binding: actual production 540-row serializer output and published artifact at each of three phases.
- Boundary: selected cell, frozen coordinates/phases, honest same-process Python/PyTorch runtime only.
