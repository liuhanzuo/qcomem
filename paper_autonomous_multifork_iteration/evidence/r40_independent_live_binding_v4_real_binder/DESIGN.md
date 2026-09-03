# Real binding boundary

The reliable serializer surface is the returned `gdn_phase_witness`, whose
storage witness contains ordered semantic rows with owner/request/layer/family,
content digest, descriptor, and normalized storage identity. This makes direct
selected-row comparison feasible.

Ordering is intentionally precise:

1. hook receives persistent source and freezes a source-distinct reference;
2. unchanged builder performs the real construction and binding;
3. verifier observes the returned real group and enforces materialized-policy
   per-coordinate content plus request/base and peer storage separation;
4. unchanged phase serializer runs;
5. verifier reads its returned rows and compares them to current live tensors;
6. completed requests must differ from retained initial storage, while
   incomplete requests must retain it.

The hook is selected-cell-only, so primary memory cells have zero hook events.
This reduces the selected-coordinate builder/serializer common-mode TCB, but
still trusts hook installation, persistent lifecycle metadata, and the chosen
six-coordinate coverage.

Formal status remains HOLD. No launcher is emitted until all v3 audit gaps are
closed together: exact rank predicates, derived aggregate counts, global
primary zero-event evidence, hash-binding every R40 source, terminal rehash,
Linux `sha256sum`, and read-only or terminal-byte-checked staging.
