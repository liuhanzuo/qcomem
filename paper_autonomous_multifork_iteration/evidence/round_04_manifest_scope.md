# Round-4 manifest scopes

The author-side complete package has 628 entries and 892,144,066 bytes.  Its
manifest SHA-256 is
`51346e18c2d2685ea57712d1823e6056ea6bea11a5718da6d24f2fe1d1b65338`,
with parent
`d6a9b71ee078c6d21c90c64ad23d9c4f624e381d262faf36ed812104e8e59633`.
It binds the replay inputs, deterministic projections, validation reports,
generated tables, and storage-witness specification.

The blind-snapshot builder creates a separate reviewer derivative: it omits
the hidden planning preimage and removes response-plan fields from governance
receipts without modifying scientific raw artifacts.  Because that derivative
is rebuilt for each frozen submission, its exact manifest SHA, parent, file
count, and byte count are written into the snapshot-local copy of this note and
`integrated_results.json`; reviewers should use those snapshot-local values.
