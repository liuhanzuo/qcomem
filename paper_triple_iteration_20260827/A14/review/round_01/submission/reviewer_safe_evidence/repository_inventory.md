# A14 repository inventory

## Located

- The submitted manuscript PDF and a hash manifest for its TeX source,
  appendix, bibliography, and compiled PDF.
- Reviewer-safe claim--evidence, experiment-registry, and method-provenance
  records.
- A partial primary-source citation lock covering the closest scoring work and
  three adjacent sources.

## Not located

- Raw per-item outputs, registered analysis files, and result artifacts named
  by the earlier manuscript.
- Persisted random reveal orders, random-mask draws, sampling-frame IDs,
  prompt bytes, choice-construction code, model/tokenizer revision hashes,
  and instrumented forward-call traces.
- A verified one-command reproducer or known formal launcher for the reported
  tables and inferential quantities.
- The opaque E10--E15 ID-to-file crosswalk.

## Consequence

Numbers can be traced only to manuscript locations and opaque evidence IDs;
they were not recomputed from an accessible experimental bundle.  The paper
defines finite-pool contrasts conditional on the recorded scorer realizations,
makes matched-L the sole primary endpoint, labels every empirical measurement
`source_reported/cannot_verify_from_current_package`, removes all figure
inclusions, and treats raw-evidence verification as blocked pending recovery
of the experimental bundle.
