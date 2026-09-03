# Assumptions and non-assumptions

## Working assumptions

1. The public baseline is the authority for the current manuscript because
   `baseline/paper.tex` and `manuscript/paper.tex` have the same SHA-256
   (`d53d4e…825d72`).
2. ICLR 2027 is the intended venue. The manager verified the official 2027
   AuthorGuidelines and ReviewerGuidelines: main text is at most 9 pages,
   double-blind review applies, an AI-use statement is mandatory (outside the
   main-text argument), and a reproducibility statement is recommended. The
   local ICLR 2026 rubric remains an internal discrete-score schema only.
3. The portfolio screen is useful triage evidence: it identifies the enable
   signal, harmful-versus-benign discrimination, real matched-path testing,
   and theory scope as decision-driving gaps. It is not a full five-reviewer
   project round and does not establish a submission score.
4. Read-only remote files are authentic bytes for the narrow fact that they
   exist and pass their own *current* manifest check. Their scientific results
   remain candidate evidence until provenance, code/version linkage, and the
   relation to the public baseline are independently reconciled.

## Explicit non-assumptions

- No missing `baseline/` dependency, bibliography entry, script, output, seed,
  configuration, or preregistration is inferred from prose.
- No remote theorem, run, citation verification, build result, or numerical
  outcome is attributed to the public baseline merely because the titles are
  similar.
- A high per-sample loss, annotation disagreement, or gradient conflict is not
  assumed to identify harmful data. It may identify benign-hard/valueful data;
  the planned experiment must test this failure mode.
- The fixed-quadratic, strongly-convex, affine-recursion, logistic-SGD, and
  nonconvex/image regimes are not treated as interchangeable theorem domains.
- Existing PDF artifacts are not treated as fresh successful builds of the
  public source package.
