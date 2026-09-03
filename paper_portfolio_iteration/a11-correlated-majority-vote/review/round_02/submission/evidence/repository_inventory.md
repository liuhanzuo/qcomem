# Repository inventory — Phase 1

Inventory date: 2026-08-22.  All paths below are project-local.  A file's presence does not independently verify its scientific content.

## Round-2 reviewer-safe snapshot boundary

`manuscript/claim_audit.py` and `manuscript/audit_evidence/` are deliberately
relocatable: the audit resolves only paths under its own directory and fails
structurally on a missing prerequisite. Its canonical root hash is SHA-256 of
the UTF-8 concatenation, sorted by POSIX relative path, of
`<raw-file-SHA256><two spaces><relative-path><LF>` lines. The manifest itself
and generated LaTeX auxiliaries are excluded. The local bundle contains the
three result excerpts needed for the checked Figure 1 labels, figure-history
records, and Figure 2 renderer/result hashes. It does emph{not} contain raw
task membership, duplicate-check outputs, original selection timestamps,
predictions, or an environment lock; independent end-to-end replay remains a
residual blocker rather than a claimed capability.

| ID | Artifact | Observed contents | Status / boundary |
|---|---|---|---|
| INV-BASE | `baseline/` | `paper.tex`, `paper.pdf`, explanation | Immutable baseline; manifest recorded. |
| INV-MAN | `manuscript/` | Self-contained revised TeX, copied dependencies, PDF, and current-manuscript audit | Revision output; current hashes and isolated build are recorded in `build/build_record.json`. |
| INV-REMOTE | `remote_snapshot/` | Paper source, style, bibliography, appendices, figures, check script/logs, results | Read-only local snapshot; 113 files observed. |
| INV-THEORY | `T1_mixture_flip_theorem.md`, `L2_L4_posterior_flip_theorems.md`, `L4b_bound_r469.md`, source appendices | Theorem/proof-support documents for mixture/replay and calibration claims | Local support only; no fresh proof verification. |
| INV-RESULT-OMR | `results/fit_cal_test_r469_result.json`, `passrate_r467_result.json`, `shard1_robust_r471_result.json` | Count-based OMR replay FIT/CAL/TEST and shard transfer artifacts | No ordered rollout records exposed in this snapshot. |
| INV-RESULT-OTHER | `results/openr1_m2_pilot_r473.json`, `rlve_n8_r474_result.json` and robustness results | Additional carrier artifacts | Ordered online validity and telemetry are not established by their presence. |
| INV-DRIFT | `drift_stress_r469_result.json`, `margin_repair_r469_result.json`, appendix table source | Pre-fixed synthetic ordered-drift stress and margin analyses | Synthetic stress cannot replace a natural ordered trace. |
| INV-ANALYSIS | `results/*r478*` through `*r508*` | TV, calibration-size, state-cap, geometry, and erratum analyses | Reanalysis artifacts; no new online correctness/cost endpoint. |
| INV-PROVENANCE | `remote_snapshot/claim_check.py`, `claim_check_r508_inplace.log`, `AUDIT_README.md`, `REPRO_README.md` | Frozen-source mechanical assertions and documented upstream paths | Local run: 426 PASS / 0 FAIL / 3 EXT against `remote_snapshot/paper.tex`; it does not audit the revised manuscript. |
| INV-CURRENT-AUDIT | `manuscript/claim_audit.py`, `claim_audit_manifest.json` | Hash-bound current-manuscript semantic audit | Revision 01b checker; it rejects stale TeX/appendix/checker hashes and checks terminal, endpoint, and C04 boundaries. |
| INV-R02-AUDIT | `manuscript/audit_monotonicity_boundary.py`, `manuscript/audit_evidence/` | Exact-rational monotonicity boundary audit and relocatable audit inputs | CPU-only; includes `(N,K,k)=(32,17,29)` and support-zero cases; not a scientific rerun. |
| INV-VISUAL-PROV | `evidence/visual_asset_provenance.json` | Figure 1 generator/edit history, frozen source hashes, current PNG hash, and label cross-check | Generated conceptual layout with evidence-linked labels; it summarizes supplied artifacts and is not independent evidence. |
| INV-FIG2-PROV | `manuscript/fig_tau_conservation.png`, `manuscript/render_fig_tau_sensitivity.py`, `remote_snapshot/results/tv_conservation_r484_result.json` | Figure 2 asset, lightweight renderer, and frozen result hash | Clean 300-dpi redraw from the hash-pinned frozen JSON passes locally; it performs no fitting or scientific recomputation and explicitly remains a fixed-rule sensitivity diagnostic, not population containment. |
| INV-LIT | `references.bib` | 11 BibTeX entries cited by the manuscript | Metadata and sentence-level support not externally verified in this phase. |
| INV-MISSING-ORDER | absent | No file identified with prompt ID, chronological rollout index, raw response, and timestamp for a new formal carrier | Critical missing evidence. |
| INV-MISSING-ENDPOINT | absent | No registered gold correctness result for online stopped output versus full-N output | Critical missing evidence. |
| INV-MISSING-COST | absent | No registered generated-token, latency, cancellation attempt/acknowledgement, or post-cancel completion totals | Critical missing evidence. |

## Read-only audit

- Hashes: all three paper-source copies are `0a09788c045e0c3f12a08e2aba5c799872a476641755ff5ca67bbe53b729acb6`; all PDFs are `540052f27ad0c880f12b93bce68c288b5da94c9e0d0c795a7ea558a84fb39b7c`.
- `python3 remote_snapshot/claim_check.py`: 426 PASS, 0 FAIL, 3 external-provenance checks unavailable at the inherited upstream path.
- A temporary, read-only remote-source build (`pdflatex`, `bibtex`, `pdflatex` twice) produced a 19-page letter PDF; no undefined citation/reference warning was found in the final log.  This is a build check, not visual inspection or venue compliance certification.
