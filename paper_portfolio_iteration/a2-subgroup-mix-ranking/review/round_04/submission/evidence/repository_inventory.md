# Repository inventory — Phase 1

## Revision-03 correctness provenance status

Round 3 repairs the certificate's missing sampling contract without claiming that the frozen
results met it.  The central theorem is conditional on: FIT-only fixed candidates/design choices;
FIT--CAL independence; conditionally i.i.d. CAL observations within each subgroup; same-point
bounded paired differences; and the stated Bonferroni cell counts.  CP uses exactly `kG` cells at
failure `delta/(kG)`; paired Hoeffding/MPB use exactly `k(k-1)G` ordered-pair/group cells at
failure `delta/[k(k-1)G]` (MPB additionally needs its `n_g >= 2` variance condition).  This is a
theorem contract, not an inference from E03: E03 lacks a split manifest, sampling/collection law,
paired sufficient statistics, and executed UCB implementation.

M3/M3.5 are reclassified throughout as oracle-stratified constructed-mixture diagnostics.  True
class is unavailable before its costly label unless an independent external stratified frame is
provided; E04/E08 do not provide one.  They are not label-acquisition procedures, deployment
allocation policies, or evidence that the CAL contract was realized.

## Revision-02 local provenance status

The baseline and `review/round_01/submission/` are immutable historical objects.  The mutable
`manuscript/` below is the Round-2 revision and must not be described with the historical
initial-source hash.  `remote_snapshot/` is a read-only evidence input, not this revision's
submission package.

### Canonical snapshot-hash procedure

For a reviewer-safe package, include only the enumerated files in a UTF-8, LF-delimited manifest
sorted by POSIX relative path. Hash each file's raw bytes with SHA-256, then hash the UTF-8 bytes
of lines `SHA256<two spaces>relative/path<LF>` in that sorted order. Exclude generated LaTex
auxiliaries, caches, source PDFs rebuilt from the included source, and all review material.
The package must ship its manifest and a one-command verifier that regenerates both file hashes
and the root hash. A successful frozen-result verifier is an artifact-consistency replay, not an
end-to-end scientific rerun. Before external review, include frozen result JSONs, exact verifier
scripts, configurations, split/data identifiers, environment lock, and this scope statement;
otherwise label numerical tables as snapshot-reported diagnostics.

## Historical frozen paper objects

| Object | Audit result | Provenance |
|---|---|---|
| `baseline/paper.tex`, `remote_snapshot/paper/paper.tex` | historical SHA-256 `258def…e2d0c` | local read-only hash |
| `manuscript/paper.tex` | mutable Round-2 revision; hash recorded only in the current build record | local build output |
| Corresponding historical baseline/snapshot PDFs | SHA-256 `fbcff…42c4d`; 14 pages by local `pdfinfo` | local read-only hash/metadata |
| `baseline/explanation.md` | Chinese explanatory readme; not manuscript evidence | local file |
| `remote_snapshot/MANIFEST.md` | r1915 candidate manifest with historical verifier/build assertions | snapshot assertion, not replayed |

## Snapshot contents

The read-only snapshot contains approximately 30 runnable Python source files, 39 result/log
files, 13 paper assets, and 14 project/theory/reproduction documents. It records CPU-based
public-carrier work on digits, Fashion-MNIST, MNIST, and 20News, together with deterministic
and controlled-synthetic allocation diagnostics. Raw data and an executable environment were
not audited in this phase.

## Registered evidence

| Evidence ID | Object | What it can support | Status / boundary |
|---|---|---|---|
| E01 | `paper/paper.tex`, P1/P2 | linear risk bands and absolute regret gate | source text; proof not independently formal-checked |
| E02 | `paper/paper.tex`, P3/Thm. 1 | conditional simultaneous paired-regret certificate | theorem source; requires the explicit Round-3 FIT/CAL contract |
| E03 | `results/SUBGMIX_M25_PAIRED_R1885_5SEED.json` | snapshot-reported matched 5-seed certificate-price frontier | aggregate artifact; not replayed and cannot show its split met the theorem contract |
| E04 | `results/SUBGMIX_M3_BUDGET_R1886.json` and M3 source | oracle-stratified constructed-mixture count diagnostic | executed snapshot artifact; 3-seed, no observable pre-label frame or collection provenance |
| E05 | `results/SUBGMIX_MINIMAX_R1897.json`, `THEORY_MINIMAX_R1895.md` | oracle width-surrogate calculations | executed/snapshot theory; not actual regret-UCB proof, label acquisition, or policy |
| E06 | `results/SUBGMIX_GATE_OC_R1899.json` | controlled synthetic gate operating characteristic | synthetic; mechanism evidence only |
| E07 | `results/SUBGMIX_M10_EXACTBAND_BUDGET_R1911.json` | data-specific exact-relative-gate emptiness / exact absolute-gate diagnostic | executed snapshot artifact; not general impossibility |
| E08 | `results/SUBGMIX_M12_FRESH5_BUDGET_R1915.json` | fresh-seed budget audit | executed snapshot artifact; not replayed |
| E09 | `RESULT_MATRIX.md`, `REPRO_README.md` | experimental protocol and reported metrics | documentation; secondary provenance |
| E10 | portfolio initial screen | current review risk and suggested repair | external local screen; not a full formal review round |
| E12 | `evidence/audit_paired_self_comparator.py` | nonnegative paired-certificate repair and tau>=0 gate invariance | deterministic local CPU audit; no scientific rerun |
| E13 | M3 JSON schema inspection | absence of per-seed allocation rows | provenance boundary only |

## Missing evidence

No registered artifact demonstrates a naturally occurring temporal and geographic subgroup-mix
turnover, tests within-subgroup invariance in that setting, or quantifies operator-level
abstention/fallback costs. No artifact proves that Eq.(mm)'s width objective equals or
bounds the actual candidate/mix regret UCB objective. These absences are deliberate blockers,
not negative results.
