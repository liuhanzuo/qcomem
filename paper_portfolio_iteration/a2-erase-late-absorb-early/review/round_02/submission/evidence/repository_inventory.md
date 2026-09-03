# Repository and artifact inventory

Audit date: 2026-08-22. This is a read-only inventory; no script, PDF, source, or remote file was modified.

## Public baseline authority

| ID | path | observed state | provenance assessment |
|---|---|---|---|
| PUB-TEX-001 | `baseline/paper.tex` | 1,010 lines; SHA-256 `d53d4e…825d72` | frozen public source authority |
| PUB-PDF-001 | `baseline/paper.pdf` | 16 pages; SHA-256 `c4e9f3…34c20` | existing artifact, not a fresh build |
| PUB-EXPL-001 | `baseline/explanation.md` | Chinese explanatory document | source-level narrative only; it repeats results but is not a run record |
| PUB-MAN-001 | `baseline/MANIFEST.sha256` | created in this initialization | immutable three-file baseline manifest |

`manuscript/paper.tex` and `manuscript/paper.pdf` have the same hashes as the two public baseline artifacts. No divergence was found between those two local copies.

The public source imports `math_commands.tex`, `ai_use_statement.tex`, `references.bib`, and `iclr2027_conference`; none is present in `baseline/`. It therefore is not a self-contained, reproducibly buildable source package. The baseline cites 14 keys but has no local bibliography file. No public scripts, run outputs, environment lockfile, seed list, or preregistration file is co-located with it.

## Read-only remote candidate (not public evidence)

| ID | path | observed state | provenance assessment |
|---|---|---|---|
| REM-TEX-001 | `remote_snapshot/paper.tex` | 1,013 lines; SHA-256 `e9d9a8…e3b3b` | distinct manuscript version |
| REM-PDF-001 | `remote_snapshot/paper.pdf` | 16 pages; SHA-256 `bd7151…c9e0c` | existing artifact, not rebuilt here |
| REM-MAN-001 | `remote_snapshot/MANIFEST_sha256.txt` | current manifest check: 76 OK, 0 bad | integrity of this copied snapshot only |
| REM-CODE-001 | `remote_snapshot/code/` | 40 Python files and 35 valid JSON output files | remote-only candidate run materials |
| REM-BIB-001 | `remote_snapshot/ref.bib` | 14 entries corresponding to 14 cited keys | candidate bibliography; no full current external re-verification performed |

The remote candidate includes source dependencies, cited bibliography, output JSON, historical audit notes, and build logs. Its current manifest is valid, but its historical notes describe different counts, hashes, and build page counts (for example, 92-entry and 14-page claims while the copied current manifest has 76 entries and current PDF metadata says 16 pages). Those notes must be interpreted as historical records attached to a different freeze state unless their exact relationship is proven.

The remote outputs are syntactically valid JSON. Examples observed without rerun: `pilot19_cifar10n_full_out.json` lists six seeds and R2 false; `eqra_loss_full_out.json` lists six seeds and P1/P2 true; `eqra_loss_p3_full_out.json` lists six seeds and P3 false. These are not registered as verified public-baseline experimental evidence because their source manuscript SHA differs from `PUB-TEX-001`.

## Baseline/remote mismatch that must remain visible

The remote diff materially changes abstract, theorem, proof, figure caption, and operational wording. In particular, the remote version contains stronger schedule/update-count, directional-rate, and deployment wording in portions of the diff, while the public baseline explicitly limits T1 to a fixed quadratic, T2 to full-norm strong-convex return with a directional invariant-subspace precondition, and T3 to an affine scalar recursion. The remote candidate must not be copied into `manuscript/` or cited as evidence for the public baseline without a recorded reconciliation and independent audit.

## Initial-screen context

`01_p5/research_portfolio_36/review_initial_20260822/INITIAL_ICLR_PORTFOLIO_SCREEN_ZH.md` ranked this project second in its triage, with cross-review scores `6/6/4` and meta score `6`. Its decision-driving objection is precisely the absence of an observable deployment enable signal that separates harmful/noisy data from benign-hard/valueful data, plus narrow real positive domain. This inventory uses that finding as a planning input, not as proof.

## Revision 01 derived package

`manuscript/` now contains identified TeX build dependencies and a revised source. The generic
style files are byte-identical to the remote copies; the manager independently established that
`iclr2027_conference.sty` (`797deef4...`) and `.bst` (`2d67552d...`) equal the official ICLR
2027 style zip. This closes a build-dependency gap only. `baseline/`, `remote_snapshot/`, and
`artifacts/checkpoints/pre_revision_01/` remain untouched. The revision withholds historical
empirical prose rather than treating remote code/JSON as public evidence.
