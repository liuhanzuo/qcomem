# Change verification — r45 evidence revision (evidence and accounting issues)

Verifier role: change verification only. Not a review of the paper.
Date: 2026-09-03
OLD: `main_r44_structure.tex` (build `build/r44_structure_v1/main_r44_structure.pdf`)
NEW: `main_r45_evidence.tex` (build `build/r45_evidence_v1/main_r45_evidence.pdf`, 36 pp., built 02:17;
tables `tables/qcomem_validation60_r42.tex` and `tables/qcomem_tradeoff_r42.tex` rewritten 02:13, so the
built PDF reflects them)

Method: `diff` on the two `.tex` files; `grep` over the manuscript, the `tables/` inputs and the figure
PDFs; `pdftotext -layout` on both builds for page placement; and independent recomputation from the
48 archived shards in `evidence/qcomem_mixed_validation_60item_20260812d/artifacts/raw/`. No file in the
repository was modified other than this report.

| Issue | Verdict |
|---|---|
| R44-4-01 — mixed reference arms in the headline trade-off | **resolved** |
| R44-5-02 — no method provenance for the named contribution | **not_resolved** |
| T-02 — Store accounting above the Eq. 3 ceiling | **partially_resolved** |
| A11 finding C4 — measured Read path is private materialization | **resolved** |

---

## 1. R44-4-01 (critical) — one reference arm for the headline trade-off

### Verdict: `resolved`

### OLD

`main_r44_structure.tex` Table 1 (via `tables/qcomem_validation60_r42.tex` as it then stood; rendered
r44 PDF p.6) had column header `∆F1 vs. split Q16 [95% CI]`, row label `Full-prefix Q16/BF16`, and the
footnote:

> Compression is relative to full-prefix Q16/BF16. F1 and its differences use the 0–100 scale; intervals
> are paired 10,000-resample bootstrap intervals against split Q16. ... Its catastrophic-regression count
> is 0/60 under the registered per-item ∆F1-versus-dense ≤ −50 point rule

Abstract (l.55–57): "reduces mean retained tensor payload from $136.235$ to $9.661$ MiB/document
($14.10\times$ smaller). Its mean F1 (0--100) is $54.24$ versus $54.62$ for split Q16, a paired difference
of $-0.39$ points with a 95\% bootstrap interval of $[-2.04,1.08]$".

Conclusion (l.622–627): "frozen Q4/Q4/Q8 is $14.10\times$ smaller than full-prefix Q16. ... an observed
mean-F1 difference from split Q16 of $-0.39$ points with 95\% interval $[-2.04,1.08]$".

Three arms in one sentence pair, exactly as diagnosed.

### NEW

`tables/qcomem_validation60_r42.tex` — column header is now `$\Delta$F1 vs. full-prefix [95\% CI]`, the
reference row is `Full-prefix, native dtype`, the frozen row reads
`9.661 & 14.10$\times$ & & 54.24 & & $-0.45$ [$-2.06$, $0.99$]`, and the footnote reads:

> Every column, compression included, uses one reference arm: the full-prefix entry as the unmodified
> stack retains it, in its native dtypes. ... Intervals are paired 10,000-resample item-level bootstrap
> percentile intervals against full-prefix (seed 20260902); deltas use unrounded means. The
> catastrophic-regression count is 0/60 against dense under the registered $\leq-50$ point rule and 0/60
> against full-prefix.

Abstract (l.62–66): "Against the same full-prefix reference, its mean F1 (0--100) is $54.24$ versus
$54.68$, a paired difference of $-0.45$ points with a 95\% bootstrap interval of $[-2.06,0.99]$ (10,000
paired item-level resamples, seed 20260902), the sign consistent across both datasets ($-0.34$ Qasper,
$-0.56$ 2WikiMQA)."

Sec. 5.4 (l.513–514): "Every quality difference here uses one reference arm, full-prefix Q16, the arm the
Store column and the compression ratios also use."

Sec. 5.1 denominators (l.457–458): "Store, compression, and every quality difference below use one
reference arm, full-prefix Q16."

Conclusion (l.659–666): "$14.10\times$ smaller than the full-prefix reference the stack actually retains
... a mean-F1 difference from full-prefix Q16 of $-0.45$ points, 95\% interval $[-2.06,0.99]$".

New Appendix B, `Complete Interval Inventory and the Reference-Arm Change` (l.807–930), lists all 42
intervals with the reference arm each one used, and states: "$-0.0585$ for splitting plus $-0.3870$ for
quantizing the split state equals the $-0.4455$ reported against full-prefix, so the previously published
$-0.39$ understated the deployment-relevant loss by about $13\%$ of its value."

### Reasoning

All three assigned checks pass.

**(a) One reference arm.** Compression column, ∆F1 column and the reference row now name full-prefix
native dtype. The catastrophic-regression rule is the one place a third arm legitimately remains, and the
required action permitted that provided it is stated: the footnote gives both counts (0/60 vs dense under
the registered rule, 0/60 vs full-prefix), and Sec. 5.4 repeats "0/60 against dense and against
full-prefix". So the third arm is named rather than hidden.

**(b) The −0.45 figure and interval.** Present in the abstract (p.1), contribution bullet 2 (p.2), Table 1
(p.7), Sec. 5.4 (p.7), the Conclusion (p.9), and the Appendix B inventory. I recomputed it independently
from the 360 archived item-level F1 values:

```
prefix mean 54.682251   frozen mean 54.236786
frozen − prefix (paired, 60/60) = −0.445465  →  −0.45
qasper −0.3354  2wikimqa −0.5556
split − prefix = −0.058480 ;  frozen − split = −0.386986
(−0.058480) + (−0.386986) = −0.445465  ✓ the compounding claim
catastrophic ≤−50: 0/60 vs dense, 0/60 vs full-prefix  ✓
```

All 15 new point estimates in Appendix B's first block reproduce to four decimals from the shards. The
ledger's expected "−0.44" came from differencing the *rounded* means 54.24 − 54.68; the unrounded paired
difference is −0.4455, so −0.45 is the correct rounding and the table footnote's "deltas use unrounded
means" explains the gap. Not an error.

**(c) No residual split-Q16 headline.** `grep` finds no sentence in the manuscript pairing "14.10" with
"−0.39". In the main text (pp.1–9) "split Q16" now survives in exactly two places: Sec. 5.2 as the
intermediate Store step (136.235 → 34.683), and the Table 1 row label, whose ∆F1 is against full-prefix.
The split-relative delta appears only as the ablation decomposition Sec. 5.4 explicitly frames as such
("splitting costs $-0.06$ points and quantizing the split state a further $-0.39$") and in the Appendix B
inventory with its arm labelled. This is precisely the "keep the split-Q16 comparison as an ablation row"
option the required action offered.

### Residual item (does not change the verdict, belongs to the artifact-scope issues)

`evidence/claim_evidence_map.tsv` (mtime 2026-09-02 12:40, i.e. untouched by this revision) still carries
the pre-revision framing for `C-QCOMEM-60-STORE-F1-01`:

> ... (14.1018x smaller), with mean F1 0.542368 versus dense 0.542888 and Q16 replay 0.546238; its
> Q16-relative paired mean delta was -0.003870 with 95% bootstrap interval [-0.020365, 0.010756]

The manuscript's Appendix I points readers at this file. The paper's headline is now full-prefix-relative
and dual-ratio; the shipped claim map is still split-Q16-relative and single-ratio. The manuscript-side
fix is complete, but the artifact now disagrees with it. This is R44-5-04 / R44-4-15 territory rather than
R44-4-01, so I have not downgraded here — flagging it so it is not lost.

---

## 2. R44-5-02 (critical) — method provenance for the named contribution

### Verdict: `not_resolved`

The 25 rows were **not merged**. They sit only in the revision draft.

### Evidence

```
evidence/method_provenance.tsv                      84 lines  mtime 2026-09-02 02:11
revision/a11_method_provenance_rows_20260902.tsv    26 lines  mtime 2026-09-02 15:54
```

84 lines = 1 header + 83 data rows: byte-for-byte the same 83 rows the reviewer counted, written
*before* the draft was authored. The reviewer's keyword census over
`evidence/method_provenance.tsv` still returns the same result it did at r44:

```
quantiz   0        dequant   0        residual  0
bootstrap 0        LongBench 0        60-item   0
```

The draft's 25 `method_id`s (`M-QCOMEM-EQ3-QUANTIZER`, `M-QCOMEM-EQ3-DEQUANTIZER`,
`M-QCOMEM-PACK-CODEC`, `M-QCOMEM-GROUP-64`, `M-QCOMEM-EQ2-WRITE-ENUM`, `M-QCOMEM-WRITE-PATH`,
`M-QCOMEM-READ-FORK`, `M-QCOMEM-READ-LOWER-REPLAY`, `M-QCOMEM-READ-SUFFIX`, `M-QCOMEM-READ-GREEDY`,
`M-QCOMEM-SPLIT-DEPTH`, `M-QCOMEM-TRUNCATION`, `M-QCOMEM-POLICY-FROZEN`, `M-QCOMEM-POLICY-SAMEMEM`,
`M-QCOMEM-POLICY-AGGRESSIVE`, `M-QCOMEM-POLICY-CALIBRATION`, `M-QCOMEM-STORE-ACCOUNTANT`,
`M-QCOMEM-STORE-Q16-DTYPE-DEFECT`, `M-QCOMEM-STORE-COMPONENTS`, `M-QCOMEM-BASELINE-FULLPREFIX`,
`M-QCOMEM-DEPLOY-STORE`, `M-QCOMEM-F1`, `M-QCOMEM-BOOTSTRAP`, `M-QCOMEM-PANEL-RUNNER`,
`M-QCOMEM-PANEL-REPLAY`) appear **nowhere else in the repository**:

```
grep -rn "M-QCOMEM-EQ3-QUANTIZER" . --exclude-dir=.git
  → revision/a11_method_provenance_rows_20260902.tsv   (only hit)
```

They are not in `evidence/method_provenance.tsv`, not in `supplement_anonymous/provenance/`, and not in
any staged submission tree. The draft's header matches the target schema exactly (`method_id`,
`method_statement`, `source_path`, `symbol_or_lines`, `configuration_or_runtime`,
`manuscript_locations`), so the merge is mechanical and was simply never performed.

The second half of the required action — "Ship the corresponding files in the artifact" — is also unmet.
`supplement_anonymous/code/` (mtime 2026-08-17, untouched) contains `qcomem_torch.py` and
`run_downstream.py` but none of `run_replay_diagnostic.py`, `analyze_validation.py`,
`aggregate_replay.py`, `aggregate_layer_sensitivity.py`, `qcomem_deployment.py`, `qcomem_paged.py`, or
`launch_mixed_validation_8gpu.sh`.

### Partial credit (does not close the issue)

The revision *does* disclose the gap honestly in the manuscript, which r44 did not. New text in
Appendix A (l.717–723, PDF p.10):

> The package archives the quantizer/split-replay module and the prompt-and-scoring driver
> byte-identically at four paths, but not the run driver, the aggregation and bootstrap scripts, the
> layer-policy search, the deployment accountant, or the launch script; an artifact reviewer can
> therefore read the code those files call but not the files that produced the table, and they must be
> released before Table~\ref{tab:qcomem-validation60} is independently re-executable.

That sentence has no counterpart in `main_r44_structure.tex` (grep: zero hits). It converts an
undisclosed provenance hole into a disclosed one — worth noting — but the issue as written is about the
artifact having zero rows covering the paper's own contribution, and that is unchanged. The manuscript
never claims the rows were added, so no false statement was introduced.

### What remains

1. Append the 25 rows of `revision/a11_method_provenance_rows_20260902.tsv` to
   `evidence/method_provenance.tsv` (schemas already match; result should be 108 data rows).
2. Re-run the reviewer's keyword census on the merged file and confirm non-zero counts for
   `quantiz`, `dequant`, `residual`, `bootstrap`, `LongBench`, `60-item`.
3. Ship the seven named source files in the artifact, or keep the Appendix A disclosure and accept that
   Table 1 is not independently re-executable.
4. Run the reviewer's stated check on the shipped quantizer: group size 64, `s = (u−m)/(2^b−1)` with
   clipped rounding to `[0, 2^b−1]`, BF16 scale/bias; and confirm the named accounting function returns
   9.660873 MiB (frozen) and 136.235352 MiB (full prefix).

---

## 3. T-02 — Store accounting versus the Eq. 3 compression ceiling

### Verdict: `partially_resolved`

All four checks I was assigned pass, and every number reproduces from the archive. I am withholding
`resolved` because the meta-review's one mandatory item — publishing the per-row component byte
breakdown — is not in the manuscript, and the printed Table 1 column still yields a ratio above the Eq. 3
ceiling unless the reader back-derives a value the paper never prints.

### OLD

r44 Table 1 row label `Full-prefix Q16/BF16`, footnote "Compression is relative to full-prefix
Q16/BF16"; Sec. 4.2 (l.279–281) "Q16 is the BF16 unpacked reference"; Figure 2 caption (l.296) "Q16
denotes the unpacked BF16 reference"; Conclusion "frozen Q4/Q4/Q8 is $14.10\times$ smaller than
full-prefix Q16". No mention anywhere of FP32 GDN recurrent state.

### NEW — the four assigned checks

**(i) Disclosure in the abstract AND Sec. 5.2 AND Table 1.** Confirmed in the built PDF at all three
sites.

- Abstract, PDF p.1 (l.57–62): "$14.10\times$, 60-item mean). That reference is what the unmodified stack
  retains, in its native dtypes; $30.000$ MiB/document of it is FP32 GDN recurrent state in excess of a
  BF16 encoding, so part of the gain is dtype narrowing rather than packing, and against an all-BF16
  reference of $106.235$ MiB/document the policy is $10.9965\times$ smaller."
- Sec. 5.2, PDF p.7 (l.473–481): "$14.10\times$ smaller than the full-prefix reference, or a $92.91\%$
  reduction. That reference is what the unmodified stack retains, in its native dtypes, and $30.000$
  MiB/document of it is FP32 GDN recurrent state in excess of a BF16 encoding, so part of the reduction is
  dtype narrowing rather than the Eq.~3 packing step; against an all-BF16 reference of $106.235$
  MiB/document the two ratios become $3.7038\times$ and $10.9965\times$ ($90.91\%$)."
- Table 1 footnote, PDF p.7: "$30.000$ MiB/document of that $136.235$ MiB is FP32 GDN recurrent state in
  excess of a BF16 encoding; an all-BF16 reference is $106.235$ MiB/document, giving
  $1.00/3.7038/10.9965/11.3073/14.0969\times$ for the rows above."
- Bonus: Conclusion p.9 and Table 2's footnote ("$42.8\%$ of its $140.34$ MiB is FP32 GDN recurrent
  state").

**(ii) No non-all-BF16 row carries a "Q16/BF16" label.** `grep "Q16/BF16" main_r45_evidence.tex tables/`
returns **zero hits** (r44 had three in the main text plus the two table row labels). Row labels are now
`Full-prefix, native dtype` (Tables 1 and 2) and `\qcomem{} split Q16`. Sec. 4.2 was rewritten to make
the label honest: "Q16 denotes the unpacked reference: the residual is cast to BF16, while cache leaves
are kept in the dtype the stack produces, which for GDN recurrent state is FP32", and the Figure 2
caption and the appendix deployment table footnote carry the same redefinition. The two figure PDFs
contain no `Q16/BF16` string either.

**(iii) Both ratios labelled as re-analyses of the same rows; neither replaces the other.** Sec. 5.2,
immediately after the two ratios: "Both are exact re-analyses of the same archived rows on 60/60 items
--- the physical saving on this stack, and the like-for-like packing saving." Table 1's footnote and the
abstract both print both. Neither is retracted in favour of the other.

**(iv) The withdrawn A1 correction is absent.** `grep` for `13.2`, `13.6`, `13.619`, `13.168`, `13.9662`,
`11.73`, `11.61` over `main_r45_evidence.tex` and `tables/` returns **zero hits**. The superseded
metadata-omission mechanism from `revision/a1_store_reconciliation_20260902.md` appears nowhere.

### Independent recomputation from the 48 archived shards

Every accounting figure in the fix reproduces exactly:

```
full-prefix   mean stored_persistent = 136.235352 MiB   (paper 136.235)
split Q16                            =  34.683105       (paper  34.683)
frozen Q4/Q4/Q8                      =   9.660873       (paper   9.661)
FP32 GDN excess, 30 lower+upper GDN layers × 524288 × 2 B = 30.0000 MiB   (paper 30.000)
all-BF16 reference 136.235352 − 30 = 106.235352 MiB                       (paper 106.235)

published ratio  136.2354 / 9.6609 = 14.1018×      (paper 14.10×)
BF16 ratios: 3.703760 / 10.996454 / 11.307349 / 14.096942
             (paper 3.7038 / 10.9965 / 11.3073 / 14.0969)  ✓ all five
BF16 reduction = 90.9062%                                   (paper 90.91%)
Table 2: 30 GDN × 524288 × 4 B = 60 MiB;  60 / 140.34 = 42.75%  (paper 42.8%)
```

Crucially, the fix dissolves the impossibility T-02 identified:

```
published split/frozen        = 34.6831 / 9.6609 = 3.5901×   ← above the Q4 ceiling 3.5556×
BF16-normalised split/frozen  = 28.6831 / 9.6609 = 2.9690×   ← within format
```

So the mechanism is correct and the manuscript's numbers are sound.

### Why not `resolved`

1. **No component breakdown.** The ledger's `required_action` and the meta-review's binding sentence
   ("The authors must publish the component breakdown; they must not be required to print 11.73x") ask
   for, per Table 1 row, the byte split into packed residual codes, residual scale/bias, lower attention
   KV plus metadata, and lower GDN codes plus metadata, satisfying
   `Σ_c ceil(n_c/64)·(8b_c + 4)`. The manuscript contains no such table and no such identity. `grep` over
   the built PDF for `3.5556`, `ceiling`, `floor`, `conv_states`, `recurrent_states`, `28.683`, and the
   per-component MiB values from `revision/a11_provenance_20260902.md` §3.5 returns nothing. The analysis
   exists (A11 §3.5 has the full per-component table, verified item-by-item) but was not carried into the
   paper or an appendix.
2. **The Store column still reads as super-ceiling.** A reader applying Eq. 3 to Table 1's printed values
   gets 34.683/9.661 = 3.59×, above the 3.5556× Q4 ceiling — the exact check that triggered T-02. The
   BF16-normalised split value (28.683 MiB) is never printed; it must be back-derived by dividing 106.235
   by the footnote's 3.7038. The paper never names the apparent violation or says explicitly that the
   split row also carries 6.000 MiB of FP32 GDN excess.
3. **Minor framing inconsistency between the two tables.** Table 1 discloses the *excess* over BF16
   (30.000 MiB, 22.0% of 136.235); Table 2 discloses the *total* FP32 recurrent state (42.8% of 140.34).
   Both sentences are individually accurate but the two percentages are not comparable, and Table 2 gets
   no BF16-normalised ratio set.

### What remains

1. Add the per-component byte table (A11 §3.5 already has it) for all Table 1 rows as an appendix, with
   the `ceil(n/64)·(8b+4)` identity stated so a reader can check each row against Eq. 3.
2. Print the BF16-normalised Store column (106.235 / 28.683 / 9.661 / 9.395 / 7.536) rather than only the
   derived ratios, and state that 6.000 of the split row's 34.683 MiB is the same FP32 excess.
3. Optionally add one sentence naming the apparent Eq. 3 violation in the physical column and pointing to
   the dtype explanation — the reader-facing check T-02 was built on.
4. Give Table 2 the same excess figure (21.4%) alongside its 42.8%, or harmonise the two framings.

---

## 4. A11 finding C4 — the measured Read path is private materialization, and ForkAudit never ran on it

### Verdict: `resolved`

### OLD

`main_r44_structure.tex` Sec. 4.3 (l.310–320) described the borrow/COW discipline with no qualification:

> A borrowed recurrent base is read-only at setup and must rebind to private storage when the registered
> transition first mutates it; a partial KV tail is copied before append.

and stopped there. Contribution bullet 3 (l.126–130) read "**Ownership-safe reuse.** The seven-target
\method{} contract couples semantic relations to phase-indexed storage evidence across 96 ownership
configurations" with no statement of which configurations. The abstract, Figure 1 caption and Conclusion
likewise claimed ownership validation without scoping it off the quantized path. `grep` for
`materializ`/`private copy` in the r44 Read section: nothing.

### NEW — both halves of C4, in the main text, at every claim site

**Half A (the measured Read path is full private materialization, not borrow/COW)** — 2 sites, both main
text:

- Sec. 4.3, PDF p.5 (l.361–363): "This discipline is what \method{} audits; the Transformers
  implementation behind Tables~\ref{tab:qcomem-validation60} and~\ref{tab:qcomem-tradeoff} instead
  materializes a full private copy of the dequantized entry per query, so it shares nothing and exercises
  neither borrowing nor copy-on-write."
- Sec. 5.6 limitations, PDF p.9 (l.640–643): "... a $j=7$ packed Q4/Q4/Q8 path whose Read step
  materializes a full private copy of the entry rather than borrowing it, so ownership there transfers by
  design argument only."

**Half B (ForkAudit has never run on the quantized Read path)** — 8 sites, all main text (pp.1–9):

| Site | PDF p. | Text |
|---|---|---|
| Abstract | 1 | "that factorial runs full-prefix BF16 KV with no split depth and no quantization, so it audits the ownership discipline and not the quantized Read path" |
| Contribution bullet 3 | 2 | heading changed to "Ownership-safe reuse, **audited off the quantized path**"; "all full-prefix BF16 KV with no split depth and no quantization; the quantized Read path is not audited" |
| Figure 1 caption | 3 | "it ran only on a full-prefix BF16 configuration with no split depth and no quantization, so the packed Read path drawn here is unaudited" |
| Sec. 4.4 | 5 | "It runs on a full-prefix BF16 configuration ... so it validates that discipline and not the dequantize-then-fork Read path; the obligations specific to a packed entry --- dequantized-view immutability, residual-chunk binding, packed-entry lifetime --- are untested." |
| Sec. 5.1 "Executed systems" | 6 | "so no \method{} verdict here covers those packed entries" |
| Sec. 5.5 opening | 8 | "and therefore audits the ownership discipline of Section~\ref{sec:read} rather than the quantized Read path measured in Tables 1 and 2" |
| Sec. 5.6 | 9 | "\method{} has not been run on the quantized Read path: every ownership cell uses full-prefix BF16 KV with no split depth and no dequantization" |
| Conclusion | 9 | "localizing request-local ownership violations on a full-prefix BF16 configuration" |

### Reasoning

Both halves are stated plainly in the running main text, not buried in an appendix — I confirmed page
placement by `pdftotext` on the built PDF, and every one of the ten sites falls on pp.1–9. The scoping
caveat now travels with the claim: every place the paper asserts ownership validation carries it,
including the abstract, the contribution bullet, the figure caption and the conclusion. Sec. 4.4 goes
further than C4 required by naming the three obligations a packed entry would need and calling them
untested. Sec. 5.6 states the honest consequence: "ownership there transfers by design argument only."

Two small observations that do not change the verdict:

- Figure 1 still *draws* "immutable document view / COW / rebind" on the Read path. Its caption now says
  that path is unaudited, but not that the measured implementation does not use COW at all. A reader who
  reads only the figure could still infer borrow/COW is what runs.
- Section 4.3's title is unchanged (`Online Read: ownership-safe fork and suffix replay`), though its
  third sentence supplies the correction.

Neither is a residual of C4 as written; both are cosmetic and I flag them only for completeness.

---

## Cross-cutting note

`revision/a1_store_reconciliation_20260902.md` — whose 13.2–13.6x correction was withdrawn — remains in
the repository. Nothing in the manuscript cites it, and none of its superseded numbers appear in
`main_r45_evidence.tex` or `tables/`. Verified clean.

`revision/r45_integration_20260903.md` heads its A11 section "closes `R44-5-01`…`R44-5-04`" but never
mentions `method_provenance.tsv`. The claim in the section header is not supported for `R44-5-02`; see
issue 2 above.
