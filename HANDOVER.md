# Q-CoMem — handover

Written 2026-09-03, at the point where work was paused after round 51.

This is the document to read first. `paper_autonomous_multifork_iteration/state/decision_log.md`
is the running journal and is long; this file is the state of play.

---

## 1. Where the paper stands

**Current manuscript:** `paper_autonomous_multifork_iteration/main_r51_bodytable.tex`
**Compiled:** `output/pdf/Q-CoMem_R51_Composition_Revision.pdf`
**Score:** internal blind panel, unanimous **4/10**, dimension medians 2/2/2.

Score history, all under the same five-role protocol:

| round | manuscript | panel | meta | criticals |
|---|---|---|---|---|
| 44 | structure revision | 4 4 4 4 4 | 4 | 6 |
| 46 | quantized-exact-cache baseline added | 4 4 4 4 4 | 4 | 4 |
| 51 | composition run + total-memory claim | 4 4 4 4 4 | pending | 5 |

The score has not moved across three rounds. Criticals fell then rose. Two
independent meta-reviews put the ceiling at 6 without new experiments and 8 only
with the composition experiment, which has since been run.

R40 through R50 are kept as checkpoints. The skill requires being able to select
an earlier round as final; do not delete them.

---

## 2. What the paper claims, and what is actually measured

**The method.** Q-CoMem writes a document once through the first `j=7` of 40
layers and retains the boundary residual plus the lower-layer attention KV and
GDN convolution/recurrent state, group-quantized at Q4/Q4/Q8 in 64-value groups
with BF16 scale and bias. Each query dequantizes that entry, replays the query
through the lower layers, and recomputes layers 7-39.

**Hardware and model.** Eight NVIDIA H20-3e, 143,771 MiB each, one rank per GPU,
batch one. Qwen3.5-35B-A3B snapshot `59d61f3`, BF16, 40 layers = 10 attention +
30 GDN, 69.3 GB resident. torch 2.11.0+cu129, transformers 5.14.1; the ForkAudit
line additionally uses vLLM 0.26.

**Cohorts.**

| cohort | size | measures |
|---|---|---|
| quality | 60 LongBench-v1 items, Qasper + 2WikiMQA source indices 6-35, 4096-token cap | retained Store, F1, paired bootstrap intervals |
| timing | the 8-item subset, indices 6-9, 3 repeats | TTFT, TPOT, throughput |
| composition (C1) | 8 items, fanout N in {1,2,4,8} | shared packed entry, ForkAudit, transient working set |
| ForkAudit factorial | PG-19 books, N in {1,8,32}, 96 configurations | ownership correctness |

**The results that are solid.** Retained state falls from 136.235 to 9.661
MiB/document, 14.10x, with a paired F1 difference against full-prefix of
-0.45 [-2.06, 0.99] on 60 items. Three separate reviewers independently
recomputed every printed ratio and byte identity and found them exact. This is
the paper's strongest evidence and it survived every round.

**The results that go against the paper, all measured, all currently reported.**
Q-CoMem reaches the first token later than both exact caching and an honest
dense baseline; the deployment bench has now shown this three times on real
items. Peak transient allocation is higher than full-prefix at every fanout
measured. Sharing in the composition run covers about 4 percent of per-request
state. The allocator result ties the paper's own paged-prefix baseline. A
conventional base invariant catches the same historical alias the audit does.

---

## 3. What is broken right now

**The body's Table 1 should probably be withdrawn.** It presents a total-memory
model `M_total(D,N) = S*D + T(N)`, introduced in round 50 as the paper's lead
claim. Five reviewers attacked it and I verified each objection myself. It has
six defects, all introduced by me:

1. It adds two terms that Section 5.1, Table 1's own footnote and Section 5.7
   each state are *not* additive, and additivity is never argued.
2. Table 8, the paper's own cohort-authorization table, excludes total memory by
   name from the three cohorts Table 1 is built from.
3. Arm mixing: `S = 9.661` comes from the private-materialization path while
   `T = 809.68 + 109.76N` is the shared-entry fit. Using the private fit
   `790.50 + 148.38N` moves the N=8 crossover from 2.01 to 4.30 documents and
   flips the D=4, N=8 row from 1.15x to 0.98x, reversing which system wins.
4. Denominator mixing: totals use native-dtype `S=136.235` while the paper's
   like-for-like headline uses 106.235. Under the paper's own convention the
   crossovers are 6.73/6.14/4.97/2.63, not 5.13/4.69/3.79/2.01.
5. The regime is stated backwards. The crossover falls as N rises, so high
   concurrency makes the method *easier* to justify. The abstract says the
   opposite.
6. Two-parameter fits over four points with residuals to 70 MiB, yet crossovers
   printed to three significant figures; and D extrapolated to 100 from runs
   that held one document, in a paper that refuses to extrapolate past N=8.

**Recommendation:** withdraw the total-memory model from the body, restore the
60-item Store/F1 table (currently appendix Table 4) to the lead position, and
keep the transient measurements as a limitation rather than a claim. A
meta-review was running to adjudicate this when work paused; check
`review/round_51/meta_review.json` before acting, and prefer its judgement over
mine.

**The main text exceeds nine pages.** The Conclusion continues onto page 10. My
earlier page check was wrong: it verified that the Conclusion *appears* on page
9, not that it *ends* there. Use the corrected check in section 5 below.

**The composition run has no provenance entry.** C1 supplies a whole section, a
body table and an abstract claim, but appears in none of
`experiment_registry.json`, `claim_evidence_map.tsv` or `method_provenance.tsv`.
Worse, the newest registry entry `E-QCOMEM-60-OPPOINT-20260903A` still states
that no frozen-versus-Q8 paired interval exists and that transient was not
measured — both of which the PDF now reports. **The artifact currently
contradicts the paper.** Fix this before any submission.

**HCache (EuroSys 2025) may be closer prior art than the cited XQuant** on the
persist-an-intermediate-and-recompute axis. Unverified; check it.

---

## 4. Traps that cost real time. Read before touching the cluster.

**The WebIDE and the jobs see different filesystems.** In the QS Code container,
`/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo` is a *local overlay* —
`findmnt` reports overlay at every level and its device id equals `/`. Jobs see
the real JuiceFS there. Anything written from the WebIDE is invisible to jobs.
This destroyed two V33 trials before it was diagnosed. `qs cloudfs list` shows
exactly one mountable path and the H20 cluster (id 53) is the only one that can
mount it; a CPU cluster will reject the job at scheduling.

**Moving code to the cluster.** Base64 a tarball into the job command. Do not
pipe a script through `tr '\n' ';'` — it destroys heredocs and Python
indentation. The `command` column truncates above roughly 63 KB, and encoding
the payload twice inflates it by a third; encode once and inline it in a
single-line command.

**Job submission that works:**
```
--queue-id 408 --cloud-id 6 --cluster-id 53 --resource-package-id 183 \
--overuse --mount-path /mnt/tidal-alsh-hilab/dataset/diandian
```
`--elastic` is rejected on this queue. Queue contention is real: jobs have sat
in `Uncommit` for over an hour while later submissions ran.

**Assets on the cluster** (not in this repo, and they should not be):
```
indep-bench_assets/models/Qwen3.5-35B-A3B-59d61f3
indep-bench_assets/data/qcomem-longbench-validation/longbench_validation.jsonl
indep-bench_assets/envs/vllm-cu129-v1
indep-bench/qcomem_gpu_r45_20260902          # the uploaded code
```

**A proxy exists** at `http://10.3.4.34:3128`, written into `~/.bashrc` on the
dev box. `huggingface.co` times out without it and returns 200 with it; real
weight byte-ranges return 206.

---

## 5. Standing rules that must not be relaxed

**The admissibility rule** — `revision/derived_vs_measured_20260902.md`.
Re-analysis of archived per-item measurements may enter the manuscript. A
projected number for a configuration that was never run may not. This rule has
already caught several projections that measurement later contradicted,
including my own ~3.1x estimate, which measurement replaced with 5.84x — itself
later found to be bit-width confounded.

**Verify builds yourself.** Every integration agent reported a clean build; on
three occasions independent rebuilding found something the report omitted,
including a version whose body contained no figure at all.

**The corrected page check.** Confirm the Conclusion *ends* on page 9 and that
page 10 opens with a statement or References, not with continued main text:
```
pdftotext -layout <pdf> - | awk 'BEGIN{RS="\f"} NR==10' | head -5
```

**Never move adverse evidence out of the body to make room.** If space is
needed, cut something that argues *for* the paper. This was violated once and
reverted.

**Update the evidence maps in the same pass as the manuscript.** Two consecutive
rounds shipped with the maps stale, and reviewers found it both times.

**Anonymity.** `scripts/anonymize_platform_receipts.py` is fail-closed and takes
`--patterns`. Its default covers only platform receipts; the evidence maps and
the experiment registry carry author paths too, and a receipt-only scan passes
while the bundle still leaks. That exact gap shipped 52 identity leaks into a
reviewer bundle. Run it over everything destined for a supplement.

---

## 6. Open actions

`review/experiment_response_plan.json` is authoritative. In priority order:

1. Adjudicate and act on the total-memory model (see section 3).
2. Register C1 in all three artifact files; correct the stale registry entry.
3. Fix the page overflow.
4. C2 — run the width-matched exact cache, the frozen bit vector applied to the
   unsplit entry, predicted near 37.85 MiB/document. This is the arm whose
   absence makes the 5.84x figure confounded.
5. C3 — attribute the full-prefix Q8 arm's 3.78-point F1 loss by state family.
6. Check the HCache citation.
7. V33 — the borrowed-transition live binding, still unrun. Re-upload the frozen
   V6 archive under a new operator path, then reuse the tested dual byte-copy
   command.

---

## 7. Repository notes

`.gitignore` excludes, with reasons recorded inline: 19.5 GB of LaTeX build
attempts, per-round submission staging copies and duplicated review PDFs,
transport tarballs and wheels, model checkpoints, and the unrelated `CampusOS/`
and `01_p5/` trees. No source, manuscript, state file or archived measurement
was dropped.

`evidence/` is 3.3 GB across 127 packages. About 0.48 GB is duplication:
`r40_ci_cost_accounting_v1/prepared_inputs/primary_manifest_view/` is a
byte-identical copy of `round_04_rr2_package/upstream/`. Deduplicating is
tempting but those packages carry `MANIFEST.sha256` integrity checks; read the
replay scripts before touching them or the checks will fail.

A single `forkaudit-shard-*.json` is 59 MB because it records per-phase
ownership traces — storage ids, full descriptors and content hashes for every
tensor at setup, transition and final, across 96 configurations. That granularity
is what the fail-closed audit contract requires. GitHub warns above 50 MB but
only rejects above 100 MB, so these push fine.
