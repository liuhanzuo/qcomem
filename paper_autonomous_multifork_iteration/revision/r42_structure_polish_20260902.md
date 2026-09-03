# R42 Q-CoMem structure and evidence revision (2026-09-02)

## Deliverables

- Manuscript source: `main_r42_qcomem_capacity.tex`
- Main 60-item table: `tables/qcomem_validation60_r42.tex`
- Main eight-item systems table: `tables/qcomem_tradeoff_r42.tex`
- Method figure: `figures/qcomem_pipeline_r42.pdf`
- Built PDF: `output/pdf/Q-CoMem_R42_Structure_Revision.pdf`

The R41 source was preserved. R42 is a non-overwriting successor.

## Story and structure

The main paper now follows Introduction, Related Work, Methodology, Experiments,
and Conclusion. Motivation is carried by the capacity-first problem statement:
reduce the complete retained per-document state of hybrid split replay. The
method follows directly: Q-CoMem quantizes the split residual together with the
required lower attention KV and convolution/recurrent state, then reconstructs
the suffix online while keeping mutable request state private. ForkAudit is
positioned as an ownership validator rather than the primary method contribution.

## Main evidence

On the archival 60-item Qasper/2WikiMQA validation cohort, frozen Q4/Q4/Q8 has
mean Store 9.661 MiB/document versus 136.235 MiB/document for full-prefix Q16
(14.10x smaller). Mean F1 is 54.24 versus 54.62 for split Q16, with paired
delta -0.39 points and 95% bootstrap interval [-2.04, 1.08]. The interval is
not presented as equivalence or noninferiority.

The 60-item table includes TTFT and Recall columns but leaves all six cells in
each column blank because neither metric was recorded. The separate eight-item
systems execution includes measured TTFT/TPOT/tokens-per-second and leaves all
six Recall cells blank. It uses source indices 6--9 from each dataset, so it is
an overlapping subset of the 60-item cohort under a different execution
protocol, not an independent replication; the panels are not pooled.

The timing evidence does not support TTFT acceleration: full-prefix Q16 records
0.163 s TTFT, versus about 0.673--0.674 s for Q-CoMem. Q-CoMem TPOT is within
2.5% of full-prefix in this eight-item execution, while generated-token
throughput is lower. The manuscript therefore claims retained-capacity
reduction with an explicit online cost.

Store denotes retained tensor payload in both tables, but the implementations
are kept distinct: the 60-item panel uses persistent-component byte accounting,
whereas the eight-item execution uses the physical byte-range union owned by a
retained entry. Neither is total/process/NVML memory or serving capacity.

## Verification

- Independent claim audit: PASS; no scientific blocker.
- 60-item raw-first replay: PASS (66 mirrored files, 48 shards, 360 F1 rows,
  24 bootstrap intervals, Store accounting, archived aggregate).
- Evidence registry: `E-QCOMEM-60-VALIDATION-20260812D-A` marked active,
  manuscript-integrated, and fresh-review-complete.
- LaTeX: 31 pages total; no overfull boxes, undefined references/citations,
  multiply defined labels, or rerun warnings in the final log.
- Visual QA: title/method figure, both main tables, limitations, conclusion,
  and references inspected after the final build.

Final SHA-256 values:

- `main_r42_qcomem_capacity.tex`: `47a4eb408bc9c10b071dd2944aab88346d5ff805bb1f31c285ad07339e46a47b`
- `tables/qcomem_validation60_r42.tex`: `83abb4131c2c0051ab8e93142ba068d2c921f0648dcef69c7693a359328e28a2`
- `tables/qcomem_tradeoff_r42.tex`: `4389a044086f78dece7dde916a8ee62c1061e4a97a57afc44d5e7ea7177003c7`
- `figures/qcomem_pipeline_r42.pdf`: `1c562f1806408eccfedd6f8095c20f882c217533d475b855e4189da4662eec35`
- `output/pdf/Q-CoMem_R42_Structure_Revision.pdf`: `43fa85087e0e579436de1db5da3e9de269a1fb17271cda76fec5016e73f7e49f`

## Concurrent formal-run status

V32 (QS Job 256220 / Trial 1939465) failed before scientific execution at the
staging single-link archive gate. Scheduling and node health were normal; the
canonical V6 archive had the correct SHA-256 but more than one hard link. This
is a preflight/deployment failure and is not experimental evidence. A minimal
non-overwriting successor keeps the gate strict and uses a fresh byte-copied,
single-link archive in a new scratch path before formal staging.

That successor is V33 at
`evidence/r40_independent_live_binding_v33_v6_singlelink_copy`. It passed
32/32 focused formal tests, 13/13 packaging tests, 15/15 tree tests, and 15/15
Linux-stage tests; the remote roundtrip was byte-identical. It was submitted as
QS Job 256220 / Trial 1939532 and was `Uncommit` at the 2026-09-02 12:49 CST
status check. V33 remains uncited pending terminal completion and independent
post-run audit.
