# A11 BoolQ ordered formal run

This directory is a submission-ready implementation of the prospective
ordered-rollout, gold-correctness, and realized-cost experiment.  It does not
reinterpret the old OpenMathReasoning pass/fail replay as an online result.

The observable bit is a parsed **Yes vote**.  `x_k` is the number of Yes votes
among the first `k` completed rollouts, the online answer is their Yes/No
majority, and the FULL-N answer is the 32-vote majority.  The policy-visible
`PromptTask` and acquisition/stopping call graph have no gold field.  BoolQ
gold is joined only after both paired TEST decisions are append-only and
hash-sealed.  This is call-graph isolation inside one process, not a claim of
separate-process secrecy.

The outcome-blind carrier allocation first retains one deterministically
selected question per exact stripped passage, then hashes those 10,144 passage
representatives into FIT/CAL/TEST.  Each split has 3,000 distinct passages and
there is zero exact-passage overlap across splits; 2,553 extra questions from
repeated passages are discarded before allocation.

## Files

- `protocol.json` freezes the data, model, split, parser, sampling, policy,
  CAL gate, TEST endpoints, and audit rules.
- `run_boolq_ordered.py` prepares the outcome-blind split, acquires FIT/CAL
  traces, locks the fitted table, applies the alpha=.05 CAL gate, executes the
  paired randomized TEST arms, seals all 6,000 primary decisions, only then
  runs the deferred shadow continuations, and audits/analyzes the ledgers.
- `launch_boolq_ordered_8gpu.sh` performs fail-closed artifact checks, verifies
  the complete pinned model snapshot staged on CloudFS (an exact 14-file
  top-level set and ledger SHA
  `3ee6c9510b7e50bfcd46d6df33cafa3e2019f13a6a09bf1d2f9e80cdfe1164e8`), starts eight
  one-GPU vLLM servers, and invokes the runner.
- `test_run_boolq_ordered.py` is a stdlib-only focused CPU test suite.
- `qs_preview.json` is the resolved QS preview.  It is not a submission receipt.

The launcher refuses an existing `RUN_DIR`.  Each task has at most one live
request, although each GPU server may service up to 16 tasks concurrently.
Consequently a BAYES-H stop has zero in-flight requests by construction; the
cancellation ledger records this as `not_applicable_sequential_no_prefetch`,
not as a claimed successful cancellation.

## Local checks

```bash
python3 -m unittest -v test_run_boolq_ordered.py
python3 -m py_compile run_boolq_ordered.py test_run_boolq_ordered.py
```

## Formal execution boundary

The intended QS entry point is `launch_boolq_ordered_8gpu.sh` on queue 471
(`Verifier`), without elastic/overuse.  It shares the original
`general-post-training` scene and authoritative quota diagnostics reported 16
available H20 GPUs, enough for the frozen eight-GPU request.
Do not call
the runner against a partially prepared directory.  FIT must complete before
the table lock; CAL must then pass the frozen alpha=.05 gate before TEST files
are opened.  A failed import, checksum, server-health, data, CAL, or integrity
gate is recorded separately and is not experimental evidence.
