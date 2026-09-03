# RR2 A1 terminal-closure evidence

## Reviewer summary

This package records the final terminally closed rerun of the existing RR2
ForkAudit primary factorial.  QuickSilver Job `246643`, Trial `1872962`
completed with process exit code zero and is classified `valid_positive`.
All eight rank shards completed, the blind aggregate passed, the retained and
terminal source ledgers matched, the model read-lease closed cleanly, and an
independent aggregate replay produced a byte-identical summary.

The formal result is not copied into this repository because the raw witness
sidecars and model-bound receipts are large.  Instead,
`formal_refreeze_20260821a/final-formal-trial1872962-terminal-validation.json`
records exact TidalFS locators and SHA-256 bindings for the formal summary,
scientific-artifact ledger, execution-identity receipt, model-load authority
and closure, and independent replay.  The large remote payload remains
immutable by content hash; this reviewer-safe index does not replace it.

## Exact outcome

- Scientific classification: `valid_positive`.
- `scientific_run_valid`, `formal_ready`, `passed`, and
  `hypothesis_passed`: all true.
- Eight of eight shards and the exact four-cell factorial completed.
- The selected oracle passed on all eight ranks; maximum recorded relative L2
  was `0.005`.
- The matched mutant campaign passed with no escaped, wrong-gate, unexpected
  crash, or clean-false-positive IDs.
- The preregistration and terminal 36-file code ledgers both have raw SHA-256
  `c18d63c72acaf00d20006278998052c12b3894d116d1724ab855f77d04af011a`.
- The terminal source tree contained no `__pycache__` directory.
- Execution identity was stable and bound four compiled-kernel artifacts and
  43 terminal cache files.  No explicit autotuning artifact was exposed by
  the bound cache roots; that limitation is recorded rather than inferred
  away.
- The 14-file model lease closed with no break, and terminal full-content
  rehashing passed.
- The independent replay summary is byte-identical to the formal summary;
  both have raw SHA-256
  `5c510e59e62181b8c31dc722a7fca1337a5972b8533cd7ff556d266cb29e34c0`.

## Failure and repair history

The first interactive focused-suite attempt produced 163 passes and two
errors because the source-only transfer package did not contain the already
frozen W FP32 archive fixture expected outside `CODE_DIR`.  Restoring the nine
fixture files byte-identically fixed the two tests without changing the
scientific protocol or the 36-file code ledger.  A manual debug invocation
then created `__pycache__`; the exact launcher correctly rejected that dirty
tree.  On a fresh tree, launcher-managed `-B` and cache redirection passed all
165 focused tests while leaving the source tree unchanged.  Invalid and
intentional-stop attempts remain recorded and are not treated as formal
evidence.

The interactive debug node was confirmed idle after validation and stopped:
Job `246644`, Trial `1872365`, terminal platform status `Terminated`.  The
pre-stop GPU/process check and cleanup receipt are recorded separately and do
not contribute scientific evidence.

## Scope boundary

This result supports the frozen Qwen3.5-35B-A3B RR2 vLLM-Q16 four-cell
ownership factorial, its selected semantic oracle and matched mutant
campaign, and the source/environment/model/artifact terminal-closure checks
executed in Trial `1872962`.  The scientific model, data, protocol, factorial,
and query/oracle inputs are unchanged from W; the only repair was
bootstrap/test-fixture packaging plus execution-identity enforcement.

It does not establish cross-runtime or cross-model generality, scheduler or
continuous-batching correctness, concurrent-kernel safety, throughput or
capacity, best-possible performance, or completeness over all faults.  No
vLLM/SGLang serving control, GDN oracle, lifecycle, Transformers, Marconi, or
scheduler-interleaving experiment was rerun for this entry.

## Reviewer-safe files

- `formal_refreeze_20260821a/final-formal-trial1872962-terminal-validation.json`:
  terminal classification, platform receipt, remote locators, hashes, debug
  history, and explicit limitations.
- `formal_refreeze_20260821a/derived/code.sha256`: exact frozen 36-file ledger.
- `formal_refreeze_20260821a/derived/release-manifest.expected.json`: frozen
  release identity.
- `formal_refreeze_20260821a/source/gpu/`: exact executed source snapshot.
- `formal_refreeze_20260821a/tests/`: affected static and ledger checks.
- `formal_refreeze_20260821a/recovery-trial1872325-terminal-invalid.json`:
  preserved invalid recovery attempt.
- `debug-node-trial1872365-cleanup.json`: idle pre-stop check and terminal QS
  cleanup receipt for the interactive node.
- `implementation/`: focused A1 implementation copies for inspection.
- `SHA256SUMS`: content bindings for the compact reviewer-safe index.  It
  intentionally excludes large raw sidecars and transfer tarballs.

The repository's current `gpu/run_qcomem_qwen35_forkaudit_review_revision.py`
may evolve after this refreeze.  Review the executed snapshot under
`formal_refreeze_20260821a/source/gpu/`, whose runner raw SHA-256 is
`9da619fc037e2c670b146d778fd9f4d5344212b7e525f3d3f26a077f79d67775`.

## Local verification

From this directory:

```bash
sha256sum -c SHA256SUMS
jq -e '
  .classification == "valid_positive" and
  .trial_id == 1872962 and
  .process_exit_code == 0 and
  .formal_result.rank_count == 8 and
  .formal_result.passed == true and
  .independent_replay.byte_identical_to_formal_summary == true and
  .formal_rerun_still_required == false
' formal_refreeze_20260821a/final-formal-trial1872962-terminal-validation.json
```

The first command verifies the compact local index.  Replaying the large raw
formal bundle requires access to the recorded TidalFS locators.
