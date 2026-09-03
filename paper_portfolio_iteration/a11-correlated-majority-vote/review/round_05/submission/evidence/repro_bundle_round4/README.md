# Round-4 reviewer-safe replay bundle

This bundle preserves an exact recovered main runner and its reported output,
then supplies a dependency-free replay of the count-level main table.  The
replay is intentionally narrower than an end-to-end raw-data reproduction:
it consumes the derived manifest below, not the source parquet or any problem
text.

## What is included

| Item | Purpose |
| --- | --- |
| `omr_problem_manifest.json` | Anonymous retained-row manifest: stable problem hash, retained source order, integer count `K`, and seeded split only. |
| `replay_fit_cal_test_stdlib.py` | Standard-library verifier that rebuilds the main result JSON from the manifest. |
| `expected_fit_cal_test_r469_result.json` | Recovered expected main result; SHA-256 is `b114c72d9ab1cf1a6ba1d2bd734433c06bd4d5cbd19bf93be0964edf6fc8a5f7`. |
| `upstream_recovered/` | Exact recovered source runners for the main, drift, shard-transfer, OpenR1, and RLVE analyses. |
| `recovered_outputs/` | Exact recovered secondary result JSONs corresponding to those runners. |

The manifest contains no raw problem text or answer text.  Its stable
`problem_sha256` field is an audit identifier, not a secrecy guarantee: anyone
already holding a candidate problem can compare its SHA-256 value.  The bundle
does not include the source parquet and therefore cannot independently establish
the origin of those hashes.

## Derived-data contract

The manifest records the exact selection rule and resulting counts:

| Quantity | Value |
| --- | ---: |
| Source rows | 22,230 |
| Valid count rows | 12,423 |
| Retained deduplicated problems | 11,607 |
| Dropped invalid rows | 9,807 |
| Dropped duplicate problems | 816 |
| Rollouts per retained problem (`K` denominator) | 32 |
| Split seed | 20,260,815 |
| FIT / CAL / TEST | 4,000 / 4,000 / 3,607 |

Rows are retained when `pass_rate_72b_tir` is non-null, non-NaN, and not the
literal `n/a`; the first occurrence of each raw problem cell is retained; and
`K = int(round(float(pass_rate_72b_tir) * 32))`.  The source-parquet identity
recorded in the manifest is SHA-256
`6640e85f89bb829702a7d30b622acbe2bdb76bb943eb37d77ef20ab145081ddc`
(220,414,147 bytes).

## Local replay

Requires only Python 3 and its standard library.  From this directory, run:

```bash
python3 replay_fit_cal_test_stdlib.py omr_problem_manifest.json \
  --output replayed.json \
  --expected expected_fit_cal_test_r469_result.json
shasum -a 256 replayed.json
```

The verifier checks the manifest header, unique problem hashes, retained-order
split reconstruction, exact hypergeometric/DP calculations, and the runner's
deterministic Monte-Carlo self-check.  It then demands both semantic equality
and byte-for-byte SHA-256 equality with the expected JSON.  A successful run
prints the expected hash above.

## Reproducibility boundary

`upstream_recovered/fit_cal_test_r469.py` is an exact recovered upstream
runner, but it needs the excluded source parquet plus `pandas` and `pyarrow`.
The standard-library verifier instead replays the reported main table from the
reviewer-safe, count-level derivation.  This establishes byte-exact replay of
the main JSON conditional on the manifest; it is not a clean-room reproduction
from the raw parquet.  The four secondary runners and their outputs are
preserved with hashes for provenance, but they are not locally rerun here.
