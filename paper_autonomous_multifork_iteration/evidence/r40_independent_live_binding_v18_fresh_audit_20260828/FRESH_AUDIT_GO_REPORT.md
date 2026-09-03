# R40 live-binding v18 fresh audit

Verdict: **GO_TO_FORMAL_H20**.

This was a fresh, isolated, read-only audit of
`evidence/r40_independent_live_binding_v18_self_contained_stage`. No frozen v18
file or archive was modified, and no GPU work was executed. The verdict only
authorizes the exact formal H20 path; it is not scientific evidence.

## Approved identities

| Object | SHA-256 |
|---|---|
| v18 source ledger | `c28418d468aeb2b2269643584e28881d121c018ae2c11a45364a0a255e86e308` |
| v18 overlay archive | `a775458bd171bf365193800b886bc5140c5caf6fcb375959d1e2b5a119431475` |
| canonical v6 archive | `306daba7b79b045a306f0b22d6434143dd568cf1f3b6af7114ad1a4ebe1d6f82` |
| frozen outer launcher | `861308d44b80c9a2f8b2cd1ff23ecf1166866fd6e44ae6696844aab53d355972` |
| generated formal launcher | `8d5ba77f9b61b760346334b4bca041e1ac0176719c5b8bd2e616a29b24226636` |
| v6 launcher input | `299907b4f95e7f5d8873ef5d810698640cc525ec9d2af647d325465b150e69ee` |
| clean-member ledger | `4aebbe31e4d7089d7e2c09c94e37f7ebbd7ae9cf19327c18430891df1a809b7d` |
| AppleDouble exclusion ledger | `384c37261b2eda860d5e9285811e7d6708c81577d49aa014a1a458b153074eba` |
| scientific payload manifest | `ae5b7e404bb8e3fd004e69df716c4a80c29f7928230210dd9798356bb0efa59a` |
| scientific equivalence record | `6c8775053b745dac835026b4435f1ac3edcce6e65cfc628bd63276f6c0ff12d1` |
| immutable runner | `9da619fc037e2c670b146d778fd9f4d5344212b7e525f3d3f26a077f79d67775` |
| immutable builder | `546efd59e2833034bc2e24d4cc0e6077f5a408275e359af43cd96f7f71cad16e` |

## Independent closure checks

- The package tree contains 41 files and five directories, with no symlinks,
  special nodes, hardlinks, bytecode, or unledgered source payloads.
- All 38 source-ledger rows are unique, sorted, and hash-correct. Those 38
  files plus the ledger exactly equal the overlay's 39 safe regular members.
  An independent deterministic rebuild is byte-identical.
- The canonical v6 archive has exactly 260 consumed members: 130 retained
  nodes (98 files and 32 directories) plus 130 validated AppleDouble members
  with retained companions.
- The clean stage has exactly 175 nodes (138 files and 37 directories), zero
  `._*` paths, and a matching stage receipt. The v16 sibling package is absent.
- All ten v18 scientific payload files are byte-equal to v16, and the
  self-contained verification accesses no v16 sibling.
- The v6 launcher is opened once with `O_NOFOLLOW`; one stable descriptor
  snapshot feeds hashing, strict UTF-8 decoding, and transformation. Path-swap
  and invalid-UTF-8 counterexamples fail closed.
- The generated 326-line launcher has the approved hash, passes `bash -n`, and
  retains the fixed stage/result paths, pre-GPU 162/162 zero-skip gate, exact
  eight-H20 UUID binding, CUDA smoke, science, final rehash, and terminal
  closure.

## Executed local tests

- Packaged suite: **86/86 passed, zero failed, zero skipped**.
- Static audit: **128/128 satisfied**; regenerated JSON is byte-identical.
- Raw v6 replay: 162 discovered, two expected AppleDouble errors, zero
  failures, one environment-only skip.
- Clean replay: 162 discovered, zero errors, zero failures, one
  environment-only skip.
- Clean-stage formal preflight: **86/86 passed, zero skipped**, with the v16
  sibling absent.

The only local skip is the real Transformers-5.14.1 Qwen call because the Mac
environment lacks that exact dependency. This is nonblocking: the frozen formal
launcher requires **162/162 with zero skips before CUDA or science**.

## Authorized formal path

Prepare the fixed clean stage using only the approved hashes:

```bash
python3 -B scripts/stage_v6_clean.py prepare \
  --output-root /mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r40_v18_clean_20260828a \
  --v6-archive "$CANONICAL_V6_ARCHIVE" \
  --overlay-archive "$V18_OVERLAY_ARCHIVE" \
  --clean-ledger v6-clean-members.json \
  --exclusion-ledger v6-appledouble-exclusions.json \
  --expected-v6-sha256 306daba7b79b045a306f0b22d6434143dd568cf1f3b6af7114ad1a4ebe1d6f82 \
  --expected-overlay-sha256 a775458bd171bf365193800b886bc5140c5caf6fcb375959d1e2b5a119431475
```

Then run only the staged launcher, explicitly unsetting both self-test flags and
supplying the two authorizations plus all three external approvals:

```bash
env -u R40_LAUNCHER_HANDLER_SELFTEST \
    -u R40_LAUNCHER_ATOMIC_GATE_SELFTEST \
  R40_H20_EXECUTION_AUTHORIZED=yes \
  R40_V18_FRESH_AUDIT_APPROVED=yes \
  R40_V18_APPROVED_SOURCE_LEDGER_SHA256=c28418d468aeb2b2269643584e28881d121c018ae2c11a45364a0a255e86e308 \
  R40_V18_APPROVED_ARCHIVE_SHA256=a775458bd171bf365193800b886bc5140c5caf6fcb375959d1e2b5a119431475 \
  R40_V18_APPROVED_V6_ARCHIVE_SHA256=306daba7b79b045a306f0b22d6434143dd568cf1f3b6af7114ad1a4ebe1d6f82 \
  R40_V18_CANONICAL_V6_ARCHIVE="$CANONICAL_V6_ARCHIVE" \
  R40_V18_OVERLAY_ARCHIVE="$V18_OVERLAY_ARCHIVE" \
  bash /mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r40_v18_clean_20260828a/paper_autonomous_multifork_iteration/evidence/r40_independent_live_binding_v18_self_contained_stage/formal/launch_h20.sh
```

The fixed stage, result root, and one-shot marker must be absent before action.
Any failure before scientific execution is an infrastructure/preflight failure,
not experimental evidence.

## Remaining runtime gates and boundary

The H20 environment must pass 162/162 with zero skips, use PyTorch
`2.11.0+cu129`, pass CUDA smoke, and bind exactly eight distinct H20 GPUs with
compute capability 9.0 and BF16 support before science. The result root must be
new and terminal closure must complete exactly.

A passing run supports only callback-visible clean-rank agreement for the fixed
N=8 cell and policies between the independently frozen 60-coordinate pre-build
reference, the real returned group, 480 direct clone edges per rank, the
64-call lifecycle rebind sequence, and actual 540-row phase serializers. It does
not establish model-wide correctness, production fault sensitivity,
opaque-forward transient-alias detection, or adversarial same-process integrity.
LB01--LB04 are local mechanism tests, not H20 faults or sensitivity evidence.
