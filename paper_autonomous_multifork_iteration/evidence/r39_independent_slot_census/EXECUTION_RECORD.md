# Local execution record

Execution date: 2026-08-26 (Asia/Shanghai)

Environment:

- macOS 26.4.1, build 25E253
- Darwin 25.4.0, arm64
- Python 3.9.6
- no GPU used
- no QS resource created, stopped, or deleted

Frozen inputs:

- R33 raw H20 capture SHA-256:
  `50d39cfcea072fb770da539d90abeddcd8a40802b88f4f95315001333c09e974`
- R33 preregistration SHA-256:
  `67c001a04e61006967befe1ff1e26018b58bc02630b8e48dd40f16067a44ea65`
- R39 protocol SHA-256:
  `6d9cc60f49027a6a22f6d36eb6d1c621a72172084aec7c47e98c2162da33758f`

Executed commands, from `paper_autonomous_multifork_iteration`:

```bash
python3 -m unittest discover -s evidence/r39_independent_slot_census/tests -p 'test_*.py' -v
```

Result: 8/8 tests passed.  The eighth test executes the complete fresh-formal
binding path over the immutable archived H20 result: preexecution census,
hash-bound clean audit, copy-only controls, and terminal aggregation.

```bash
python3 evidence/r39_independent_slot_census/scripts/audit_independent_slot_census.py \
  --protocol evidence/r39_independent_slot_census/protocol.json \
  --input evidence/r33_independent_capture/formal_h20/result/raw/out-of-process-gdn-capture.json \
  --preregistration evidence/r33_independent_capture/formal_h20/result/preregistration/preregistration.json \
  --output evidence/r39_independent_slot_census/artifacts/clean_audit.json \
  --census-output evidence/r39_independent_slot_census/artifacts/expected_slot_census.json
```

Result: passed; 180 expected slots/capture, 2 cells, 6 captures, 1,080 rows,
96,660 relations.  Expected census semantic SHA-256:
`31d788fd9e39f2a8431edf695d732c46fffa750e27569484d199224decedf65a`.

```bash
python3 evidence/r39_independent_slot_census/scripts/run_negative_controls.py \
  --protocol evidence/r39_independent_slot_census/protocol.json \
  --input evidence/r33_independent_capture/formal_h20/result/raw/out-of-process-gdn-capture.json \
  --output evidence/r39_independent_slot_census/artifacts/negative_controls.json
```

Result: passed; 3/3 controls failed closed with exact expected codes, and 3/3
had valid recomputed internal row and relation digests.

Output SHA-256 digests:

- clean audit:
  `10de7249ab0deaac007fd694905bdc1f548e4e9098340de49fa151df5f25922c`
- derived expected census:
  `de9ef7a30cf68e8e93849d40c67bddc8bfb936705c95e6e32454b5a76705385a`
- negative controls:
  `598ff2c2fa8292b4c6f52ea1cf307280e44a923005272b635f2a20434030f2db`

Executed source SHA-256 digests:

- `audit_independent_slot_census.py`:
  `cc305e6636b2e4e3c8a05d404a1afa8caa5010daf15e5b60d0ef5a7a19ecfba6`
- `run_negative_controls.py`:
  `7fc700c1e8eb52d0b6ceceaff99bb86fafe011bbbc9145283a6708a7431a147f`
- `test_independent_slot_census.py`:
  `4757c1e0c0f833907abf3f8d56fbc710980a449e295b66d7aad2579a8722df5f`

Fresh-formal binder source SHA-256 digests:

- `generate_preexecution_census.py`:
  `415bbc86b8b9bc823b4d98ac721073a669b5dc453507d025662604bb197d9b2e`
- `aggregate_formal_run.py`:
  `50968621747ef2290b6076f8c671023ad445e7e66d5c641fe8ad547f10d68644`
- `launch_r39_h20.sh`:
  `195fa925af610d8996e2369b6a39df72f6fe8794b7947772fe746c0a7e99ecc0`
- `launch_trial_1907358.sh`:
  `af411cddc77bd56f23dbcf90f0d5818ceff2a91e6bdd3b1ab5fcc1ec31b9f159`
- `test_formal_pipeline.py`:
  `d956549ba3690a10fe34adb02d09cc146703efd30d29aa46035bb78db1989329`

Transfer package:

- package SHA-256:
  `4cb727c1a7500bc997891bef868f50b174cd455a8b2ae0bc213b78c74b2d1621`
- build receipt SHA-256:
  `9bbd3b265b06c422065a2116de45e155b53bb7a760270ab275973feee0325a3f`
