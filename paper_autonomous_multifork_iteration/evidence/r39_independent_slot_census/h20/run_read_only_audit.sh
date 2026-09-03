#!/usr/bin/env bash
set -euo pipefail

# Read-only audit of a copied R33 result on an H20 host.  This command neither
# allocates nor manages QS resources and launches no GPU kernels; it is supplied
# only so the same hash-bound CPU verifier can be rerun next to the archived
# execution environment.

PAPER_ROOT="${1:?usage: run_read_only_audit.sh PAPER_ROOT [OUTPUT_DIR]}"
OUTPUT_DIR="${2:-${PAPER_ROOT}/paper_autonomous_multifork_iteration/evidence/r39_independent_slot_census/h20_result}"
EVIDENCE_DIR="${PAPER_ROOT}/paper_autonomous_multifork_iteration/evidence/r39_independent_slot_census"
R33_DIR="${PAPER_ROOT}/paper_autonomous_multifork_iteration/evidence/r33_independent_capture/formal_h20/result"

mkdir -p "${OUTPUT_DIR}"
python3 "${EVIDENCE_DIR}/scripts/audit_independent_slot_census.py" \
  --protocol "${EVIDENCE_DIR}/protocol.json" \
  --input "${R33_DIR}/raw/out-of-process-gdn-capture.json" \
  --preregistration "${R33_DIR}/preregistration/preregistration.json" \
  --output "${OUTPUT_DIR}/clean_audit.json" \
  --census-output "${OUTPUT_DIR}/expected_slot_census.json"
python3 "${EVIDENCE_DIR}/scripts/run_negative_controls.py" \
  --protocol "${EVIDENCE_DIR}/protocol.json" \
  --input "${R33_DIR}/raw/out-of-process-gdn-capture.json" \
  --output "${OUTPUT_DIR}/negative_controls.json"
