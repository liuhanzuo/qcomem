#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
required_files=(
  README.md
  baseline/MANIFEST.sha256
  baseline/paper.tex
  manuscript/paper.tex
  state/paper_state.json
  state/assumptions.md
  state/decision_log.md
  state/score_trajectory.json
  state/iteration_plan.md
  evidence/repository_inventory.md
  evidence/claim_evidence_map.tsv
  evidence/experiment_registry.json
  evidence/method_provenance.tsv
  literature/citation_requests.json
  literature/citation_lock.json
  review/issue_ledger.json
  review/best_checkpoint.json
  review/round_00_initial_screen.md
  build/build_record.json
  experiments/decisive_experiment_protocol.md
)
json_files=(
  state/paper_state.json
  state/score_trajectory.json
  evidence/experiment_registry.json
  literature/citation_requests.json
  literature/citation_lock.json
  review/issue_ledger.json
  review/best_checkpoint.json
  build/build_record.json
)

failed=0
for project_dir in "$workspace_dir"/a11-correlated-majority-vote \
                   "$workspace_dir"/a2-erase-late-absorb-early \
                   "$workspace_dir"/a2-subgroup-mix-ranking; do
  project_name="$(basename "$project_dir")"
  echo "[$project_name]"

  for relpath in "${required_files[@]}"; do
    if [[ ! -s "$project_dir/$relpath" ]]; then
      echo "  missing-or-empty: $relpath"
      failed=1
    fi
  done

  for relpath in "${json_files[@]}"; do
    if [[ -s "$project_dir/$relpath" ]] && ! jq empty "$project_dir/$relpath"; then
      echo "  invalid-json: $relpath"
      failed=1
    fi
  done

  for relpath in evidence/claim_evidence_map.tsv evidence/method_provenance.tsv; do
    if [[ -s "$project_dir/$relpath" ]] && [[ "$(wc -l < "$project_dir/$relpath")" -lt 2 ]]; then
      echo "  header-only-tsv: $relpath"
      failed=1
    fi
  done

  if [[ -s "$project_dir/baseline/MANIFEST.sha256" ]]; then
    if (cd "$project_dir/baseline" && shasum -a 256 -c MANIFEST.sha256 >/dev/null); then
      echo "  baseline-manifest: ok"
    else
      echo "  baseline-manifest: failed"
      failed=1
    fi
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "workspace validation failed"
  exit 1
fi

echo "workspace validation passed"
