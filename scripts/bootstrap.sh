#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uv_bin="${UV_BIN:-$HOME/.local/bin/uv}"

if [[ ! -x "$uv_bin" ]]; then
  echo "Installing uv into the user account..."
  installer="$(mktemp)"
  trap 'rm -f "$installer"' EXIT
  curl --fail --location --proto '=https' --tlsv1.2 \
    https://astral.sh/uv/install.sh --output "$installer"
  sh "$installer"
fi

"$uv_bin" python install 3.12
"$uv_bin" venv --python 3.12 "$repo_root/.venv"
"$uv_bin" pip install --python "$repo_root/.venv/bin/python" -e "$repo_root"

echo "Environment ready. Activate with:"
echo "  source $repo_root/.venv/bin/activate"
