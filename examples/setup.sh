#!/usr/bin/env bash

set -euo pipefail

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3.10 or later is required." >&2
  exit 1
fi

if ! python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
  echo "Python 3.10 or later is required." >&2
  exit 1
fi

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q

echo
echo "Setup complete. Configure an API key or start Ollama, then run:"
echo "  enumeraite doctor"
echo "  enumeraite run examples/sample_paths.txt --target example.com"
