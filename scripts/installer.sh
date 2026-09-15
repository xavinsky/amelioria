#!/usr/bin/env bash
# Crée le venv du dépôt et installe le socle (+ extras optionnels).
#   bash scripts/installer.sh            # socle
#   bash scripts/installer.sh --extras   # socle + bitsandbytes, torchao, diffusers...
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
if [ "${1:-}" = "--extras" ]; then
  grep -v '^#' requirements-extras.txt | sed 's/#.*//' | grep -v llmcompressor | xargs pip install
fi
