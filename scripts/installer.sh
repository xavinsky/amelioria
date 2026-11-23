#!/usr/bin/env bash
# Crée le venv du dépôt et installe le socle (+ extras optionnels).
#   bash scripts/installer.sh                    # socle
#   bash scripts/installer.sh --extras           # socle + bitsandbytes, torchao, diffusers, torchcodec
#   bash scripts/installer.sh --vllm             # venv séparé .venv-vllm (génération rapide de 07)
#   bash scripts/installer.sh --llmcompressor    # venv séparé .venv-llmcompressor (15 : gptq, awq)
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 bash scripts/installer.sh --extras
#     torch et torchvision depuis l'index PyTorch de la version CUDA du poste (versions accordées)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "${1:-}" = "--llmcompressor" ]; then
  # llm-compressor impose ses propres versions de torch et de transformers : environnement dédié.
  python3 -m venv .venv-llmcompressor
  .venv-llmcompressor/bin/pip install -U pip
  .venv-llmcompressor/bin/pip install llmcompressor datasets pyyaml
  exit 0
fi

if [ "${1:-}" = "--vllm" ]; then
  # vLLM impose ses propres versions de torch : environnement dédié, utilisé par `generation.vllm.python` (07).
  python3 -m venv .venv-vllm
  .venv-vllm/bin/pip install -U pip
  .venv-vllm/bin/pip install vllm
  exit 0
fi

python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
if [ -n "${TORCH_INDEX_URL:-}" ]; then
  pip install torch torchvision --index-url "$TORCH_INDEX_URL"
fi
pip install -r requirements.txt
if [ "${1:-}" = "--extras" ]; then
  # deepspeed et llmcompressor sont exclus : grosse infra et environnement séparé.
  grep -v '^#' requirements-extras.txt | sed 's/#.*//' | grep -v -E 'llmcompressor|deepspeed' | xargs pip install
fi
