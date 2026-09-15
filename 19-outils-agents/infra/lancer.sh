#!/usr/bin/env bash
# Agents sur grosse infra : modèle servi avec parsing d'outils natif, puis boucle agentique cliente.
set -euo pipefail
cd "$(dirname "$0")/../.."
vllm serve Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 4 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser hermes &
until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done
python 19-outils-agents/run.py --config 19-outils-agents/infra/config.yaml
# Parsers selon la famille : hermes (Qwen2.5, Hermes), llama3_json (Llama 3.1+), mistral, qwen3_coder, deepseek_v3...
# Entraîner un modèle à mieux utiliser les outils : SFT sur traces d'appels (02/03), puis RL multi-tours
# avec récompense de réussite (05 : GRPOTrainer(tools=..., environment_factory=...), verl, SkyRL).
