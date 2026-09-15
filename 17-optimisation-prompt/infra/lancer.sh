#!/usr/bin/env bash
# Optimisation de prompt sur grosse infra : le modèle est servi (vLLM tensor parallel), les scripts l'appellent en parallèle.
set -euo pipefail
cd "$(dirname "$0")/../.."
vllm serve Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 4 --port 8000 &
until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done
python 17-optimisation-prompt/run.py --config 17-optimisation-prompt/infra/config.yaml
python 17-optimisation-prompt/infra/dspy_miprov2.py --modele openai/Qwen/Qwen2.5-72B-Instruct --api-base http://localhost:8000/v1
