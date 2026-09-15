#!/usr/bin/env bash
# Test-time compute sur grosse infra : vLLM en parallélisme de données (un réplica par GPU) pour générer N réponses par question.
set -euo pipefail
cd "$(dirname "$0")/../.."
vllm serve Qwen/Qwen2.5-7B-Instruct --data-parallel-size 8 --port 8000 --enable-prefix-caching &
until curl -s localhost:8000/v1/models > /dev/null; do sleep 10; done
python 20-calcul-inference/run.py --config 20-calcul-inference/infra/config.yaml
# Recherche guidée par un Process Reward Model (beam search / DVTS sur les étapes) : huggingface/search-and-learn.
# Modèles de raisonnement (budget de « réflexion ») : Qwen3 (enable_thinking), DeepSeek-R1, avec max_tokens élevé.
