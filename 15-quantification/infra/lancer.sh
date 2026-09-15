#!/usr/bin/env bash
# Quantification de gros modèles : calibration sur 1-2 GPU, service multi-GPU, évaluation de la dégradation.
set -euo pipefail
cd "$(dirname "$0")/../.."
python 15-quantification/infra/gptq_70b_llmcompressor.py --modele meta-llama/Llama-3.3-70B-Instruct --schema W4A16
bash commun/infra/evaluer_lm_eval.sh sorties/Llama-3.3-70B-Instruct-W4A16 "gsm8k,mmlu,ifeval"
bash commun/infra/evaluer_lm_eval.sh meta-llama/Llama-3.3-70B-Instruct "gsm8k,mmlu,ifeval"   # référence bf16 pour comparer
# Cache KV quantifié au service (contextes longs) : vllm serve ... --kv-cache-dtype fp8
