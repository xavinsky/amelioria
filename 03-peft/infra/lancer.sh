#!/usr/bin/env bash
# QLoRA 70B sur un nœud multi-GPU (FSDP + QLoRA), puis service multi-adaptateurs avec vLLM.
set -euo pipefail
cd "$(dirname "$0")/../.."

accelerate launch --config_file commun/infra/accelerate_fsdp.yaml --num_processes "${NB_GPU:-8}" \
  03-peft/run.py --config 03-peft/infra/config.yaml "$@"

# Servir le modèle de base + l'adaptateur (plusieurs adaptateurs peuvent partager la même base) :
#   vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 4 \
#     --enable-lora --max-lora-rank 64 \
#     --lora-modules arith=sorties/03-peft/adaptateur-qlora
#   puis appeler l'API OpenAI avec "model": "arith"
