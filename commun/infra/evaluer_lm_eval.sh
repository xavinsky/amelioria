#!/usr/bin/env bash
# Évaluation d'un modèle produit par n'importe quelle méthode, sur benchmarks standards,
# avec lm-evaluation-harness + vLLM (multi-GPU). À utiliser sur grosse infra à la place de
# l'évaluation intégrée aux run.py (désactivée dans les infra/config.yaml).
#
#   bash commun/infra/evaluer_lm_eval.sh sorties/02-fine-tuning-supervise/modele [taches]
#
# pip install "lm_eval[vllm]"
set -euo pipefail
MODELE=${1:?chemin ou nom du modèle}
TACHES=${2:-gsm8k,mmlu,arc_challenge,hellaswag,ifeval}
TP=${TP:-$(nvidia-smi -L | wc -l)}

lm_eval --model vllm \
  --model_args "pretrained=$MODELE,tensor_parallel_size=$TP,dtype=bfloat16,gpu_memory_utilization=0.85" \
  --tasks "$TACHES" \
  --batch_size auto \
  --apply_chat_template \
  --output_path "$MODELE/lm_eval"
