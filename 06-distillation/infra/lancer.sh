#!/usr/bin/env bash
# Distillation sur grosse infra : deux montages.
set -euo pipefail
cd "$(dirname "$0")/../.."
MONTAGE=${1:-logits}

if [ "$MONTAGE" = "logits" ]; then
  # A) Logits 8B -> 1B : l'enseignant est répliqué sur chaque GPU (16 Go en bf16), élève en ZeRO-2.
  sed 's/zero_stage: 3/zero_stage: 2/' commun/infra/accelerate_zero3.yaml > /tmp/accelerate_zero2.yaml
  accelerate launch --config_file /tmp/accelerate_zero2.yaml \
    06-distillation/run.py --config 06-distillation/infra/config.yaml
  # Enseignant trop gros pour un GPU (70B+) : pré-calculer les top-k logits hors ligne
  # (ex. arcee-ai/DistillKit), ou passer au montage B.

elif [ "$MONTAGE" = "sequence" ]; then
  # B) Séquences : un enseignant 70B (ou DeepSeek-R1) génère avec vLLM, puis SFT de l'élève (recette des modèles « R1-Distill »).
  python 06-distillation/infra/generer_vllm.py --enseignant deepseek-ai/DeepSeek-R1-Distill-Llama-70B --tp 8 \
    --dataset openai/gsm8k --config main --colonne question --cible answer --n 8 --sortie sorties/distill-gsm8k.jsonl
  accelerate launch --config_file commun/infra/accelerate_fsdp.yaml \
    02-fine-tuning-supervise/run.py --config 02-fine-tuning-supervise/infra/config.yaml \
    --set donnees.source=hf --set donnees.nom=json --set donnees.fichiers=sorties/distill-gsm8k.jsonl \
    --set donnees.split_train=train --set "donnees.champs={question: question, reponse: reponse, cible: cible}" \
    --set modele.nom=meta-llama/Llama-3.2-1B-Instruct --set entrainement.max_length=16384
fi
