#!/usr/bin/env bash
# Boucle STaR / ReST-EM à grande échelle : générer (vLLM) -> filtrer (vérificateur) -> SFT (FSDP) -> recommencer.
#   bash 07-donnees-synthetiques/infra/lancer.sh [nb_tours]
set -euo pipefail
cd "$(dirname "$0")/../.."
TOURS=${1:-3}
MODELE=${MODELE:-Qwen/Qwen2.5-7B-Instruct}

for tour in $(seq 1 "$TOURS"); do
  # 1) Génération : k=16 solutions par question, filtrées par la réponse de référence (rejection sampling).
  python 06-distillation/infra/generer_vllm.py --enseignant "$MODELE" --tp 8 \
    --dataset openai/gsm8k --config main --colonne question --cible answer --n 16 \
    --sortie "sorties/star-tour-$tour.jsonl"

  # 2) SFT : ReST-EM repart du modèle initial à chaque tour (évite l'accumulation de biais) ; STaR repart du précédent.
  accelerate launch --config_file commun/infra/accelerate_fsdp.yaml \
    02-fine-tuning-supervise/run.py --config 02-fine-tuning-supervise/infra/config.yaml \
    --sortie "sorties/star-tour-$tour" \
    --set modele.nom=Qwen/Qwen2.5-7B-Instruct \
    --set donnees.source=hf --set donnees.nom=json --set donnees.fichiers="sorties/star-tour-$tour.jsonl" \
    --set donnees.split_train=train --set "donnees.champs={question: question, reponse: reponse, cible: cible}"

  MODELE="sorties/star-tour-$tour/modele"
  bash commun/infra/evaluer_lm_eval.sh "$MODELE" gsm8k
done

# Outils de génération de données à grande échelle (pipelines, juges, déduplication) :
#   distilabel : https://distilabel.argilla.io  (Self-Instruct, Evol-Instruct, UltraFeedback, juges LLM)
#   NeMo Curator / NeMo-Skills (NVIDIA) : génération + filtrage de solutions mathématiques à l'échelle de millions
