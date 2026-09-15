#!/usr/bin/env bash
# DPO complet sur grosse infra (DeepSpeed ZeRO-3 : politique + modèle de référence découpés).
#   Slurm : sbatch commun/infra/slurm_multinoeud.sbatch 04-optimisation-preferences commun/infra/accelerate_zero3.yaml
# Alternatives : open-instruct (allenai), OpenRLHF (openrlhf.cli.train_dpo), LLaMA-Factory (stage: dpo).
set -euo pipefail
cd "$(dirname "$0")/../.."
accelerate launch --config_file commun/infra/accelerate_zero3.yaml \
  04-optimisation-preferences/run.py --config 04-optimisation-preferences/infra/config.yaml "$@"
bash commun/infra/evaluer_lm_eval.sh sorties/04-optimisation-preferences/modele-dpo_norm "ifeval,gsm8k,mmlu"
# Pour les préférences « de chat », compléter par AlpacaEval 2 / Arena-Hard (juge LLM).
