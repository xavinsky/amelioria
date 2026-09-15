#!/usr/bin/env bash
# SFT complet sur grosse infra.
#   1 nœud 8 GPU : bash 02-fine-tuning-supervise/infra/lancer.sh
#   Slurm        : sbatch commun/infra/slurm_multinoeud.sbatch 02-fine-tuning-supervise commun/infra/accelerate_fsdp.yaml
#   Alternative prête à l'emploi (même recette) : trl sft --config ... / axolotl / LLaMA-Factory (voir README)
set -euo pipefail
cd "$(dirname "$0")/../.."
accelerate launch --config_file commun/infra/accelerate_fsdp.yaml \
  02-fine-tuning-supervise/run.py --config 02-fine-tuning-supervise/infra/config.yaml "$@"
bash commun/infra/evaluer_lm_eval.sh sorties/02-fine-tuning-supervise/modele gsm8k,ifeval,mmlu
