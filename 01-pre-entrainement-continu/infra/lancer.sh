#!/usr/bin/env bash
# Pré-entraînement continu sur grosse infra.
#   1 nœud  : bash 01-pre-entrainement-continu/infra/lancer.sh
#   Slurm   : sbatch commun/infra/slurm_multinoeud.sbatch 01-pre-entrainement-continu commun/infra/accelerate_fsdp.yaml
set -euo pipefail
cd "$(dirname "$0")/../.."
accelerate launch --config_file commun/infra/accelerate_fsdp.yaml \
  01-pre-entrainement-continu/run.py --config 01-pre-entrainement-continu/infra/config.yaml "$@"
bash commun/infra/evaluer_lm_eval.sh sorties/01-pre-entrainement-continu/modele "arc_challenge,hellaswag,belebele_fra_Latn,m_mmlu_fr"
