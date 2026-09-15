#!/usr/bin/env bash
# Croissance d'architecture sur grosse infra.
#   A) transformation (quelques minutes, CPU suffit) : run.py ou mergekit (passthrough, moe)
#   B) cicatrisation : pré-entraînement continu multi-nœuds, c'est là qu'est le vrai coût (milliards de tokens)
set -euo pipefail
cd "$(dirname "$0")/../.."
sbatch commun/infra/slurm_multinoeud.sbatch 14-croissance-architecture commun/infra/accelerate_fsdp.yaml
# Variante sans code : mergekit-yaml 14-croissance-architecture/infra/mergekit_passthrough_solar.yml sorties/mistral-10.7b --lazy-unpickle
#   puis 01-pre-entrainement-continu/infra avec --set modele.nom=sorties/mistral-10.7b
# MoE entraîné à grande échelle : Megatron-LM / NeMo (upcycling natif, parallélisme d'experts), ou torchtitan.
