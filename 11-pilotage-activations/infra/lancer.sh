#!/usr/bin/env bash
# Pilotage par activations sur de gros modèles.
#
# A) run.py sur un 70B, réparti sur plusieurs GPU d'un nœud (device_map="auto" : inférence seulement, pas de FSDP).
#    python 11-pilotage-activations/run.py --config 11-pilotage-activations/infra/config.yaml
#
# B) Service en production avec vecteurs de pilotage : vLLM ne l'expose pas nativement ;
#    - modèle « abliteré » (mode ablation) : c'est un checkpoint normal, servi tel quel par vLLM/TGI ;
#    - vecteurs dynamiques : bibliothèques repeng (control vectors), steering-vectors, ou llama.cpp
#      (--control-vector fichier.gguf --control-vector-scaled fichier.gguf 0.8), EasySteer (vLLM patché).
#
# C) Recherche de directions interprétables : autoencodeurs parcimonieux (SAE) pré-entraînés
#    (Gemma Scope, Llama Scope) avec SAELens ; piloter par une feature SAE plutôt qu'une moyenne de différences.
set -euo pipefail
cd "$(dirname "$0")/../.."
python 11-pilotage-activations/run.py --config 11-pilotage-activations/infra/config.yaml "$@"
