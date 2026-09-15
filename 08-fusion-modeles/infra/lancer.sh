#!/usr/bin/env bash
# Fusion de gros modèles avec mergekit.
# La fusion elle-même ne demande PAS de GPU : --lazy-unpickle charge les tenseurs un par un,
# un 70B se fusionne avec ~32 Go de RAM et ~150 Go de disque. La grosse infra sert à ÉVALUER les candidats.
set -euo pipefail
cd "$(dirname "$0")/../.."
pip install -q mergekit

# 1) Fusion TIES 70B
mergekit-yaml 08-fusion-modeles/infra/ties_70b.yml sorties/fusion-ties-70b --lazy-unpickle --out-shard-size 5B

# 2) Évaluation multi-GPU (vLLM tensor parallel)
bash commun/infra/evaluer_lm_eval.sh sorties/fusion-ties-70b "ifeval,gsm8k,mmlu,arc_challenge"

# 3) Fusion évolutionnaire (Sakana AI) : cherche automatiquement poids et densités qui maximisent un score.
#    Là, les GPU sont indispensables : chaque candidat est fusionné puis évalué.
#    pip install "mergekit[evolve]" ; mergekit-evolve --storage-path /scratch/evolve 08-fusion-modeles/infra/evolve.yml
