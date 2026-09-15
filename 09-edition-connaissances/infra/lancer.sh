#!/usr/bin/env bash
# Édition de connaissances à grande échelle.
#   A) run.py (ROME maison) sur un modèle 7-8B : 1 GPU 40-80 Go
#   B) MEMIT / AlphaEdit via EasyEdit : milliers de faits d'un coup
set -euo pipefail
cd "$(dirname "$0")/../.."

python 09-edition-connaissances/run.py --config 09-edition-connaissances/infra/config.yaml

python 09-edition-connaissances/infra/memit_easyedit.py --hparams 09-edition-connaissances/infra/memit_llama3-8b.yaml --n 1000
bash commun/infra/evaluer_lm_eval.sh sorties/09-edition-connaissances-memit "mmlu,hellaswag"   # vérifier que le reste n'a pas bougé
