#!/usr/bin/env bash
# Extension de vocabulaire pour une langue sur grosse infra.
#   1) tokenizer : BPE appris sur la langue cible puis fusionné (entrainer_tokenizer.py), ou added_tokens (run.py)
#   2) phase 1 (embeddings seuls) + phase 2 (pré-entraînement continu complet, voir aussi 01-pre-entrainement-continu/infra)
set -euo pipefail
cd "$(dirname "$0")/../.."
accelerate launch --config_file commun/infra/accelerate_fsdp.yaml \
  13-extension-vocabulaire/run.py --config 13-extension-vocabulaire/infra/config.yaml "$@"
