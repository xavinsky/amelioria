#!/usr/bin/env bash
# Extension de contexte sur grosse infra.
#   A) jusqu'à ~32k : run.py + FSDP (activations d'une séquence entière sur chaque GPU)
#   B) 128k et plus : parallélisme de séquence / de contexte (axolotl + ring-flash-attn, torchtitan CP, Megatron CP)
set -euo pipefail
cd "$(dirname "$0")/../.."
accelerate launch --config_file commun/infra/accelerate_fsdp.yaml \
  12-extension-contexte/run.py --config 12-extension-contexte/infra/config.yaml "$@"
# axolotl train 12-extension-contexte/infra/axolotl_yarn_128k.yml
# Évaluation long contexte : RULER (NVIDIA/RULER), LongBench v2, needle-in-a-haystack, servies par vLLM
#   vllm serve sorties/12-extension-contexte/modele-yarn --max-model-len 32768
