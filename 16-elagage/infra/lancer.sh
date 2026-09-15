#!/usr/bin/env bash
# Élagage sur grosse infra.
#   A) élagage structuré + distillation (Minitron) : l'élagage prend des minutes, la distillation des jours multi-nœuds
#   B) SparseGPT / Wanda 2:4 d'un 70B couche par couche avec llm-compressor, servi par vLLM (noyaux 2:4)
set -euo pipefail
cd "$(dirname "$0")/../.."

sbatch commun/infra/slurm_multinoeud.sbatch 16-elagage commun/infra/accelerate_fsdp.yaml

# B) pip install llmcompressor
# python - <<'PY'
# from llmcompressor import oneshot
# from llmcompressor.modifiers.obcq import SparseGPTModifier
# oneshot(model="meta-llama/Llama-3.3-70B-Instruct", dataset="open_platypus",
#         recipe=SparseGPTModifier(sparsity=0.5, mask_structure="2:4", targets=["Linear"], ignore=["lm_head"]),
#         max_seq_length=2048, num_calibration_samples=512, output_dir="sorties/llama-70b-2of4")
# PY
# NVIDIA TensorRT Model Optimizer (modelopt) : élagage Minitron (profondeur/largeur/têtes) + distillation dans NeMo.
