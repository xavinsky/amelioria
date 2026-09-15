#!/usr/bin/env bash
# Désapprentissage à grande échelle avec open-unlearning (benchmarks TOFU, MUSE, WMDP ; méthodes GA, GradDiff, NPO, SimNPO, RMU...).
#   git clone https://github.com/locuslab/open-unlearning && cd open-unlearning && pip install -e ".[lm_eval]"
set -euo pipefail

# TOFU forget10 avec NPO sur Llama-3.1-8B, multi-GPU (DeepSpeed ZeRO-3 via accelerate)
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate/default_config.yaml \
  src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
  model=Llama-3.1-8B-Instruct trainer=NPO \
  forget_split=forget10 retain_split=retain90 \
  model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.1-8B-Instruct_full \
  task_name=tofu_npo_forget10

# Évaluation TOFU (forget quality, model utility) et MMLU pour la non-régression
python src/eval.py experiment=eval/tofu/default forget_split=forget10 \
  model=Llama-3.1-8B-Instruct model.model_args.pretrained_model_name_or_path=saves/unlearn/tofu_npo_forget10 \
  task_name=tofu_npo_forget10_eval

# Désapprentissage de connaissances dangereuses (WMDP, biosécurité/cybersécurité) : trainer=RMU sur Zephyr-7B / Llama-3-8B.
