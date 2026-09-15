#!/usr/bin/env bash
# RL en ligne sur grosse infra.
#
# A) TRL + vLLM colocalisé (1 nœud, 8 GPU) : génération et entraînement partagent les GPU.
#    bash 05-rl-en-ligne/infra/lancer.sh
#
# B) TRL + serveur vLLM séparé (modèles 32B+) : 2 GPU dédiés à la génération, 6 à l'entraînement.
#    CUDA_VISIBLE_DEVICES=0,1 trl vllm-serve --model Qwen/Qwen2.5-32B-Instruct --tensor-parallel-size 2 &
#    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --config_file commun/infra/accelerate_zero3.yaml --num_processes 6 \
#      05-rl-en-ligne/run.py --config 05-rl-en-ligne/infra/config.yaml \
#      --set entrainement.autres.vllm_mode=server
#
# C) Frameworks RL dédiés, multi-nœuds, PPO avec critique, 70B+ : verl (Ray + FSDP/Megatron + vLLM/SGLang) ou OpenRLHF.
#    Exemple verl (GRPO, GSM8K, Qwen2.5-7B), après `pip install verl` et préparation parquet (examples/data_preprocess/gsm8k.py) :
#    python3 -m verl.trainer.main_ppo algorithm.adv_estimator=grpo \
#      data.train_files=$HOME/data/gsm8k/train.parquet data.val_files=$HOME/data/gsm8k/test.parquet \
#      data.train_batch_size=1024 data.max_response_length=2048 \
#      actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct actor_rollout_ref.actor.optim.lr=1e-6 \
#      actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 \
#      actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=16 actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
#      trainer.n_gpus_per_node=8 trainer.nnodes=2
set -euo pipefail
cd "$(dirname "$0")/../.."
accelerate launch --config_file commun/infra/accelerate_zero3.yaml \
  05-rl-en-ligne/run.py --config 05-rl-en-ligne/infra/config.yaml "$@"
bash commun/infra/evaluer_lm_eval.sh sorties/05-rl-en-ligne/modele-dapo "gsm8k,minerva_math"
