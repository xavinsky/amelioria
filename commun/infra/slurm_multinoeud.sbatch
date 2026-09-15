#!/bin/bash
# Lancement multi-nœuds générique sur un cluster Slurm (ex. 2 nœuds x 8 H100/B200).
#
#   sbatch commun/infra/slurm_multinoeud.sbatch <dossier-methode> <config-accelerate> [arguments run.py...]
#   sbatch commun/infra/slurm_multinoeud.sbatch 02-fine-tuning-supervise commun/infra/accelerate_fsdp.yaml
#
# Le script de la méthode reçoit toujours `--config <dossier-methode>/infra/config.yaml`.
#SBATCH --job-name=ameliorer-modele
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
METHODE=$1
CONFIG_ACCELERATE=$2
shift 2

GPUS_PAR_NOEUD=${SLURM_GPUS_PER_NODE:-8}
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR MASTER_PORT=29500
export NCCL_ASYNC_ERROR_HANDLING=1 HF_HUB_ENABLE_HF_TRANSFER=1

srun --kill-on-bad-exit=1 bash -c "
  accelerate launch \
    --config_file $CONFIG_ACCELERATE \
    --num_machines $SLURM_NNODES \
    --num_processes $((SLURM_NNODES * GPUS_PAR_NOEUD)) \
    --machine_rank \$SLURM_PROCID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    $METHODE/run.py --config $METHODE/infra/config.yaml $*
"
