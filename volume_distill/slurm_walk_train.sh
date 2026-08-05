#!/bin/bash
#SBATCH --job-name=walk_v1dec
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=volume_distill/walk_v1dec_train.log
#SBATCH --error=volume_distill/walk_v1dec_train.log

export MAMBA_EXE='/opt/micromamba/bin/micromamba'
export MAMBA_ROOT_PREFIX='/opt/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate muscle-bake

export PYTHONUNBUFFERED=1
cd ~/muscle_imitation_learning_study
stdbuf -oL python3 -u -m volume_distill.walk_overfit_v1dec
