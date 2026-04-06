#!/bin/bash
#SBATCH --job-name=ipc_p2
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --output=data/ipc_phase2_bake.log
#SBATCH --error=data/ipc_phase2_bake.log
#SBATCH --time=01:00:00

export MAMBA_EXE='/opt/micromamba/bin/micromamba'
export MAMBA_ROOT_PREFIX='/opt/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate muscle-bake
export PYTHONUNBUFFERED=1

cd ~/muscle_imitation_learning_study
python3 -u tools/bake_ipc_phase2.py "$@"
