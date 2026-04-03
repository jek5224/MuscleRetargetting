#!/bin/bash
#SBATCH --job-name=pyuipc_bake
#SBATCH --output=pyuipc_bake_%j.log
#SBATCH --error=pyuipc_bake_%j.log
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

export MAMBA_EXE='/opt/micromamba/bin/micromamba'
export MAMBA_ROOT_PREFIX='/opt/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate muscle-bake

cd ~/muscle_imitation_learning_study

python tools/bake_pyuipc.py "$@"
