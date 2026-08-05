#!/bin/bash
#SBATCH --job-name=orig_bake
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=data/motion_cache/logs/slurm_orig_%j.log
#SBATCH --error=data/motion_cache/logs/slurm_orig_%j.log
#SBATCH --time=02:00:00

set -euo pipefail
cd ~/muscle_imitation_learning_study
mkdir -p data/motion_cache/logs

export MAMBA_EXE='/opt/micromamba/bin/micromamba'
export MAMBA_ROOT_PREFIX='/opt/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate muscle-bake

echo "Started: $(date)"
python tools/bake_original_mesh.py --bvh data/motion/walk.bvh --sides L --start-frame 0 --end-frame 5 --backend taichi
echo "Finished: $(date)"
