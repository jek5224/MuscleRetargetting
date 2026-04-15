#!/bin/bash
#SBATCH --job-name=ipc_bake
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=data/motion_cache/logs/slurm_ipc_%j.log
#SBATCH --error=data/motion_cache/logs/slurm_ipc_%j.log
#SBATCH --time=04:00:00

set -euo pipefail
cd ~/muscle_imitation_learning_study
mkdir -p data/motion_cache/logs

export MAMBA_EXE='/opt/micromamba/bin/micromamba'
export MAMBA_ROOT_PREFIX='/opt/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate muscle-bake

echo "Started: $(date)"
echo "Side: ${SIDE:-L}"
python tools/bake_surface_contact.py \
    --bvh data/motion/walk.bvh \
    --sides "${SIDE:-L}" \
    --start-frame "${START:-0}" \
    --end-frame "${END:-131}" \
    --dhat "${DHAT:-5.0}" \
    --kappa "${KAPPA:-1000}" \
    --outer-iters "${OUTER:-3}" \
    --backend taichi
echo "Finished: $(date)"
