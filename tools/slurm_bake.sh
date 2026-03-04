#!/bin/bash
#SBATCH --job-name=bake
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=data/motion_cache/logs/slurm_%A_%a.log
#SBATCH --error=data/motion_cache/logs/slurm_%A_%a.log

# ── Usage ──────────────────────────────────────────────────────────────
#   cd ~/muscle_imitation_learning_study
#   python tools/slurm_gen_tasks.py          # generates task list
#   N=$(wc -l < data/motion_cache/task_list.txt)
#   sbatch --array=1-${N}%5 tools/slurm_bake.sh
#
#   # %5 = max 5 concurrent tasks (one per GPU)
#   # Monitor:  squeue -u $USER
#   # Cancel:   scancel <jobid>
#   # After:    python tools/slurm_merge.py
# ───────────────────────────────────────────────────────────────────────

set -euo pipefail

cd ~/muscle_imitation_learning_study
mkdir -p data/motion_cache/logs

# Activate micromamba
export MAMBA_EXE='/opt/micromamba/bin/micromamba'
export MAMBA_ROOT_PREFIX='/opt/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate muscle-bake

# Each line in task_list.txt: <region_tag> <muscles_json> <bvh_path>
TASK_LIST="data/motion_cache/task_list.txt"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_LIST")

if [ -z "$LINE" ]; then
    echo "ERROR: No task at line ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

REGION=$(echo "$LINE" | awk '{print $1}')
MUSCLES=$(echo "$LINE" | awk '{print $2}')
BVH_FILE=$(echo "$LINE" | awk '{print $3}')
STEM=$(basename "$BVH_FILE" .bvh)
DONE_MARKER="data/motion_cache/${STEM}/${REGION}/.done"

# Skip if already baked
if [ -f "$DONE_MARKER" ]; then
    echo "SKIP: ${STEM}/${REGION} already baked"
    exit 0
fi

echo "=== Baking: ${STEM} / ${REGION} ==="
echo "Task ID: ${SLURM_ARRAY_TASK_ID}, Job ID: ${SLURM_JOB_ID}"
echo "BVH: ${BVH_FILE}"
echo "Muscles: ${MUSCLES}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Started: $(date)"

python tools/bake_headless.py \
    --bvh "$BVH_FILE" \
    --muscles "$MUSCLES" \
    --region-tag "$REGION" \
    --backend taichi

echo "Finished: $(date)"
