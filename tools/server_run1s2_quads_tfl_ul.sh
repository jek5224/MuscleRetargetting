#!/bin/bash
#SBATCH --job-name=r1s2_qtfl_ul
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=data/motion_cache/logs/slurm_%j.log
#SBATCH --error=data/motion_cache/logs/slurm_%j.log

set -euo pipefail
cd ~/muscle_imitation_learning_study
mkdir -p data/motion_cache/logs

export MAMBA_EXE='/opt/micromamba/bin/micromamba'
export MAMBA_ROOT_PREFIX='/opt/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
micromamba activate muscle-bake

INCLUDE="L_Rectus_Femoris,R_Rectus_Femoris,L_Vastus_Lateralis,R_Vastus_Lateralis,L_Vastus_Medialis,R_Vastus_Medialis,L_Vastus_Intermedius,R_Vastus_Intermedius,L_Tensor_Fascia_Lata,R_Tensor_Fascia_Lata"
BVH=data/motion/run1_subject2.bvh
STEM=run1_subject2

echo "GPU: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Started: $(date)"

for REGION in L_UpLeg R_UpLeg; do
    OUTDIR=data/motion_cache/${STEM}/${REGION}_quads_tfl_skin
    rm -rf "$OUTDIR"
    echo "=== Baking ${STEM} ${REGION} (quads+TFL skin) ==="
    date
    python tools/bake_headless.py \
        --bvh "$BVH" \
        --muscles ".muscles_${REGION}.json" \
        --region-tag "${REGION}_quads_tfl_skin" \
        --skin-prior --skin-prior-include "$INCLUDE" \
        --no-plateau-exit --settle-iters 50 \
        --backend taichi
    echo "=== Done ${REGION} ==="
    date
done

echo "Finished: $(date)"
