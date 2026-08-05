#!/bin/bash
# walk.bvh + run.bvh, 4 regions each, NO skin prior, --settle-iters 150.
# Output: data/motion_cache/{stem}/{REGION}_iter150/
set -euo pipefail
cd /home/jek/muscle_imitation_learning_study
source pyMAC/bin/activate
mkdir -p data/motion_cache/logs

for BVH in walk.bvh run.bvh; do
    STEM="${BVH%.bvh}"
    for REGION in L_UpLeg L_LowLeg R_UpLeg R_LowLeg; do
        LOG="data/motion_cache/logs/${STEM}_iter150_${REGION}.log"
        echo "=== Baking ${BVH} ${REGION} (iter150, no skin) ===" | tee -a "$LOG"
        date | tee -a "$LOG"
        rm -rf "data/motion_cache/${STEM}/${REGION}_iter150"
        python tools/bake_headless.py \
            --bvh "data/motion/${BVH}" \
            --muscles ".muscles_${REGION}.json" \
            --region-tag "${REGION}_iter150" \
            --no-plateau-exit --settle-iters 150 \
            --backend taichi 2>&1 | tee -a "$LOG"
        echo "=== Done ${BVH} ${REGION} ===" | tee -a "$LOG"
        date | tee -a "$LOG"
    done
done
echo "=== All iter150 bakes done ==="
date
