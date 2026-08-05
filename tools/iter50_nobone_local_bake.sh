#!/bin/bash
# walk.bvh + run.bvh, 4 regions, NO skin prior, NO bone contact, iter50.
set -euo pipefail
cd /home/jek/muscle_imitation_learning_study
source pyMAC/bin/activate
mkdir -p data/motion_cache/logs

for BVH in walk.bvh run.bvh; do
    STEM="${BVH%.bvh}"
    for REGION in L_UpLeg L_LowLeg R_UpLeg R_LowLeg; do
        LOG="data/motion_cache/logs/${STEM}_iter50_nobone_${REGION}.log"
        echo "=== Baking ${BVH} ${REGION} (iter50, no skin, no bone) ===" | tee -a "$LOG"
        date | tee -a "$LOG"
        rm -rf "data/motion_cache/${STEM}/${REGION}_iter50_nobone"
        python tools/bake_headless.py \
            --bvh "data/motion/${BVH}" \
            --muscles ".muscles_${REGION}.json" \
            --region-tag "${REGION}_iter50_nobone" \
            --no-plateau-exit --settle-iters 50 \
            --backend taichi 2>&1 | tee -a "$LOG"
        echo "=== Done ${BVH} ${REGION} ===" | tee -a "$LOG"
        date | tee -a "$LOG"
    done
done
echo "=== Done all iter50_nobone ==="
date
