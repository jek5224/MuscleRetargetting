#!/bin/bash
# Bake walk.bvh + run.bvh on local PC with NO skin prior, 4 regions each.
# Output goes to data/motion_cache/{stem}/{REGION}_noskin/ to preserve
# existing skin-prior caches in the canonical {REGION}/ subdirs.
set -euo pipefail
cd /home/jek/muscle_imitation_learning_study
source pyMAC/bin/activate
mkdir -p data/motion_cache/logs

for BVH in walk.bvh run.bvh; do
    STEM="${BVH%.bvh}"
    for REGION in L_UpLeg L_LowLeg R_UpLeg R_LowLeg; do
        LOG="data/motion_cache/logs/${STEM}_noskin_${REGION}.log"
        echo "=== Baking ${BVH} ${REGION} (no skin prior) ===" | tee -a "$LOG"
        date | tee -a "$LOG"
        rm -rf "data/motion_cache/${STEM}/${REGION}_noskin"
        python tools/bake_headless.py \
            --bvh "data/motion/${BVH}" \
            --muscles ".muscles_${REGION}.json" \
            --region-tag "${REGION}_noskin" \
            --no-plateau-exit --settle-iters 50 \
            --backend taichi 2>&1 | tee -a "$LOG"
        echo "=== Done ${BVH} ${REGION} ===" | tee -a "$LOG"
        date | tee -a "$LOG"
    done
done
echo "=== All noskin bakes done ==="
date
