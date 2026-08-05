#!/bin/bash
set -euo pipefail
cd /home/jek/muscle_imitation_learning_study
source pyMAC/bin/activate
mkdir -p data/motion_cache/logs

for REGION in L_UpLeg L_LowLeg R_UpLeg R_LowLeg; do
    LOG="data/motion_cache/logs/run1_local_${REGION}.log"
    echo "=== Starting ${REGION} ===" | tee -a "$LOG"
    date | tee -a "$LOG"
    python tools/bake_headless.py \
        --bvh data/motion/run1_subject2.bvh \
        --muscles .muscles_${REGION}.json \
        --region-tag ${REGION} \
        --skin-prior --no-plateau-exit --settle-iters 50 \
        --skin-prior-exclude L_Popliteus,R_Popliteus \
        --backend taichi 2>&1 | tee -a "$LOG"
    echo "=== Done ${REGION} ===" | tee -a "$LOG"
    date | tee -a "$LOG"
done
