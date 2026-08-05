#!/bin/bash
# Rebake run.bvh, 4 regions, skin prior ON only for quadriceps + TFL.
# Output dir suffix: _quads_tfl_skin
set -euo pipefail
cd /home/jek/muscle_imitation_learning_study
source pyMAC/bin/activate
mkdir -p data/motion_cache/logs

INCLUDE="L_Rectus_Femoris,R_Rectus_Femoris,L_Vastus_Lateralis,R_Vastus_Lateralis,L_Vastus_Medialis,R_Vastus_Medialis,L_Vastus_Intermedius,R_Vastus_Intermedius,L_Tensor_Fascia_Lata,R_Tensor_Fascia_Lata"

for REGION in L_UpLeg L_LowLeg R_UpLeg R_LowLeg; do
    LOG="data/motion_cache/logs/run_quads_tfl_${REGION}.log"
    echo "=== Baking run.bvh ${REGION} (skin prior on quads+TFL only) ===" | tee -a "$LOG"
    date | tee -a "$LOG"
    rm -rf "data/motion_cache/run/${REGION}_quads_tfl_skin"
    python tools/bake_headless.py \
        --bvh data/motion/run.bvh \
        --muscles ".muscles_${REGION}.json" \
        --region-tag "${REGION}_quads_tfl_skin" \
        --skin-prior --skin-prior-include "$INCLUDE" \
        --no-plateau-exit --settle-iters 50 \
        --backend taichi 2>&1 | tee -a "$LOG"
    echo "=== Done ${REGION} ===" | tee -a "$LOG"
    date | tee -a "$LOG"
done
echo "=== Done all run quads_tfl_skin ==="
date
