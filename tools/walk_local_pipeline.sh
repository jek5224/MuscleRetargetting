#!/bin/bash
# Full pipeline: bake walk.bvh 4 regions locally + reverse-LBS solve + XML inject.
set -euo pipefail
cd /home/jek/muscle_imitation_learning_study
source pyMAC/bin/activate
mkdir -p data/motion_cache/logs

# Clear old walk cache subdirs (keep flat-level chunks from prior bakes)
rm -rf data/motion_cache/walk/{L_UpLeg,L_LowLeg,R_UpLeg,R_LowLeg}

for REGION in L_UpLeg L_LowLeg R_UpLeg R_LowLeg; do
    LOG="data/motion_cache/logs/walk_local_${REGION}.log"
    echo "=== Baking walk.bvh ${REGION} ===" | tee -a "$LOG"
    date | tee -a "$LOG"
    python tools/bake_headless.py \
        --bvh data/motion/walk.bvh \
        --muscles .muscles_${REGION}.json \
        --region-tag ${REGION} \
        --skin-prior --no-plateau-exit --settle-iters 50 \
        --skin-prior-exclude L_Popliteus,R_Popliteus \
        --backend taichi 2>&1 | tee -a "$LOG"
    echo "=== Done ${REGION} ===" | tee -a "$LOG"
    date | tee -a "$LOG"
done

echo "=== Reverse-LBS solve ==="
date
python tools/reverse_lbs_solve.py 2>&1 | tee data/motion_cache/logs/walk_local_reverse_lbs.log | tail -5
date

echo "=== Re-emit zygote_muscle_revised.xml + LBS attrs ==="
python tools/revise_muscle_xml.py
python tools/inject_lbs_into_xml.py
echo "=== Re-emit zygote_muscle_reverse.xml ==="
python tools/write_reverse_lbs_xml.py
date
echo "=== All done ==="
