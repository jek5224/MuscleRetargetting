#!/bin/bash
# Re-run moment-arm analysis on R muscles using iter50_nobone baked cache.
set -e
cd /home/jek/muscle_imitation_learning_study
source pyMAC/bin/activate

UPLEG=(R_Adductor_Brevis R_Adductor_Longus R_Adductor_Magnus R_Biceps_Femoris
       R_Gluteus_Maximus R_Gluteus_Medius R_Gluteus_Minimus R_Gracilis
       R_Iliacus R_Inferior_Gemellus R_Obturator_Externus R_Obturator_Internus
       R_Pectineus R_Piriformis R_Quadratus_Femoris R_Rectus_Femoris
       R_Sartorius R_Semimembranosus R_Semitendinosus R_Superior_Gemellus
       R_Tensor_Fascia_Lata R_Vastus_Intermedius R_Vastus_Lateralis
       R_Vastus_Medialis R_Popliteus)

LOWLEG=(R_Extensor_Digitorum_Longus R_Extensor_Hallucis_Longus
        R_Flexor_Digitorum_Longus R_Flexor_Hallucis R_Gastrocnemius
        R_Peroneus_Brevis R_Peroneus_Longus R_Peroneus_Tertius
        R_Plantaris R_Soleus R_Tibialis_Anterior R_Tibialis_Posterior)

for M in "${UPLEG[@]}"; do
    python3 tools/fiber_moment_arm_analyze.py --muscle "$M" \
        --cache-dir data/motion_cache/walk/R_UpLeg_iter50_nobone --clusters 8 \
        2>&1 | grep -E "^STATS|Saved" || true
done
for M in "${LOWLEG[@]}"; do
    python3 tools/fiber_moment_arm_analyze.py --muscle "$M" \
        --cache-dir data/motion_cache/walk/R_LowLeg_iter50_nobone --clusters 8 \
        2>&1 | grep -E "^STATS|Saved" || true
done
