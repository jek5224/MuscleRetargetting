#!/bin/bash
# Move large data dirs from /home repo to external storage, leave symlinks.
# motion_cache is already symlinked.  This script picks the next-biggest
# offenders and migrates them under the same external parent.
set -euo pipefail

EXT="/media/jek/68CC202CCC1FF2D4/JEK/volume_muscle_sim"
REPO="/home/jek/muscle_imitation_learning_study"

if [ ! -d "$EXT" ]; then
    echo "External path missing: $EXT" >&2
    exit 1
fi
cd "$REPO"

# Big writable / archive dirs to move.  KEEP locally: code, Zygote_Meshes_251229
# (active asset), tet/ (active), pyMAC, dart (binaries with RPATH).
ENTRIES=(
    "volume_distill/dance_v1dec_checkpoints"
    "volume_distill/walk_v1dec_v4_checkpoints"
    "volume_distill/walk_v1dec_v3_checkpoints"
    "volume_distill/walk_v1dec_v2_checkpoints"
    "volume_distill/walk_mirror_checkpoints"
    "volume_distill/dance"
    "volume_distill/dof_grid_checkpoints"
    "volume_distill/dance_checkpoints"
    "volume_distill/walk_checkpoints"
    "volume_distill/checkpoints"
    "volume_distill/runs"
    "volume_distill/dof_grid_runs"
    "volume_distill/walk_runs"
    "volume_distill/walk_mirror_runs"
    "volume_distill/dance_v1dec_runs"
    "volume_distill/dance_runs"
    "volume_distill/walk_v1dec_runs"
    "volume_distill/walk_v1dec_v2_runs"
    "volume_distill/walk_v1dec_v3_runs"
    "volume_distill/walk_v1dec_v4_runs"
    "ray_results"
    "tet_subdiv"
    "tet_hybrid"
    "tet_fine"
    "tet_old"
    "tet_orig_std"
    "Zygote_Meshes"
    "Zygote_Meshes_Subdivided"
    "Zygote_Meshes_Revised"
    "Zygote_Meshes_Revised_Subdivided"
    "Zygote_Meshes_Original"
    "analysis_images"
    "reverse_lbs_results"
    "bake_pyuipc"
    "bp_viz"
    "Result_Images"
)

for entry in "${ENTRIES[@]}"; do
    if [ -L "$entry" ]; then
        echo "[skip] $entry already a symlink"
        continue
    fi
    if [ ! -e "$entry" ]; then
        echo "[skip] $entry missing"
        continue
    fi
    target="$EXT/$entry"
    parent_target=$(dirname "$target")
    mkdir -p "$parent_target"
    if [ -e "$target" ]; then
        echo "[skip] $target already exists at destination — manual review"
        continue
    fi
    echo "[move] $entry -> $target"
    # rsync first, verify, then nuke source
    rsync -a "$entry/" "$target/"
    if [ $? -ne 0 ]; then
        echo "[error] rsync failed for $entry — leaving in place"
        continue
    fi
    rm -rf "$entry"
    ln -s "$target" "$entry"
done

echo
echo "Done. df -h /home:"
df -h /home
echo
echo "Current symlinks under repo:"
find . -maxdepth 2 -type l -ls 2>/dev/null | head -50
