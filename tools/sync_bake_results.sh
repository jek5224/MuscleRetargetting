#!/bin/bash
# Polls the server for completed region bakes, downloads them, and removes
# from the server to free disk space.
#
# Usage (run on LOCAL machine):
#   bash tools/sync_bake_results.sh          # one-shot
#   bash tools/sync_bake_results.sh --loop   # poll every 60s until all done

set -euo pipefail
cd "$(dirname "$0")/.."

SSH="ssh -p 7777 -i ~/.ssh/id_ed25519_a6000 jek5224@a6000"
RSYNC_SSH="ssh -p 7777 -i ~/.ssh/id_ed25519_a6000"
REMOTE_BASE="~/muscle_imitation_learning_study/data/motion_cache"
LOCAL_BASE="data/motion_cache"
REGIONS=(L_UpLeg R_UpLeg L_LowLeg R_LowLeg)

sync_once() {
    # Find all .done markers on server
    local found=0
    local synced=0

    for done_path in $($SSH "find $REMOTE_BASE -name .done -path '*/*/' 2>/dev/null" || true); do
        # done_path like: ~/muscle_imitation_learning_study/data/motion_cache/walk1_subject1/L_UpLeg/.done
        # Extract bvh_stem/region
        local rel=${done_path#*motion_cache/}  # walk1_subject1/L_UpLeg/.done
        local region_dir=$(dirname "$rel")      # walk1_subject1/L_UpLeg
        local bvh_stem=$(dirname "$region_dir") # walk1_subject1
        local region=$(basename "$region_dir")  # L_UpLeg

        found=$((found + 1))

        # Skip if already downloaded locally
        if [ -f "${LOCAL_BASE}/${region_dir}/.done.synced" ]; then
            continue
        fi

        echo "  SYNC  ${bvh_stem}/${region}"
        mkdir -p "${LOCAL_BASE}/${region_dir}"

        # Download
        rsync -az -e "$RSYNC_SSH" \
            "jek5224@a6000:${REMOTE_BASE}/${region_dir}/" \
            "${LOCAL_BASE}/${region_dir}/"

        # Mark locally as synced
        touch "${LOCAL_BASE}/${region_dir}/.done.synced"

        # Remove from server to free space
        $SSH "rm -rf ${REMOTE_BASE}/${region_dir}" 2>/dev/null || true

        synced=$((synced + 1))
    done

    # Count how many total tasks we expect
    local n_bvh=$($SSH "wc -l < ${REMOTE_BASE}/task_list.txt 2>/dev/null" || echo 0)
    local n_local=$(find "$LOCAL_BASE" -name ".done.synced" 2>/dev/null | wc -l)

    echo "[$(date +%H:%M:%S)] Found ${found} done on server, synced ${synced} new. Local total: ${n_local}/${n_bvh}"

    # Return 1 if all done
    if [ "$n_bvh" -gt 0 ] && [ "$n_local" -ge "$n_bvh" ]; then
        echo "ALL TASKS COMPLETE"
        return 1
    fi
    return 0
}

if [ "${1:-}" = "--loop" ]; then
    echo "Polling server every 60s for completed bakes... (Ctrl+C to stop)"
    while true; do
        sync_once || break
        sleep 60
    done
else
    sync_once || true
fi
