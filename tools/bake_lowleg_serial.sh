#!/bin/bash
# Strict-serial LowLeg bake orchestrator for walk-series BVHs.
# Per BVH: submit L_LowLeg + R_LowLeg → wait → rsync down → delete on server → next.
#
# Run on local machine:
#   nohup bash tools/bake_lowleg_serial.sh > /tmp/lowleg_bake.log 2>&1 &
#
# Tail progress:  tail -f /tmp/lowleg_bake.log

set -uo pipefail
cd "$(dirname "$0")/.."

SSH="ssh -p 7777 -i $HOME/.ssh/id_ed25519_a6000 jek5224@147.47.206.51"
RSYNC_SSH="ssh -p 7777 -i $HOME/.ssh/id_ed25519_a6000"
REMOTE_REPO="muscle_imitation_learning_study"
REMOTE_CACHE="$REMOTE_REPO/data/motion_cache"
LOCAL_CACHE="data/motion_cache"

SERVER_MIN_FREE_GB=4   # bail if server free space drops below this
LOCAL_MIN_FREE_GB=10   # bail if local drops below this

# Walk BVHs to bake (LowLeg L+R per BVH, serial). Includes walk1_subject1 since
# UpLeg already baked it overnight; user may exclude later if needed.
BVHS=(
    walk
    walk1_subject1
    walk1_subject2
    walk1_subject5
    walk2_subject1
    walk2_subject3
    walk2_subject4
    walk3_subject1
    walk3_subject2
)
SIDES=(L_LowLeg R_LowLeg)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

server_free_gb() {
    $SSH "df --output=avail -BG /home | tail -1 | tr -dc '0-9'" 2>/dev/null || echo 0
}
local_free_gb() {
    # Check the filesystem actually backing data/motion_cache (it is symlinked
    # to the external drive with hundreds of GB free), not the root NVMe.
    df --output=avail -BG "$LOCAL_CACHE" 2>/dev/null | tail -1 | tr -dc '0-9'
}

submit_pair() {
    local bvh="$1"
    local job_ids=()
    for side in "${SIDES[@]}"; do
        local jid
        jid=$($SSH "cd $REMOTE_REPO && BVH_NAME=$bvh SIDE=$side sbatch --parsable --export=ALL,BVH_NAME=$bvh,SIDE=$side /tmp/slurm_walks.sh")
        log "  submitted $bvh / $side as job $jid"
        job_ids+=("$jid")
    done
    echo "${job_ids[@]}"
}

wait_jobs_done() {
    local ids=("$@")
    local id_re
    id_re=$(IFS='|'; echo "${ids[*]}")
    while true; do
        local n_remaining
        n_remaining=$($SSH "squeue -u jek5224 -h -o '%i' 2>/dev/null | grep -E '^(${id_re})\$' | wc -l" 2>/dev/null || echo 0)
        if [ "$n_remaining" = "0" ]; then break; fi
        sleep 60
    done
}

sync_pair() {
    local bvh="$1"
    mkdir -p "${LOCAL_CACHE}/${bvh}"
    for side in "${SIDES[@]}"; do
        local rel="${bvh}/_${side}"
        local local_dir="${LOCAL_CACHE}/${rel}"
        mkdir -p "$local_dir"
        log "  rsync ${bvh}/_${side}"
        rsync -az -e "$RSYNC_SSH" \
            "jek5224@147.47.206.51:${REMOTE_CACHE}/${rel}/" "${local_dir}/"
    done
}

verify_done() {
    local bvh="$1"
    for side in "${SIDES[@]}"; do
        local marker="${LOCAL_CACHE}/${bvh}/_${side}/.done"
        if [ ! -f "$marker" ]; then
            log "  MISSING $marker — bake may have failed"
            return 1
        fi
    done
    return 0
}

delete_remote() {
    local bvh="$1"
    for side in "${SIDES[@]}"; do
        $SSH "rm -rf ${REMOTE_CACHE}/${bvh}/_${side}"
    done
    log "  deleted server-side ${bvh}/_*"
}

for bvh in "${BVHS[@]}"; do
    log ""
    log "===== ${bvh} ====="

    server_gb=$(server_free_gb)
    local_gb=$(local_free_gb)
    log "  server free=${server_gb}G, local free=${local_gb}G"
    if [ "$server_gb" -lt "$SERVER_MIN_FREE_GB" ]; then
        log "ABORT: server <${SERVER_MIN_FREE_GB}G free"
        exit 1
    fi
    if [ "$local_gb" -lt "$LOCAL_MIN_FREE_GB" ]; then
        log "ABORT: local <${LOCAL_MIN_FREE_GB}G free"
        exit 1
    fi

    log "  submit pair"
    ids=$(submit_pair "$bvh")
    log "  job ids: $ids"

    log "  wait for both"
    wait_jobs_done $ids
    log "  jobs finished"

    log "  sync down"
    sync_pair "$bvh"

    if verify_done "$bvh"; then
        log "  verified .done markers — deleting server copies"
        delete_remote "$bvh"
    else
        log "  SKIP delete (no .done markers); leaving server copy for inspection"
        log "ABORT to avoid running next bake without sync confirmation"
        exit 2
    fi
done

log ""
log "ALL DONE — ${#BVHS[@]} BVHs LowLeg L+R baked + synced"
