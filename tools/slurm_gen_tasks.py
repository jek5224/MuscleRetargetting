#!/usr/bin/env python3
"""Generate task_list.txt for slurm_bake.sh — one line per (region, bvh) pair.

Skips tasks that are already baked:
  - Local:  data/motion_cache/{stem}/.done  (all regions done)
  - Server: data/motion_cache/{stem}/{region}/.done  (that region done)

Usage:
  python tools/slurm_gen_tasks.py                          # all 4 regions
  python tools/slurm_gen_tasks.py --regions L_UpLeg,L_LowLeg
"""
import argparse
import glob
import os
import subprocess

ALL_REGIONS = [
    ("L_UpLeg",  "tools/muscles_L_UpLeg.json"),
    ("R_UpLeg",  "tools/muscles_R_UpLeg.json"),
    ("L_LowLeg", "tools/muscles_L_LowLeg.json"),
    ("R_LowLeg", "tools/muscles_R_LowLeg.json"),
]

SSH_CMD = "ssh -p 7777 -i ~/.ssh/id_ed25519_a6000 jek5224@a6000"
REMOTE_BASE = "~/muscle_imitation_learning_study/data/motion_cache"
TASK_LIST = os.path.join("data", "motion_cache", "task_list.txt")


def get_local_done_stems():
    """Return set of BVH stems that are fully baked locally."""
    stems = set()
    for path in glob.glob("data/motion_cache/*/.done"):
        stem = os.path.basename(os.path.dirname(path))
        stems.add(stem)
    return stems


def get_server_done(regions):
    """Return set of (stem, region) pairs already baked on the server."""
    region_tags = {r for r, _ in regions}
    # Single SSH call: find all .done markers under region dirs
    find_cmd = f"find {REMOTE_BASE} -mindepth 3 -maxdepth 3 -name .done 2>/dev/null"
    try:
        result = subprocess.run(
            SSH_CMD.split() + [find_cmd],
            capture_output=True, text=True, timeout=30,
        )
        done = set()
        for line in result.stdout.strip().splitlines():
            # e.g. ~/muscle_.../data/motion_cache/walk1_subject1/L_UpLeg/.done
            parts = line.rstrip("/").split("/")
            if len(parts) >= 3:
                region = parts[-2]
                stem = parts[-3]
                if region in region_tags:
                    done.add((stem, region))
        return done
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"WARNING: Could not check server .done markers: {e}")
        return set()


def main():
    parser = argparse.ArgumentParser(description="Generate slurm bake task list")
    parser.add_argument(
        "--regions", type=str, default=None,
        help="Comma-separated region tags to include (default: all 4). "
             "e.g. --regions L_UpLeg,L_LowLeg",
    )
    args = parser.parse_args()

    # Filter regions
    if args.regions:
        selected = set(args.regions.split(","))
        regions = [(r, j) for r, j in ALL_REGIONS if r in selected]
        unknown = selected - {r for r, _ in ALL_REGIONS}
        if unknown:
            print(f"WARNING: Unknown regions: {unknown}")
    else:
        regions = ALL_REGIONS

    bvh_files = sorted(glob.glob("data/motion/*.bvh"))
    if not bvh_files:
        print("ERROR: No BVH files found in data/motion/")
        return

    # Check what's already done
    local_done = get_local_done_stems()
    server_done = get_server_done(regions)

    print(f"Found {len(local_done)} fully-baked BVHs locally, "
          f"{len(server_done)} region-bakes on server")

    os.makedirs(os.path.dirname(TASK_LIST), exist_ok=True)

    lines = []
    skipped_local = 0
    skipped_server = 0
    for region_tag, muscles_json in regions:
        if not os.path.exists(muscles_json):
            print(f"WARNING: {muscles_json} not found, skipping {region_tag}")
            continue
        for bvh in bvh_files:
            stem = os.path.splitext(os.path.basename(bvh))[0]
            if stem in local_done:
                skipped_local += 1
                continue
            if (stem, region_tag) in server_done:
                skipped_server += 1
                continue
            lines.append(f"{region_tag} {muscles_json} {bvh}")

    with open(TASK_LIST, "w") as f:
        f.write("\n".join(lines) + "\n")

    n_regions = sum(1 for _, j in regions if os.path.exists(j))
    total_possible = n_regions * len(bvh_files)
    print(f"\nSkipped {skipped_local} (local done) + {skipped_server} (server done) "
          f"= {skipped_local + skipped_server} of {total_possible} possible tasks")
    print(f"Generated {len(lines)} tasks")
    print(f"  -> {TASK_LIST}")
    if lines:
        print(f"\nTo submit:")
        print(f"  sbatch --array=1-{len(lines)}%5 tools/slurm_bake.sh")


if __name__ == "__main__":
    main()
