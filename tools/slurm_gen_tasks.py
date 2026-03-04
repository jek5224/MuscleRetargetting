#!/usr/bin/env python3
"""Generate task_list.txt for slurm_bake.sh — one line per (region, bvh) pair."""
import glob
import os

REGIONS = [
    ("L_UpLeg",  "tools/muscles_L_UpLeg.json"),
    ("R_UpLeg",  "tools/muscles_R_UpLeg.json"),
    ("L_LowLeg", "tools/muscles_L_LowLeg.json"),
    ("R_LowLeg", "tools/muscles_R_LowLeg.json"),
]

TASK_LIST = os.path.join("data", "motion_cache", "task_list.txt")


def main():
    bvh_files = sorted(glob.glob("data/motion/*.bvh"))
    if not bvh_files:
        print("ERROR: No BVH files found in data/motion/")
        return

    os.makedirs(os.path.dirname(TASK_LIST), exist_ok=True)

    lines = []
    for region_tag, muscles_json in REGIONS:
        if not os.path.exists(muscles_json):
            print(f"WARNING: {muscles_json} not found, skipping {region_tag}")
            continue
        for bvh in bvh_files:
            lines.append(f"{region_tag} {muscles_json} {bvh}")

    with open(TASK_LIST, "w") as f:
        f.write("\n".join(lines) + "\n")

    n_regions = sum(1 for _, j in REGIONS if os.path.exists(j))
    print(f"Generated {len(lines)} tasks ({n_regions} regions x {len(bvh_files)} BVH files)")
    print(f"  -> {TASK_LIST}")
    print(f"\nTo submit:")
    print(f"  sbatch --array=1-{len(lines)}%5 tools/slurm_bake.sh")


if __name__ == "__main__":
    main()
