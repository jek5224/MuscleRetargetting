#!/usr/bin/env python3
"""Merge per-region bake outputs into the flat cache layout the viewer expects.

After slurm baking, cache looks like:
    data/motion_cache/<bvh_stem>/L_UpLeg/L_Adductor_Brevis_chunk_0000.npz
    data/motion_cache/<bvh_stem>/R_UpLeg/R_Adductor_Brevis_chunk_0000.npz
    ...

The viewer expects:
    data/motion_cache/<bvh_stem>/L_Adductor_Brevis_chunk_0000.npz
    data/motion_cache/<bvh_stem>/R_Adductor_Brevis_chunk_0000.npz
    ...

This script moves files from region subdirs up to the bvh_stem level,
then writes a combined .done marker.
"""
import glob
import os
import shutil
import sys

CACHE_DIR = os.path.join("data", "motion_cache")
REGIONS = ["L_UpLeg", "R_UpLeg", "L_LowLeg", "R_LowLeg"]


def merge_bvh(bvh_stem):
    """Merge region subdirs for one BVH into flat layout."""
    base = os.path.join(CACHE_DIR, bvh_stem)
    if not os.path.isdir(base):
        return False, "directory not found"

    # Check all regions are done (.done from server, .done.synced from sync script)
    missing = []
    for region in REGIONS:
        region_dir = os.path.join(base, region)
        has_done = (os.path.exists(os.path.join(region_dir, ".done"))
                    or os.path.exists(os.path.join(region_dir, ".done.synced")))
        if not has_done:
            missing.append(region)

    if missing:
        return False, f"missing .done: {', '.join(missing)}"

    # Move all chunk files up
    moved = 0
    for region in REGIONS:
        region_dir = os.path.join(base, region)
        for f in glob.glob(os.path.join(region_dir, "*_chunk_*.npz")):
            dest = os.path.join(base, os.path.basename(f))
            shutil.move(f, dest)
            moved += 1
        # Clean up region dir
        shutil.rmtree(region_dir)

    # Write combined .done
    with open(os.path.join(base, ".done"), "w") as f:
        f.write(f"merged {len(REGIONS)} regions, {moved} files\n")

    return True, f"{moved} files merged"


def main():
    # Find all bvh stems that have region subdirs
    if len(sys.argv) > 1:
        stems = sys.argv[1:]
    else:
        stems = sorted(set(
            d for d in os.listdir(CACHE_DIR)
            if os.path.isdir(os.path.join(CACHE_DIR, d))
            and d != "logs"
            and any(os.path.isdir(os.path.join(CACHE_DIR, d, r)) for r in REGIONS)
        ))

    if not stems:
        print("Nothing to merge.")
        return

    ok = 0
    fail = 0
    for stem in stems:
        success, msg = merge_bvh(stem)
        status = "OK" if success else "SKIP"
        print(f"  {status:>4}  {stem}: {msg}")
        if success:
            ok += 1
        else:
            fail += 1

    print(f"\nMerged: {ok}, Skipped: {fail}")


if __name__ == "__main__":
    main()
