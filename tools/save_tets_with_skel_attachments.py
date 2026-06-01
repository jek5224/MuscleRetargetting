"""Load each leg muscle, init soft body (auto-detects attach_skeleton_names),
then save tet back so attach_skeleton_names is persisted on disk.

Needed because reverse_lbs_solve_multi.py reads attach_skeleton_names from
the tet npz, and the mirrored R tets / freshly-resampled L tets had empty
attach_skeleton_names — auto_detect only runs at init_soft_body time.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.patch_25fiber_waypoints import (
    load_skeleton, load_skeleton_meshes, discover_leg_muscles, load_muscle_meshes,
)


def main():
    skel, bvh_info, mesh_info = load_skeleton()
    skeleton_meshes = load_skeleton_meshes()
    muscle_paths = discover_leg_muscles()
    print(f"Discovered {len(muscle_paths)} leg muscles")
    muscle_meshes = load_muscle_meshes(muscle_paths)

    saved = 0
    skipped = 0
    for name, mobj in muscle_meshes.items():
        mobj.load_tetrahedron_mesh(name)
        if mobj.tet_vertices is None:
            print(f"  SKIP {name}: no tet")
            skipped += 1
            continue
        mobj.init_soft_body(
            skeleton_meshes=skeleton_meshes,
            skeleton=skel,
            mesh_info=mesh_info,
        )
        asn = getattr(mobj, "attach_skeleton_names", None)
        if not asn:
            print(f"  WARN {name}: no attach_skeleton_names after init")
        ok = mobj.save_tetrahedron_mesh(name)
        if ok:
            saved += 1
            print(f"  Saved {name}: asn={asn}")
        else:
            skipped += 1
    print(f"Done. Saved {saved}, skipped {skipped}")


if __name__ == "__main__":
    main()
