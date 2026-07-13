#!/usr/bin/env python3
"""Quick post-bake bone-penetration counter.

For a given cache dir + frame, builds bone trimeshes at that frame's pose
and reports per-muscle inside-bone vert counts.

Usage: python tools/check_bone_penetration.py <cache_dir> <frame>
"""
import sys
import os
import glob
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.dartHelper import buildFromInfo, saveSkeletonInfo
from core.bvhparser import MyBVH

SKEL_XML = "data/zygote_skel.xml"
BONE_OBJ_DIR = "Zygote_Meshes_251229/Skeleton"
MESH_SCALE = 0.01


def main():
    if len(sys.argv) < 3:
        print("usage: check_bone_penetration.py <cache_dir> <frame>")
        sys.exit(1)
    cache_dir = sys.argv[1]
    frame = int(sys.argv[2])

    sk_info, root, bvh_info, _, _, _ = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(sk_info, root)
    skel.setPositions(np.zeros(skel.getNumDofs()))
    bone_pairs = []
    for fname in os.listdir(BONE_OBJ_DIR):
        if not fname.lower().endswith('.obj'):
            continue
        stem = os.path.splitext(fname)[0]
        try:
            tm = trimesh.load(os.path.join(BONE_OBJ_DIR, fname), process=False)
        except Exception:
            continue
        if not isinstance(tm, trimesh.Trimesh):
            continue
        tm.vertices = tm.vertices * MESH_SCALE
        bn = None
        for cand in (stem + '0', stem):
            try:
                bn = skel.getBodyNode(cand)
                if bn is not None:
                    break
            except Exception:
                continue
        if bn is None:
            continue
        bone_pairs.append((bn, tm))

    # Rest-pose transforms (zero pose) — same convention as bake_layered.
    rest_xfm = {}
    for bn, _tm in bone_pairs:
        wt = bn.getWorldTransform()
        rest_xfm[bn.getName()] = (wt.rotation().copy(), wt.translation().copy())

    motion = MyBVH('data/motion/walk.bvh', bvh_info, skel, T_frame=None)
    skel.setPositions(motion.mocap_refs[frame])
    bone_meshes = []
    for bn, tm in bone_pairs:
        R0, t0 = rest_xfm[bn.getName()]
        wt = bn.getWorldTransform()
        R1, t1 = wt.rotation(), wt.translation()
        local = (R0.T @ (tm.vertices - t0).T).T
        v = (R1 @ local.T).T + t1
        bone_meshes.append(trimesh.Trimesh(vertices=v, faces=tm.faces, process=True))

    import pickle
    total = 0
    bad_muscles = []
    for chunk_path in sorted(glob.glob(os.path.join(cache_dir, '*_chunk_*.npz'))):
        mname = os.path.basename(chunk_path).split('_chunk_')[0]
        d = np.load(chunk_path, allow_pickle=True)
        frames = d['frames'].tolist()
        if frame not in frames:
            continue
        idx = frames.index(frame)
        pos = d['positions'][idx]
        # Build cap-vert exclusion set from tet's cap_attachments (these are
        # the origin/insertion verts pinned to bones — they're SUPPOSED to be
        # on bone surface, not real penetrations).
        cap_set = set()
        try:
            with open(f'tet/{mname}_tet.npz', 'rb') as f:
                t = pickle.load(f)
            ca = t.get('cap_attachments')
            if ca is not None and len(ca) > 0:
                cap_set = set(int(v) for v in np.asarray(ca)[:, 0])
            for vi in t.get('anchor_vertices', []):
                cap_set.add(int(vi))
        except Exception:
            pass
        n_inside = 0
        for bm in bone_meshes:
            try:
                bbmin = bm.bounds[0] - 0.01
                bbmax = bm.bounds[1] + 0.01
                bb_mask = np.all((pos >= bbmin) & (pos <= bbmax), axis=1)
                if not np.any(bb_mask):
                    continue
                bb_idx_arr = np.where(bb_mask)[0]
                bb_pos = pos[bb_idx_arr]
                inside = bm.contains(bb_pos)
                # Exclude any cap vert from the count.
                for k in np.where(inside)[0]:
                    if int(bb_idx_arr[k]) in cap_set:
                        continue
                    n_inside += 1
            except Exception:
                continue
        total += n_inside
        if n_inside > 0:
            bad_muscles.append((mname, n_inside, len(pos)))

    print(f"  PENETRATION: {total} verts inside any bone, "
          f"{len(bad_muscles)} muscles affected")
    for mname, n, total_v in sorted(bad_muscles, key=lambda x: -x[1])[:5]:
        print(f"    {mname}: {n}/{total_v} verts inside bone")


if __name__ == '__main__':
    main()
