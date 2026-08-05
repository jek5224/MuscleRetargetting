"""Compare two motion-cache directories per muscle, per frame.

Outputs per-muscle stats (max/RMS L2 vertex delta over all frames) plus
the global worst frame/muscle. Useful for finding the minimum acceptable
--settle-iters by diffing against a high-iter ground truth.

Usage:
    python tools/diff_caches.py <gt_dir> <variant_dir>
"""
import sys, os, glob, re
from collections import defaultdict
import numpy as np


def load_muscle_frames(cache_dir, muscle):
    chunks = sorted(glob.glob(os.path.join(cache_dir, f"{muscle}_chunk_*.npz")))
    if not chunks:
        return None, None
    frames_all = []
    pos_all = []
    for c in chunks:
        d = np.load(c)
        frames_all.append(d['frames'])
        pos_all.append(d['positions'])
    frames = np.concatenate(frames_all)
    pos = np.concatenate(pos_all, axis=0)
    order = np.argsort(frames)
    return frames[order], pos[order]


def list_muscles(cache_dir):
    out = set()
    for f in glob.glob(os.path.join(cache_dir, '*_chunk_*.npz')):
        bn = os.path.basename(f)
        m = re.match(r'(.+)_chunk_\d+\.npz', bn)
        if m:
            out.add(m.group(1))
    return sorted(out)


def diff(gt_dir, var_dir):
    muscles = list_muscles(gt_dir)
    print(f"{len(muscles)} muscles")
    summary = []
    worst_frame_global = (-1, -1.0, '', -1)  # (frame, max_d, muscle, vi)
    for m in muscles:
        fg, pg = load_muscle_frames(gt_dir, m)
        fv, pv = load_muscle_frames(var_dir, m)
        if fv is None:
            print(f"  MISSING {m} in variant")
            continue
        # align frame intersection
        common = np.intersect1d(fg, fv)
        if len(common) == 0:
            continue
        ig = np.searchsorted(fg, common)
        iv = np.searchsorted(fv, common)
        pg2 = pg[ig]
        pv2 = pv[iv]
        if pg2.shape[1] != pv2.shape[1]:
            print(f"  vert mismatch {m}: gt {pg2.shape[1]} vs var {pv2.shape[1]}")
            continue
        d = np.linalg.norm(pg2 - pv2, axis=-1)  # (frames, verts)
        per_frame_max = d.max(axis=1)  # mm-ish; positions are meters
        per_frame_rms = np.sqrt((d**2).mean(axis=1))
        max_d = per_frame_max.max()
        max_frame_idx = int(np.argmax(per_frame_max))
        max_frame = int(common[max_frame_idx])
        max_vi = int(np.argmax(d[max_frame_idx]))
        mean_max = per_frame_max.mean()
        mean_rms = per_frame_rms.mean()
        p95_max = np.percentile(per_frame_max, 95)
        summary.append((m, max_d, mean_max, mean_rms, p95_max, max_frame, max_vi))
        if max_d > worst_frame_global[1]:
            worst_frame_global = (max_frame, max_d, m, max_vi)
    print()
    print(f"{'muscle':<32} {'max_mm':>9} {'mean_max_mm':>12} {'mean_rms_mm':>12} {'p95_max_mm':>11} {'worst_frame':>12}")
    summary.sort(key=lambda x: -x[1])
    for m, mx, mm, mr, p95, fr, vi in summary:
        print(f"{m:<32} {mx*1000:>9.3f} {mm*1000:>12.3f} {mr*1000:>12.3f} {p95*1000:>11.3f} {fr:>12}")
    fr, mx, m, vi = worst_frame_global
    print()
    print(f"Global worst: frame={fr} muscle={m} vi={vi} delta={mx*1000:.3f}mm")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    diff(sys.argv[1], sys.argv[2])
