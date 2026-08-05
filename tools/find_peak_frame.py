"""Find frame with maximum per-vertex displacement vs previous frame.

Useful for picking a single frame to test bake quality on.

Usage:
    python tools/find_peak_frame.py <cache_dir> [topk]
"""
import sys, os, glob, re
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
        m = re.match(r'(.+)_chunk_\d+\.npz', os.path.basename(f))
        if m:
            out.add(m.group(1))
    return sorted(out)


def main(cache_dir, topk=10):
    muscles = list_muscles(cache_dir)
    # frame -> total max-disp across muscles
    per_frame = {}
    per_muscle_peak = []
    for m in muscles:
        f, p = load_muscle_frames(cache_dir, m)
        if f is None or len(p) < 2:
            continue
        # displacement between consecutive frames
        diff = np.linalg.norm(p[1:] - p[:-1], axis=-1)  # (F-1, V)
        per_frame_max = diff.max(axis=1)  # max vert displacement per frame
        per_frame_mean = diff.mean(axis=1)
        worst_f_idx = int(np.argmax(per_frame_max))
        per_muscle_peak.append(
            (m, float(per_frame_max[worst_f_idx]), int(f[worst_f_idx + 1]))
        )
        for i, fr in enumerate(f[1:]):
            cur = per_frame.get(int(fr), 0.0)
            per_frame[int(fr)] = max(cur, float(per_frame_max[i]))
    ranked = sorted(per_frame.items(), key=lambda x: -x[1])
    print("Top frames by max per-vert displacement (any muscle):")
    for fr, d in ranked[:topk]:
        print(f"  frame={fr:4d}  max_disp={d*1000:.3f}mm")
    print()
    print("Per-muscle peak frame:")
    for m, d, fr in sorted(per_muscle_peak, key=lambda x: -x[1]):
        print(f"  {m:<32} frame={fr:4d}  {d*1000:.3f}mm")


if __name__ == '__main__':
    cache_dir = sys.argv[1]
    topk = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    main(cache_dir, topk)
