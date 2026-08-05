"""Decompress every *_chunk_*.npz under given cache dirs in place.

np.savez_compressed → np.savez. Trades 2× disk for ~4× load speed.

Usage:
    python tools/decompress_cache.py data/motion_cache/walk/L_LowLeg [...]
"""
import sys, os, glob
import numpy as np


def decompress_dir(d):
    chunks = glob.glob(os.path.join(d, '*_chunk_*.npz'))
    if not chunks:
        print(f"  no chunks in {d}")
        return
    n_done = 0
    for c in chunks:
        data = np.load(c, allow_pickle=True)
        payload = {k: data[k] for k in data.files}
        tmp_base = c + '.tmp'   # np.savez appends .npz automatically
        np.savez(tmp_base, **payload)
        os.replace(tmp_base + '.npz', c)
        n_done += 1
    print(f"  {d}: {n_done} chunks decompressed")


if __name__ == '__main__':
    for d in sys.argv[1:]:
        decompress_dir(d)
