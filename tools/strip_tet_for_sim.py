"""Strip pipeline data from L tet files, keeping only simulation-essential data.

Creates lightweight copies in tet_sim/ for fast loading and server upload.

Usage: python tools/strip_tet_for_sim.py
"""
import glob
import os
import pickle

STRIP_KEYS = {
    'contours', 'fiber_architecture', 'bounding_planes',
    'contour_to_tet_mapping',
    'mvc_weights', '_stream_endpoints',
    'stream_contours', 'stream_bounding_planes', 'stream_groups',
}

def strip_tet_file(src_path, dst_path):
    with open(src_path, 'rb') as f:
        data = pickle.load(f)

    stripped = {k: v for k, v in data.items() if k not in STRIP_KEYS}

    # Generate draw_contour_stream from waypoints
    if data.get("waypoints") is not None:
        dcs = []
        for stream in data["waypoints"]:
            dcs.append([True] * len(stream))
        stripped["draw_contour_stream"] = dcs

    with open(dst_path, 'wb') as f:
        pickle.dump(stripped, f)

    src_size = os.path.getsize(src_path)
    dst_size = os.path.getsize(dst_path)
    return src_size, dst_size


def main():
    os.makedirs('tet_sim', exist_ok=True)

    files = sorted(glob.glob('tet/*_tet.npz'))
    total_src = 0
    total_dst = 0

    for src in files:
        name = os.path.basename(src)
        dst = os.path.join('tet_sim', name)
        src_size, dst_size = strip_tet_file(src, dst)
        total_src += src_size
        total_dst += dst_size
        print(f"  {name}: {src_size/1024:.0f}K → {dst_size/1024:.0f}K")

    print(f"\nTotal: {total_src/1024/1024:.1f}MB → {total_dst/1024/1024:.1f}MB "
          f"({100*total_dst/total_src:.0f}%)")


if __name__ == '__main__':
    main()
