"""Patch R-side tet files: collapse doubled-digit body names introduced by
the mirror script bug (e.g. R_Toe550 -> R_Toe50, R_Toe110 -> R_Toe10).

Walks every R_*_tet.npz under tet/, fixes waypoint_bary_coords skeleton refs
and attach_skeleton_names. Writes files in place.
"""
import argparse
import glob
import os
import pickle
import re


# Match a name ending with two-or-more identical digits followed by a single 0.
# Collapses the run of repeated digits to one (R_Toe550 -> R_Toe50).
DOUBLED_DIGIT_PAT = re.compile(r'(\d)\1+0$')


def fix_name(name):
    if not isinstance(name, str):
        return name
    m = DOUBLED_DIGIT_PAT.search(name)
    if not m:
        return name
    return DOUBLED_DIGIT_PAT.sub(m.group(1) + '0', name)


def fix_wbc(wbc):
    """Walk waypoint_bary_coords, fix doubled-digit body names. Returns count fixed."""
    if wbc is None:
        return 0
    n_fixed = 0
    for stream in wbc:
        if stream is None:
            continue
        for contour in stream:
            if contour is None:
                continue
            for i, fiber in enumerate(contour):
                if fiber is None:
                    continue
                if isinstance(fiber, tuple) and fiber[0] == 'skeleton':
                    body = fiber[1]
                    new_body = fix_name(body)
                    if new_body != body:
                        contour[i] = ('skeleton', new_body, fiber[2])
                        n_fixed += 1
    return n_fixed


def fix_asn(asn):
    if asn is None:
        return 0
    n_fixed = 0
    for group in asn:
        for i, name in enumerate(group):
            new_name = fix_name(name)
            if new_name != name:
                group[i] = new_name
                n_fixed += 1
    return n_fixed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tet-dir', default='tet')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.tet_dir, 'R_*_tet.npz')))
    print(f'Found {len(paths)} R tet files')

    total = 0
    for path in paths:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        wbc_fixed = fix_wbc(data.get('waypoint_bary_coords'))
        asn_fixed = fix_asn(data.get('attach_skeleton_names'))
        if wbc_fixed or asn_fixed:
            print(f'  {os.path.basename(path)}: wbc={wbc_fixed} asn={asn_fixed}')
            total += wbc_fixed + asn_fixed
            if not args.dry_run:
                with open(path, 'wb') as f:
                    pickle.dump(data, f)
    print(f'Total fixed: {total} entries')


if __name__ == '__main__':
    main()
