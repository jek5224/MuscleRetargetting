"""Emit data/zygote_muscle_reverse.xml from reverse_lbs_results/*.npz.

Per fiber: groups solver records by (stream, fiber_idx), sorts by level,
reconstructs world rest position per waypoint, picks the highest-weight bone
as the waypoint's `body` attribute, and writes a standard <Unit><Fiber>
<Waypoint body=.. p=..> XML.

CAVEAT: this XML cannot carry the solver's per-bone local positions or
multi-bone weights — only world rest + single bone per waypoint.  Loading it
through `loading_zygote_muscle_info` (arc-length scheme) loses solver tuning
off-rest.  For full fidelity, use the new `addMuscleAnchorExplicit` binding
with `reverse_lbs_results/*.npz` directly.
"""
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.dartHelper import saveSkeletonInfo, buildFromInfo


def base_resolve(name):
    return re.sub(r'\d+$', '', name) if name else name


def load_revised_props(path):
    """Pull (f0, lm, lt, pen_angle, lmax) per muscle base name from revised XML."""
    props = {}
    if not os.path.exists(path):
        return props
    tree = ET.parse(path)
    for u in tree.getroot().findall('Unit'):
        # Map split names back to parent: e.g. L_Biceps_Femoris_Long -> L_Biceps_Femoris
        base = u.attrib['name']
        for suf in ('_Long', '_Short', '_Medial', '_Lateral'):
            if base.endswith(suf):
                base = base[:-len(suf)]
                break
        props[base] = {
            'f0': u.attrib.get('f0', '1000.0'),
            'lm': u.attrib.get('lm', '1.2'),
            'lt': u.attrib.get('lt', '0.2'),
            'pen_angle': u.attrib.get('pen_angle', '0.0'),
            'lmax': u.attrib.get('lmax', '-0.1'),
        }
    return props


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/home/jek/muscle_imitation_learning_study/reverse_lbs_results')
    ap.add_argument('--out', default='/home/jek/muscle_imitation_learning_study/data/zygote_muscle_reverse.xml')
    args = ap.parse_args()
    skel_info, root_name, *_ = saveSkeletonInfo('/home/jek/muscle_imitation_learning_study/data/zygote_skel.xml')
    skel = buildFromInfo(skel_info, root_name)
    skel.resetPositions()
    body_T = {}
    def get_T(name):
        if name not in body_T:
            b = skel.getBodyNode(name)
            if b is None:
                body_T[name] = None
            else:
                T = np.asarray(b.getWorldTransform().matrix())
                body_T[name] = (T[:3, :3].copy(), T[:3, 3].copy())
        return body_T[name]

    props_map = load_revised_props('/home/jek/muscle_imitation_learning_study/data/zygote_muscle_revised.xml')
    root = ET.Element('Muscle')

    files = sorted(glob.glob(os.path.join(args.src, '*.npz')))
    print(f'Processing {len(files)} muscle NPZ files')

    for path in files:
        muscle_name = os.path.basename(path).replace('.npz', '')
        d = np.load(path, allow_pickle=True)
        records = list(d['records'])

        fiber_groups = defaultdict(list)
        for r in records:
            fiber_groups[(int(r['stream']), int(r['fiber']))].append(r)

        # Default props if no XML match
        p = props_map.get(muscle_name, {'f0': '1000.0', 'lm': '1.2', 'lt': '0.2',
                                        'pen_angle': '0.0', 'lmax': '-0.1'})
        unit = ET.SubElement(root, 'Unit', attrib={'name': muscle_name, **p})

        for (s_idx, f_idx), recs in sorted(fiber_groups.items()):
            recs.sort(key=lambda r: int(r['level']))
            fiber = ET.SubElement(unit, 'Fiber', attrib={
                'stream': str(s_idx), 'fiber': str(f_idx)})
            for r in recs:
                w_o = float(r.get('w_o', 0.0))
                w_m = float(r.get('w_m', 0.0))
                w_i = float(r.get('w_i', 0.0))
                origin_body = r['origin_body']
                insertion_body = r['insertion_body']
                mid_body = r.get('mid_body')

                # Reconstruct world rest position from solver's (local, weight) data
                world = np.zeros(3, dtype=np.float64)
                if w_o > 0 and get_T(origin_body) is not None:
                    R, t = get_T(origin_body)
                    world += w_o * (R @ np.asarray(r['local_o'], dtype=np.float64) + t)
                if w_m > 0 and mid_body and get_T(mid_body) is not None:
                    R, t = get_T(mid_body)
                    world += w_m * (R @ np.asarray(r['local_m'], dtype=np.float64) + t)
                if w_i > 0 and get_T(insertion_body) is not None:
                    R, t = get_T(insertion_body)
                    world += w_i * (R @ np.asarray(r['local_i'], dtype=np.float64) + t)

                # Anchor body = highest-weight bone
                level = int(r['level'])
                if level == 0 or w_o >= max(w_m, w_i):
                    body = origin_body
                elif r is recs[-1] or w_i >= max(w_o, w_m):
                    body = insertion_body
                else:
                    body = mid_body if mid_body else origin_body

                # K-bone LBS attrs (skip zero-weight bones)
                bones, locals_, weights = [], [], []
                if w_o > 0:
                    bones.append(origin_body)
                    locals_.append(np.asarray(r['local_o'], dtype=np.float64))
                    weights.append(w_o)
                if w_m > 0 and mid_body:
                    bones.append(mid_body)
                    locals_.append(np.asarray(r['local_m'], dtype=np.float64))
                    weights.append(w_m)
                if w_i > 0:
                    bones.append(insertion_body)
                    locals_.append(np.asarray(r['local_i'], dtype=np.float64))
                    weights.append(w_i)

                p_str = f'{world[0]:.6f} {world[1]:.6f} {world[2]:.6f}'
                lbs_locals_str = '; '.join(
                    f'{lp[0]:.6f} {lp[1]:.6f} {lp[2]:.6f}' for lp in locals_)
                lbs_weights_str = ' '.join(f'{w:.6f}' for w in weights)
                ET.SubElement(fiber, 'Waypoint', attrib={
                    'body': body, 'p': p_str, 'level': str(level),
                    'lbs_bones': ','.join(bones),
                    'lbs_locals': lbs_locals_str,
                    'lbs_weights': lbs_weights_str,
                })

    # Pretty-indent (Python 3.8 compat)
    def _indent(elem, level=0):
        i = '\n' + level * '\t'
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + '\t'
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                _indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
    _indent(root)

    out = args.out
    ET.ElementTree(root).write(out, encoding='utf-8', xml_declaration=False)
    n_units = len(root.findall('Unit'))
    n_fibers = sum(len(u.findall('Fiber')) for u in root.findall('Unit'))
    n_wp = sum(len(f.findall('Waypoint'))
               for u in root.findall('Unit')
               for f in u.findall('Fiber'))
    print(f'Wrote {n_units} Units, {n_fibers} Fibers, {n_wp} Waypoints to {out}')


if __name__ == '__main__':
    main()
