"""Add LBS attributes (lbs_bones, lbs_locals, lbs_weights) to every
<Waypoint> in data/zygote_muscle_revised.xml, sourced from
reverse_lbs_results/*.npz.

Existing `body` and `p` attrs preserved (back-compat).  LBS-aware loaders
can read the new attrs; legacy loaders ignore them.

Mapping XML Unit -> solver NPZ:
  * Single-stream muscles: direct name match.
  * Split units (L_Biceps_Femoris_Long, _Short, L_Gastrocnemius_Medial,
    _Lateral) -> parent NPZ, filtered:
        Biceps_Femoris_Long  -> records whose origin_body has 'Os_Coxae'
        Biceps_Femoris_Short -> records whose origin_body has 'Femur'
        Gastrocnemius_Medial -> first half of records (stream 0)
        Gastrocnemius_Lateral-> second half (stream 1)
"""
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np


REVERSE_DIR = '/home/jek/muscle_imitation_learning_study/reverse_lbs_results'
XML_PATH = '/home/jek/muscle_imitation_learning_study/data/zygote_muscle_revised.xml'


def fmt_vec(v):
    return f'{float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}'


def load_solver_records(muscle_name):
    """Load records for a muscle (or parent if split)."""
    path = os.path.join(REVERSE_DIR, muscle_name + '.npz')
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return list(d['records'])


def select_records_for_unit(unit_name, all_recs):
    """For split units, filter parent's records to the relevant subset.
    Returns: list of records ordered the same way XML emits fibers."""
    if unit_name.endswith('_Long'):
        recs = [r for r in all_recs if 'Os_Coxae' in r['origin_body']]
    elif unit_name.endswith('_Short'):
        recs = [r for r in all_recs if 'Femur' in r['origin_body']]
    elif unit_name.endswith('_Medial'):
        # stream 0
        recs = [r for r in all_recs if int(r['stream']) == 0]
    elif unit_name.endswith('_Lateral'):
        recs = [r for r in all_recs if int(r['stream']) == 1]
    else:
        recs = list(all_recs)
    return recs


def parent_name(unit_name):
    for suf in ('_Long', '_Short', '_Medial', '_Lateral'):
        if unit_name.endswith(suf):
            return unit_name[:-len(suf)]
    return unit_name


def main():
    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    parent_cache = {}  # parent_name -> all_recs

    n_units = 0
    n_fibers = 0
    n_wp = 0
    n_missing = 0

    for unit in root.findall('Unit'):
        uname = unit.attrib['name']
        pname = parent_name(uname)
        if pname not in parent_cache:
            parent_cache[pname] = load_solver_records(pname)
        all_recs = parent_cache[pname]
        if all_recs is None:
            print(f'  {uname}: no NPZ for parent {pname}, skipping LBS attrs')
            continue
        recs = select_records_for_unit(uname, all_recs)
        # Group by (stream, fiber), sort within each by level
        groups = defaultdict(list)
        for r in recs:
            groups[(int(r['stream']), int(r['fiber']))].append(r)
        for k in groups:
            groups[k].sort(key=lambda r: int(r['level']))
        ordered_keys = sorted(groups.keys())

        xml_fibers = unit.findall('Fiber')
        if len(xml_fibers) != len(ordered_keys):
            print(f'  WARN {uname}: XML has {len(xml_fibers)} fibers, NPZ has {len(ordered_keys)}')
            continue

        for fi, fb in enumerate(xml_fibers):
            recs_for_fiber = groups[ordered_keys[fi]]
            xml_wps = fb.findall('Waypoint')
            if len(xml_wps) != len(recs_for_fiber):
                print(f'  WARN {uname} fiber {fi}: XML wps {len(xml_wps)} vs NPZ recs {len(recs_for_fiber)}')
                n_missing += 1
                continue
            for wi, wp in enumerate(xml_wps):
                r = recs_for_fiber[wi]
                w_o = float(r.get('w_o', 0.0))
                w_m = float(r.get('w_m', 0.0))
                w_i = float(r.get('w_i', 0.0))
                bones = []
                locals_ = []
                weights = []
                if w_o > 0:
                    bones.append(r['origin_body'])
                    locals_.append(r['local_o'])
                    weights.append(w_o)
                if w_m > 0 and r.get('mid_body'):
                    bones.append(r['mid_body'])
                    locals_.append(r['local_m'])
                    weights.append(w_m)
                if w_i > 0:
                    bones.append(r['insertion_body'])
                    locals_.append(r['local_i'])
                    weights.append(w_i)
                wp.set('lbs_bones', ','.join(bones))
                wp.set('lbs_locals', '; '.join(fmt_vec(v) for v in locals_))
                wp.set('lbs_weights', ' '.join(f'{w:.6f}' for w in weights))
                n_wp += 1
            n_fibers += 1
        n_units += 1

    # Pretty-indent
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

    tree.write(XML_PATH, encoding='utf-8', xml_declaration=False)
    print(f'\nUpdated {n_units} Units, {n_fibers} Fibers, {n_wp} Waypoints with LBS attrs.')
    if n_missing:
        print(f'  {n_missing} fibers skipped due to size mismatch')


if __name__ == '__main__':
    main()
