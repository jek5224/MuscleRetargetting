"""Audit zygote_muscle_revised.xml: detect fibers whose waypoint ordering is
reversed relative to the muscle's anatomical origin → insertion direction.

Reference origin/insertion comes from each muscle's tet file
(`tet/{muscle}_tet.npz` → `attach_skeleton_names[stream][0/1]`).  For each
fiber in the XML we check:
  - First-waypoint body == origin bone (after suffix-aware resolve).
  - Last-waypoint body == insertion bone.
If swapped or non-matching, fiber is flagged.
"""
import os
import pickle
import re
import xml.etree.ElementTree as ET


def base_resolve(name):
    """Strip trailing digits to match attach_skeleton_names format."""
    if not name:
        return name
    # attach_skeleton_names uses 'L_Femur' but XML uses 'L_Femur0' — match
    # by stripping trailing digits from one side.
    return re.sub(r'\d+$', '', name)


def load_tet_origin_insertion(muscle_name):
    """Return list of (origin_body, insertion_body) per stream, or None."""
    # Handle split names: Biceps_Femoris_Long -> Biceps_Femoris tet, Gastrocnemius_Medial -> Gastrocnemius tet
    candidates = [muscle_name]
    for suffix in ('_Long', '_Short', '_Medial', '_Lateral'):
        if muscle_name.endswith(suffix):
            candidates.append(muscle_name[:-len(suffix)])
    for cand in candidates:
        path = f'/home/jek/muscle_imitation_learning_study/tet/{cand}_tet.npz'
        if not os.path.exists(path):
            continue
        with open(path, 'rb') as f:
            d = pickle.load(f)
        asn = d.get('attach_skeleton_names')
        if asn is None:
            return None
        return [(s[0], s[1]) if len(s) >= 2 else None for s in asn]
    return None


def main():
    tree = ET.parse('/home/jek/muscle_imitation_learning_study/data/zygote_muscle_revised.xml')
    root = tree.getroot()
    issues = []
    units_audited = 0
    fibers_audited = 0

    for unit in root.findall('Unit'):
        uname = unit.attrib['name']
        oi = load_tet_origin_insertion(uname)
        if oi is None:
            issues.append((uname, '-', 'no_tet_metadata', '', ''))
            continue
        units_audited += 1
        fibers = unit.findall('Fiber')
        # Build set of origin bones / insertion bones across streams (any
        # may apply to a fiber, depending on which stream it belongs to).
        # Without per-fiber stream tagging in XML, we just verify the first
        # waypoint body matches ANY origin, last matches ANY insertion.
        origins = {base_resolve(o) for (o, _) in oi if o}
        insertions = {base_resolve(i) for (_, i) in oi if i}
        for fi, fb in enumerate(fibers):
            wps = fb.findall('Waypoint')
            if len(wps) < 2:
                continue
            fibers_audited += 1
            first_body = base_resolve(wps[0].attrib.get('body', ''))
            last_body = base_resolve(wps[-1].attrib.get('body', ''))
            # Case 1: correctly oriented
            if first_body in origins and last_body in insertions:
                continue
            # Case 2: reversed (insertion → origin)
            if first_body in insertions and last_body in origins:
                issues.append((uname, str(fi), 'REVERSED', first_body, last_body))
                continue
            # Case 3: ambiguous (e.g. both endpoints attach to same bone)
            issues.append((uname, str(fi), 'AMBIGUOUS', first_body, last_body))

    print(f'Audited {units_audited} units, {fibers_audited} fibers')
    print(f'Issues: {len(issues)}')
    if issues:
        # Aggregate by (unit, status)
        from collections import defaultdict
        agg = defaultdict(int)
        for u, _, status, fb, lb in issues:
            agg[(u, status, fb, lb)] += 1
        print(f'\n{"unit":<40} {"status":<15} {"first→last":<35} {"count":>5}')
        print('-' * 100)
        for (u, status, fb, lb), c in sorted(agg.items()):
            print(f'{u:<40} {status:<15} {fb + " -> " + lb:<35} {c:>5}')
    else:
        print('All fibers properly oriented (first=origin, last=insertion).')


if __name__ == '__main__':
    main()
