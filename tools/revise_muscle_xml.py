"""Write data/zygote_muscle_revised.xml from data/zygote_muscle.xml.

Updates each Unit's f0 attribute using values from data/muscle284.xml.
Special handling:
  - Iliacus: use Psoas_Major f0 (muscle284 has no Iliacus).
  - Biceps_Femoris (200 fibers): split into _Long (100, origin=pelvis) and
    _Short (100, origin=femur).  f0s 705.2 vs 157.9.
  - Gastrocnemius (200 fibers): split into _Medial (100, first stream) and
    _Lateral (100, second stream).  f0s 1308.0 vs 606.4.
Multi-line muscles in muscle284 (e.g. Gluteus_Maximus split into _Maximus,
_Maximus1..4) all share a single f0 — use it.
"""
import re
import xml.etree.ElementTree as ET


def base_name(n):
    return re.sub(r'\d+$', '', n)


def load_m284_f0(path):
    tree = ET.parse(path)
    by_full = {}
    for u in tree.getroot().findall('Unit'):
        by_full[u.attrib['name']] = float(u.attrib['f0'])
    return by_full


def f0_for_zygote(zname, m284):
    """Return list of (f0, label_suffix or None) — usually single entry. None
    suffix means no split needed.  Returns None if unmappable."""
    # Direct base-name match (single or multi-line, all sharing one f0)
    matches = [n for n in m284 if base_name(n) == zname]
    if len(matches) >= 1:
        f0s = {m284[m] for m in matches}
        if len(f0s) == 1:
            return [(f0s.pop(), None)]

    # Iliacus → Psoas_Major
    if zname.endswith('Iliacus'):
        side = zname[:2]  # 'L_' or 'R_'
        psoas = [n for n in m284 if base_name(n) == side + 'Psoas_Major']
        if psoas:
            return [(m284[psoas[0]], None)]

    # Biceps_Femoris → split Long + Short
    if zname.endswith('Biceps_Femoris'):
        side = zname[:2]
        long_matches = [n for n in m284 if base_name(n) == side + 'Bicep_Femoris_Longus']
        short_matches = [n for n in m284 if base_name(n) == side + 'Bicep_Femoris_Short']
        if long_matches and short_matches:
            return [
                (m284[long_matches[0]], '_Long'),
                (m284[short_matches[0]], '_Short'),
            ]

    # Gastrocnemius → split Medial + Lateral
    if zname.endswith('Gastrocnemius'):
        side = zname[:2]
        med = [n for n in m284 if base_name(n) == side + 'Gastrocnemius_Medial_Head']
        lat = [n for n in m284 if base_name(n) == side + 'Gastrocnemius_Lateral_Head']
        if med and lat:
            return [
                (m284[med[0]], '_Medial'),
                (m284[lat[0]], '_Lateral'),
            ]

    return None


def split_fibers_biceps(fibers):
    """Biceps Femoris: split by first-waypoint body name.  Long = Os_Coxae,
    Short = Femur."""
    long_f, short_f = [], []
    for fb in fibers:
        wp0 = fb.find('Waypoint')
        body = wp0.attrib.get('body', '')
        if 'Os_Coxae' in body:
            long_f.append(fb)
        elif 'Femur' in body:
            short_f.append(fb)
        else:
            long_f.append(fb)
    return long_f, short_f


def split_fibers_gastroc(fibers):
    """Gastrocnemius: stream-0 (first half) = Medial, stream-1 (second half) =
    Lateral.  Determined by the zygote bake order: first 100 fibers belong to
    the medial-side contour, next 100 to lateral.  Verified by frame-0 X-coord
    inspection."""
    n = len(fibers)
    half = n // 2
    return list(fibers)[:half], list(fibers)[half:]


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


def main():
    m284 = load_m284_f0('/home/jek/muscle_imitation_learning_study/data/muscle284.xml')
    tree = ET.parse('/home/jek/muscle_imitation_learning_study/data/zygote_muscle.xml')
    src_root = tree.getroot()

    new_root = ET.Element('Muscle')
    n_units_out = 0
    summary = []
    for unit in src_root.findall('Unit'):
        zname = unit.attrib['name']
        result = f0_for_zygote(zname, m284)
        if result is None:
            # No mapping found — keep original f0
            n_unit = ET.SubElement(new_root, 'Unit', attrib=dict(unit.attrib))
            for child in unit:
                n_unit.append(child)
            summary.append((zname, 'KEPT_ORIGINAL', float(unit.attrib['f0'])))
            n_units_out += 1
            continue
        if len(result) == 1:
            f0, _ = result[0]
            new_attrs = dict(unit.attrib)
            new_attrs['f0'] = f'{f0:.6f}'
            n_unit = ET.SubElement(new_root, 'Unit', attrib=new_attrs)
            for child in unit:
                n_unit.append(child)
            summary.append((zname, f'f0={f0}', float(unit.attrib['f0'])))
            n_units_out += 1
        else:
            # Split into multiple Units
            fibers = unit.findall('Fiber')
            if zname.endswith('Biceps_Femoris'):
                groups = split_fibers_biceps(fibers)
            elif zname.endswith('Gastrocnemius'):
                groups = split_fibers_gastroc(fibers)
            else:
                # Unexpected split case — fallback to first f0
                groups = [list(fibers)]
            for (f0, suffix), group_fibers in zip(result, groups):
                new_name = zname + (suffix or '')
                new_attrs = dict(unit.attrib)
                new_attrs['name'] = new_name
                new_attrs['f0'] = f'{f0:.6f}'
                n_unit = ET.SubElement(new_root, 'Unit', attrib=new_attrs)
                for fb in group_fibers:
                    n_unit.append(fb)
                summary.append((new_name, f'SPLIT f0={f0} fibers={len(group_fibers)}',
                                float(unit.attrib['f0'])))
                n_units_out += 1

    _indent(new_root)
    out_path = '/home/jek/muscle_imitation_learning_study/data/zygote_muscle_revised.xml'
    ET.ElementTree(new_root).write(out_path, encoding='utf-8', xml_declaration=False)
    print(f'Wrote {n_units_out} Units to {out_path}')
    print()
    print(f'{"name":<40} {"action":<40} {"prev_f0":>10}')
    print('-' * 95)
    for name, act, prev in summary:
        print(f'{name:<40} {act:<40} {prev:>10.2f}')


if __name__ == '__main__':
    main()
