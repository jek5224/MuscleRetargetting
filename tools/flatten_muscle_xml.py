"""Flatten zygote_muscle.xml: convert <Unit><Fiber/></Unit> to <Unit/>.

Each <Fiber> becomes its own <Unit name="{parent}_{i}"> where i is 0-indexed.
Unit-level attributes (f0, lm, lt, pen_angle, lmax) are copied verbatim onto
every emitted <Unit>.  Waypoint elements are written unchanged.
"""
import argparse
import os
import xml.etree.ElementTree as ET


def flatten(src_path, dst_path):
    tree = ET.parse(src_path)
    root = tree.getroot()
    if root.tag != 'Muscle':
        raise ValueError(f'expected root <Muscle>, got <{root.tag}>')

    new_root = ET.Element('Muscle')
    n_units = 0
    n_fibers = 0
    for unit in root.findall('Unit'):
        parent_name = unit.attrib['name']
        unit_attrs = {k: v for k, v in unit.attrib.items() if k != 'name'}
        fibers = unit.findall('Fiber')
        if not fibers:
            # No <Fiber>: keep unit as-is, just clone with no children/children
            n_unit = ET.SubElement(new_root, 'Unit',
                                   attrib={'name': parent_name, **unit_attrs})
            for wp in unit.findall('Waypoint'):
                ET.SubElement(n_unit, 'Waypoint', attrib=wp.attrib)
            n_units += 1
            continue
        for i, fiber in enumerate(fibers):
            new_name = f'{parent_name}_{i}'
            n_unit = ET.SubElement(new_root, 'Unit',
                                   attrib={'name': new_name, **unit_attrs})
            for wp in fiber.findall('Waypoint'):
                ET.SubElement(n_unit, 'Waypoint', attrib=wp.attrib)
            n_units += 1
            n_fibers += 1
    # Manual tab-indent for Python <3.9 compat.
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
    _indent(new_root)
    new_tree = ET.ElementTree(new_root)
    new_tree.write(dst_path, encoding='utf-8', xml_declaration=False)
    print(f'Wrote {n_units} units ({n_fibers} fibers) -> {dst_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='data/zygote_muscle.xml')
    p.add_argument('--dst', default='data/zygote_muscle_100.xml')
    args = p.parse_args()
    flatten(args.src, args.dst)


if __name__ == '__main__':
    main()
