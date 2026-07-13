"""Insert a Sternum0 body into data/zygote_skel.xml as a Ball-joint child
of a chosen thoracic vertebra (default T50). Also moves Sternum.obj from
Zygote_Meshes_251229/Skeleton/Thorax/ to the flat Skeleton/ dir so the
viewer's startup scan picks it up.

body_t = sternum mesh centroid (world, m)
body_r = identity (axis-aligned OBB approximation)
size   = mesh bbox extents (m)
joint_t (world) = R_parent_body @ local_offset + t_parent_body
joint_r = identity
joint_type = Ball with bvh="Sternum"
"""
import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.dartHelper import saveSkeletonInfo, buildFromInfo

SKEL_XML = "data/zygote_skel.xml"
ZYGOTE_DIR = "Zygote_Meshes_251229"
STERNUM_OBJ_SRC = os.path.join(ZYGOTE_DIR, "Skeleton/Thorax/Sternum.obj")
STERNUM_OBJ_DST = os.path.join(ZYGOTE_DIR, "Skeleton/Sternum.obj")
STERNUM_OBJ_REL = "Zygote_Meshes_251229/Skeleton/Sternum.obj"


def fmt_v(arr):
    return " ".join(f"{x:.6f}" for x in np.asarray(arr).flatten())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", default="T50",
                    help="Parent body in DART skel (default T50)")
    ap.add_argument("--local-offset", default="-0.001 -0.120 -0.115",
                    help="Local offset in parent body frame (m), 3 floats")
    args = ap.parse_args()

    local_offset = np.array([float(x) for x in args.local_offset.split()],
                            dtype=np.float64)
    assert local_offset.shape == (3,)

    # Move OBJ file
    if not os.path.exists(STERNUM_OBJ_DST):
        if not os.path.exists(STERNUM_OBJ_SRC):
            sys.exit(f"Sternum source not found: {STERNUM_OBJ_SRC}")
        shutil.copy2(STERNUM_OBJ_SRC, STERNUM_OBJ_DST)
        print(f"Copied {STERNUM_OBJ_SRC} -> {STERNUM_OBJ_DST}")
    else:
        print(f"{STERNUM_OBJ_DST} already exists, skipping copy")

    # Load mesh + scale to meters
    tri = trimesh.load_mesh(STERNUM_OBJ_DST)
    verts = np.asarray(tri.vertices, dtype=np.float64) * 0.01
    mean = verts.mean(axis=0)
    # Use bbox geometric center (not vertex mean) as the body/joint anchor.
    # Sternum mesh has biased vertex density at the manubrium → mean is
    # 18mm above true geometric center, causing visual Y gap between box
    # (centered on body_t) and OBJ (rendered around joint_t).
    centroid = (verts.max(0) + verts.min(0)) / 2.0
    centered = verts - mean  # PCA uses mean-centered coords for stable axes
    # X-aligned OBB: x = world X (sagittal symmetry preserved), y/z from
    # PCA on YZ plane. body_r encodes the tilt (DART body box gets oriented
    # along these axes). joint_r is kept identity so the OBJ mesh stays in
    # its rest world pose under drawObj's joint-transform-based render.
    from numpy.linalg import eigh
    yz = centered[:, 1:]
    H = yz.T @ yz / len(yz)
    w, V = eigh(H)
    order = np.argsort(w)[::-1]
    V = V[:, order]
    axes = np.zeros((3, 3))
    axes[0] = np.array([1.0, 0.0, 0.0])
    pc1 = np.array([0.0, V[0, 0], V[1, 0]])
    if pc1[1] < 0:  # principal axis points +Y (up)
        pc1 = -pc1
    axes[1] = pc1
    pc2 = np.cross(axes[0], axes[1])
    if pc2[2] < 0:  # secondary axis points +Z (anterior)
        pc2 = -pc2
        axes[1] = np.cross(pc2, axes[0])
    axes[2] = pc2
    body_r_pca = axes.T  # columns = local-axis directions in world frame
    # Project verts onto PCA axes and find the BOX CENTER that tightly covers
    # all extents along each axis. Project around any reference, then shift
    # box center so projection range is symmetric in box-local frame.
    rel = verts - mean
    proj = rel @ body_r_pca
    proj_min = proj.min(0)
    proj_max = proj.max(0)
    size = proj_max - proj_min
    proj_mid = (proj_max + proj_min) / 2.0  # local-frame offset from `mean` to true box center
    # World-space box center = mean + R_pca @ proj_mid (since body_r_pca cols
    # are world-axes of local axes, world_offset = body_r_pca @ proj_mid).
    centroid = mean + body_r_pca @ proj_mid
    print(f"Sternum mean: {mean}")
    print(f"OBB world center: {centroid}")
    print(f"X-aligned PCA OBB size={size}")
    print(f"body_r=\n{body_r_pca}")

    # Build DART skel to get parent rest transform
    skel_info, root_name, *_ = saveSkeletonInfo(SKEL_XML)
    skel = buildFromInfo(skel_info, root_name)
    parent_node = skel.getBodyNode(args.parent)
    if parent_node is None:
        sys.exit(f"Parent body {args.parent} not in skel")
    skel.resetPositions()
    T_parent = np.asarray(parent_node.getWorldTransform().matrix(),
                          dtype=np.float64)
    R_par = T_parent[:3, :3]
    t_par = T_parent[:3, 3]
    # Pivot at rib 1 midpoint (manubrium superior border, sternal notch).
    # Upper sternum hugs rib 1 across motion; lower sternum swings to fit
    # ribs 2-10 via Kabsch rotation. body_t stays at bbox center so the OBB
    # encloses the OBJ snugly even though joint_t is offset upward.
    rib1_l_path = os.path.join(ZYGOTE_DIR, "Skeleton/L_Rib1.obj")
    rib1_r_path = os.path.join(ZYGOTE_DIR, "Skeleton/R_Rib1.obj")
    rib1_l_verts = np.asarray(trimesh.load_mesh(rib1_l_path).vertices,
                              dtype=np.float64) * 0.01
    rib1_r_verts = np.asarray(trimesh.load_mesh(rib1_r_path).vertices,
                              dtype=np.float64) * 0.01
    rib1_l_tip = rib1_l_verts[int(np.argmax(rib1_l_verts[:, 2]))]
    rib1_r_tip = rib1_r_verts[int(np.argmax(rib1_r_verts[:, 2]))]
    joint_t_world = (rib1_l_tip + rib1_r_tip) / 2.0
    local_offset = R_par.T @ (joint_t_world - t_par)
    print(f"Rib1 anchor (world, joint_t): {joint_t_world}")
    print(f"Override local_offset (joint at rib1 mid): {local_offset}")
    # joint_r = identity so the OBJ mesh, rendered via drawObj at joint
    # world transform, stays in its original Zygote rest orientation.
    # The BVH script compensates with R_local(f) = R_par_rest @ R_par(f).T @ R_target(f).
    joint_r = np.eye(3)
    print(f"Parent {args.parent} rest body_t (world): {t_par}")
    print(f"Computed joint_t world (sternum pivot): {joint_t_world}")
    print(f"joint_r (= parent body_r): {joint_r}")

    # Body transform: PCA-aligned rotation, body_t = mesh centroid (world)
    body_r = body_r_pca
    body_t = centroid

    # Mass ratio same as exportBoundingBoxes (0.5 kg per 0.11064*0.134131*0.059748)
    mass = float(np.prod(size) * 0.5 / (0.11064 * 0.134131 * 0.059748))

    # Patch XML
    tree = ET.parse(SKEL_XML)
    root = tree.getroot()
    # Remove existing Sternum0 if present (idempotent)
    for old in root.findall("Node[@name='Sternum0']"):
        root.remove(old)
    # saveSkeletonInfo requires parent-before-child ordering. Find insert
    # position: just before the first node whose parent_str == "Sternum0",
    # OR at end if no such child yet.
    children = list(root)
    insert_idx = len(children)
    for i, child in enumerate(children):
        if child.tag == "Node" and child.attrib.get("parent") == "Sternum0":
            insert_idx = i
            break
    node = ET.Element("Node", {"name": "Sternum0", "parent": args.parent})
    root.insert(insert_idx, node)
    print(f"Inserted Sternum0 at index {insert_idx}; child count now {len(list(root))}")
    body = ET.SubElement(node, "Body", {
        "type": "Box",
        "mass": f"{mass:.6f}",
        "size": fmt_v(size),
        "contact": "Off",
        "color": "0.9 0.9 0.9 1.0",
        "obj": STERNUM_OBJ_REL,
        "stretch": "0 0 0",
    })
    ET.SubElement(body, "Transformation", {
        "linear": fmt_v(body_r),
        "translation": fmt_v(body_t),
    })
    joint = ET.SubElement(node, "Joint", {
        "type": "Ball",
        "bvh": "Sternum",
        "lower": "-1.57 -1.57 -1.57",
        "upper": "1.57 1.57 1.57",
    })
    ET.SubElement(joint, "Transformation", {
        "linear": fmt_v(joint_r),
        "translation": fmt_v(joint_t_world),
    })

    # Write back (Python 3.8 lacks ET.indent; format manually for readability)
    def _indent(elem, level=0):
        i = "\n" + "\t" * level
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "\t"
            for child in elem:
                _indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
    _indent(root)
    tree.write(SKEL_XML)
    print(f"Patched {SKEL_XML}: added Sternum0 (parent={args.parent}, "
          f"bvh=\"Sternum\", joint_t={joint_t_world})")


if __name__ == "__main__":
    main()
