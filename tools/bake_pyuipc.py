"""
Minimal pyuipc test: single muscle tet mesh with IPC contact.

Runs on A6000 server (Python 3.10, CUDA 12.8, pyuipc installed).
Tests: Stable Neo-Hookean, hard attachment BCs, IPC contact, zero inversions.

Usage:
    python tools/bake_pyuipc.py
"""
import os
import sys
import time
import numpy as np

from uipc import view, Animation, Vector3
import uipc.builtin as builtin
from uipc.core import Engine, World, Scene
from uipc.geometry import (
    tetmesh, label_surface, label_triangle_orient,
    flip_inward_triangles,
)
from uipc.constitution import StableNeoHookean, SoftPositionConstraint, ElasticModuli
from uipc.unit import GPa, Pa


def main():
    # Load a single muscle tet mesh
    tet_path = 'tet/L_Biceps_Femoris_tet.npz'
    if not os.path.exists(tet_path):
        print(f"File not found: {tet_path}")
        return

    d = np.load(tet_path, allow_pickle=True)
    verts = d['vertices'].astype(np.float64)
    tets = d['tetrahedra'].astype(np.int32)
    anchors = d['anchor_vertices']  # indices of attachment vertices
    print(f"Loaded: {len(verts)} verts, {len(tets)} tets, {len(anchors)} anchor verts", flush=True)

    # --- pyuipc setup ---
    engine = Engine('cuda')
    world = World(engine)

    config = Scene.default_config()
    config['dt'] = 0.01
    config['gravity'] = [[0.0], [0.0], [0.0]]  # quasistatic
    config['contact']['friction']['enable'] = False
    config['contact']['d_hat'] = 0.002
    scene = Scene(config)

    scene.contact_tabular().default_model(0.0, 1.0 * GPa)

    # Material
    snh = StableNeoHookean()
    spc = SoftPositionConstraint()
    # Material in mm units: E = 500 Pa = 0.5 kPa = 0.0005 MPa
    from uipc.unit import kPa
    moduli = ElasticModuli.youngs_poisson(0.5 * kPa, 0.40)

    # Scale to mm (pyuipc works better with larger coordinates)
    # Our mesh is in meters, volumes are ~1e-5. In mm, volumes are ~1e4.
    SCALE = 1000.0  # m → mm
    verts = verts * SCALE

    # Fix tet orientation for pyuipc.
    # Try the convention from libuipc source: uses (v1-v0) x (v2-v0) . (v3-v0)
    # which is the scalar triple product = 6 * signed volume
    v0 = verts[tets[:, 0]]
    e01 = verts[tets[:, 1]] - v0
    e02 = verts[tets[:, 2]] - v0
    e03 = verts[tets[:, 3]] - v0
    # Scalar triple product: (e01 x e02) . e03
    cross = np.cross(e01, e02)
    vol = np.einsum('ij,ij->i', cross, e03) / 6.0
    print(f"  Raw volumes: min={vol.min():.8e}, max={vol.max():.8e}, "
          f"neg={int(np.sum(vol < 0))}, zero={int(np.sum(np.abs(vol) < 1e-14))}", flush=True)

    # Flip negative-volume tets (swap v1 and v2)
    neg = vol < 0
    if np.any(neg):
        n_neg = int(np.sum(neg))
        print(f"  Flipping {n_neg} tets", flush=True)
        tets[neg, 1], tets[neg, 2] = tets[neg, 2].copy(), tets[neg, 1].copy()

    # Recompute volumes
    v0 = verts[tets[:, 0]]
    e1 = verts[tets[:, 1]] - v0
    e2 = verts[tets[:, 2]] - v0
    e3 = verts[tets[:, 3]] - v0
    vol = np.linalg.det(np.stack([e1, e2, e3], axis=-1)) / 6.0

    # Remove tets with volume <= 0 (truly degenerate)
    # Threshold scales with cube of SCALE (volume = length^3)
    good = vol > 0.01  # remove tets with volume < 0.01 mm³ (or 1e-11 m³)
    n_removed = int(np.sum(~good))
    if n_removed > 0:
        print(f"  Removing {n_removed} degenerate tets", flush=True)
        tets = tets[good]
        vol = vol[good]

    print(f"  Final: {len(tets)} tets, vol range=[{vol.min():.8e}, {vol.max():.8e}]", flush=True)

    mesh = tetmesh(verts, tets)
    label_surface(mesh)
    label_triangle_orient(mesh)
    # Don't flip — we already fixed tet orientation above

    snh.apply_to(mesh, moduli, mass_density=1060.0)
    spc.apply_to(mesh, 1e6)  # very stiff position constraint

    obj = scene.objects().create('muscle')
    geo_slot, rest_geo_slot = obj.geometries().create(mesh)

    # Animation: fix anchor vertices, move them slightly to test
    def animate(info: Animation.UpdateInfo):
        geo = info.geo_slots()[0].geometry()
        rest_geo = info.rest_geo_slots()[0].geometry()

        is_constrained = geo.vertices().find(builtin.is_constrained)
        cv = view(is_constrained)

        aim_pos = geo.vertices().find(builtin.aim_position)
        aim_view = view(aim_pos)
        rest_view = rest_geo.positions().view()

        # Fix anchor vertices and move them 1cm in Y
        frame = info.frame()
        dy = 0.01 * min(frame, 10)  # ramp up over 10 frames

        for idx in anchors:
            cv[idx] = 1
            aim_view[idx] = rest_view[idx] + np.array([0.0, dy, 0.0])

    animator = scene.animator()
    animator.insert(obj, animate)

    # Init and run
    print("\nInitializing pyuipc world...")
    world.init(scene)

    print("Running 10 frames...")
    for frame in range(10):
        t0 = time.time()
        world.advance()
        world.retrieve()
        dt = time.time() - t0

        # Read positions
        geo = geo_slot.geometry()
        pos = np.array(geo.positions().view())

        # Check inversions
        x3 = pos[tets[:, 3]]
        Ds = np.stack([pos[tets[:, 0]] - x3, pos[tets[:, 1]] - x3,
                       pos[tets[:, 2]] - x3], axis=-1)
        J = np.linalg.det(Ds)
        n_inv = int(np.sum(J <= 0))

        # Check anchor positions
        anchor_err = np.linalg.norm(
            pos[anchors] - (verts[anchors] + np.array([0, 0.01 * min(frame+1, 10), 0])),
            axis=1
        )

        print(f"  Frame {frame}: {dt:.2f}s, inv={n_inv}/{len(tets)}, "
              f"J=[{J.min():.4f},{J.max():.4f}], "
              f"anchor_err max={anchor_err.max()*1000:.2f}mm")

    print("\nDone.")


if __name__ == '__main__':
    main()
