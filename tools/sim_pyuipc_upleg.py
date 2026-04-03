"""
pyuipc collision-aware simulation for L/R upper leg muscles.

Loads all UpLeg tet meshes, sets up IPC contact between all muscles,
fixes anchor vertices, runs quasistatic simulation.

Usage (on A6000 server):
    python tools/sim_pyuipc_upleg.py [--frames 10] [--tet-dir tet_sim]
"""
import argparse
import glob
import json
import os
import pickle
import time

import numpy as np

from uipc import view
from uipc.core import Engine, World, Scene
from uipc.geometry import tetmesh, label_surface, label_triangle_orient
from uipc.constitution import StableNeoHookean, ElasticModuli
from uipc.unit import kPa


SCALE = 1000.0  # m → mm

UPLEG_MUSCLES = [
    "Adductor_Brevis", "Adductor_Longus", "Adductor_Magnus",
    "Biceps_Femoris", "Gluteus_Maximus", "Gluteus_Medius",
    "Gluteus_Minimus", "Gracilis", "Iliacus",
    "Inferior_Gemellus", "Obturator_Externus", "Obturator_Internus",
    "Pectineus", "Piriformis", "Popliteus",
    "Quadratus_Femoris", "Rectus_Femoris", "Sartorius",
    "Semimembranosus", "Semitendinosus", "Superior_Gemellus",
    "Tensor_Fascia_Lata", "Vastus_Intermedius", "Vastus_Lateralis",
    "Vastus_Medialis",
]


def load_muscle(tet_dir, name):
    """Load a single muscle tet mesh. Returns dict or None."""
    path = os.path.join(tet_dir, f"{name}_tet.npz")
    if not os.path.exists(path):
        print(f"  [SKIP] {name}: file not found")
        return None

    with open(path, 'rb') as f:
        data = pickle.load(f)

    verts = data['vertices'].astype(np.float64) * SCALE
    tets = data['tetrahedra'].astype(np.int32)

    # Fix tet orientation
    v0 = verts[tets[:, 0]]
    cross = np.cross(verts[tets[:, 1]] - v0, verts[tets[:, 2]] - v0)
    vol = np.einsum('ij,ij->i', cross, verts[tets[:, 3]] - v0) / 6.0

    neg = vol < 0
    if np.any(neg):
        tets[neg, 1], tets[neg, 2] = tets[neg, 2].copy(), tets[neg, 1].copy()

    # Remove degenerate tets
    v0 = verts[tets[:, 0]]
    cross = np.cross(verts[tets[:, 1]] - v0, verts[tets[:, 2]] - v0)
    vol = np.einsum('ij,ij->i', cross, verts[tets[:, 3]] - v0) / 6.0
    good = vol > 0.01  # > 0.01 mm³
    if not np.all(good):
        n_removed = int(np.sum(~good))
        print(f"  [{name}] Removed {n_removed} degenerate tets")
        tets = tets[good]

    # Collect fixed vertices before remap
    anchors = data.get('anchor_vertices', np.array([], dtype=np.int32))
    cap_faces = data.get('cap_face_indices', [])
    fixed_verts = set()
    sim_faces = data.get('sim_faces', data.get('faces'))
    if sim_faces is not None and len(cap_faces) > 0:
        for fi in cap_faces:
            if fi < len(sim_faces):
                for vi in sim_faces[fi]:
                    fixed_verts.add(int(vi))
    for vi in anchors:
        fixed_verts.add(int(vi))

    # Remove unused vertices (pyuipc preconditioner fails with zero-row vertices)
    used = np.unique(tets.ravel())
    if len(used) < len(verts):
        remap = np.full(len(verts), -1, dtype=np.int32)
        remap[used] = np.arange(len(used), dtype=np.int32)
        verts = verts[used]
        tets = remap[tets]
        # Remap fixed vertices
        fixed_verts = {int(remap[vi]) for vi in fixed_verts if vi < len(remap) and remap[vi] >= 0}

    return {
        'name': name,
        'vertices': verts,
        'tetrahedra': tets,
        'fixed_vertices': sorted(fixed_verts),
        'n_tets': len(tets),
        'n_verts': len(verts),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--tet-dir', default='tet_sim')
    parser.add_argument('--sides', default='LR', help='L, R, or LR')
    args = parser.parse_args()

    # Load muscles
    muscles = []
    for side in args.sides:
        for muscle_name in UPLEG_MUSCLES:
            name = f"{side}_{muscle_name}"
            data = load_muscle(args.tet_dir, name)
            if data is not None:
                muscles.append(data)

    if not muscles:
        print("No muscles loaded!")
        return

    total_tets = sum(m['n_tets'] for m in muscles)
    total_verts = sum(m['n_verts'] for m in muscles)
    total_fixed = sum(len(m['fixed_vertices']) for m in muscles)
    print(f"\nLoaded {len(muscles)} muscles: {total_verts} verts, {total_tets} tets, {total_fixed} fixed")

    # Setup pyuipc
    engine = Engine('cuda')
    world = World(engine)

    config = Scene.default_config()
    config['dt'] = 0.01
    config['gravity'] = [[0.0], [0.0], [0.0]]  # quasistatic
    config['contact']['friction']['enable'] = False
    config['contact']['d_hat'] = 0.1  # 0.1mm contact distance (reduce initial contacts)
    config['sanity_check'] = {'enable': False}
    scene = Scene(config)

    # Contact: all muscles collide with each other
    from uipc.unit import GPa
    scene.contact_tabular().default_model(0.0, 1.0 * GPa)

    # Material
    from uipc.constitution import SoftPositionConstraint
    from uipc import Animation
    import uipc.builtin as builtin
    snh = StableNeoHookean()
    spc = SoftPositionConstraint()
    moduli = ElasticModuli.youngs_poisson(0.5 * kPa, 0.40)

    # Create muscle objects
    geo_slots = []
    muscle_info = []  # (name, tets, fixed_verts, vert_offset)

    for m in muscles:
        mesh = tetmesh(m['vertices'], m['tetrahedra'])
        label_surface(mesh)
        label_triangle_orient(mesh)

        snh.apply_to(mesh, moduli, mass_density=1060.0)
        if len(m['fixed_vertices']) > 0:
            spc.apply_to(mesh, 1e4)

        obj = scene.objects().create(m['name'])
        geo_slot, _ = obj.geometries().create(mesh)
        geo_slots.append(geo_slot)

        muscle_info.append({
            'name': m['name'],
            'tets': m['tetrahedra'],
            'fixed': m['fixed_vertices'],
            'geo_slot': geo_slot,
        })

        # Animator: fix anchor vertices at rest position
        fixed = m['fixed_vertices']
        if len(fixed) > 0:
            def make_animate(fixed_verts):
                def animate(info: Animation.UpdateInfo):
                    geo = info.geo_slots()[0].geometry()
                    rest_geo = info.rest_geo_slots()[0].geometry()
                    cv = view(geo.vertices().find(builtin.is_constrained))
                    av = view(geo.vertices().find(builtin.aim_position))
                    rv = rest_geo.positions().view()
                    for idx in fixed_verts:
                        if idx < len(cv):
                            cv[idx] = 1
                            av[idx] = rv[idx]
                return animate
            scene.animator().insert(obj, make_animate(fixed))

        print(f"  {m['name']}: {m['n_verts']} verts, {m['n_tets']} tets, {len(m['fixed_vertices'])} fixed")

    # Init
    print(f"\nInitializing pyuipc ({len(muscles)} objects)...")
    t0 = time.time()
    world.init(scene)
    print(f"Init done in {time.time()-t0:.1f}s")

    # Run
    print(f"\nRunning {args.frames} frames...")
    for frame in range(args.frames):
        t0 = time.time()
        world.advance()
        world.retrieve()
        dt = time.time() - t0

        # Check inversions across all muscles
        total_inv = 0
        for mi in muscle_info:
            geo = mi['geo_slot'].geometry()
            pos = np.array(geo.positions().view()).reshape(-1, 3)
            tets = mi['tets']
            if len(tets) == 0:
                continue
            v0 = pos[tets[:, 0]]
            cross = np.cross(pos[tets[:, 1]] - v0, pos[tets[:, 2]] - v0)
            vol = np.einsum('ij,ij->i', cross, pos[tets[:, 3]] - v0) / 6.0
            n_inv = int(np.sum(vol <= 0))
            total_inv += n_inv

        print(f"  Frame {frame}: {dt:.2f}s, inversions={total_inv}/{total_tets}")

    print(f"\nDone. {len(muscles)} muscles, {args.frames} frames.")


if __name__ == '__main__':
    main()
