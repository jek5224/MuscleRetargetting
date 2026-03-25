"""Test pyuipc with a single muscle tet mesh."""
import numpy as np
from uipc.geometry import tetmesh, label_surface, label_triangle_orient
from uipc.constitution import StableNeoHookean, SoftPositionConstraint, ElasticModuli
from uipc.unit import kPa
from uipc.core import Engine, World, Scene
from uipc import view, Animation
import uipc.builtin as builtin

d = np.load("tet/L_Biceps_Femoris_tet.npz", allow_pickle=True)
verts = d["vertices"].astype(np.float64) * 1000.0  # mm
tets = d["tetrahedra"].astype(np.int32)
anchors = d["anchor_vertices"]

# Fix tet orientation
v0 = verts[tets[:, 0]]
cross = np.cross(verts[tets[:, 1]] - v0, verts[tets[:, 2]] - v0)
vol = np.einsum("ij,ij->i", cross, verts[tets[:, 3]] - v0) / 6.0
neg = vol < 0
tets[neg, 1], tets[neg, 2] = tets[neg, 2].copy(), tets[neg, 1].copy()

# Recompute
v0 = verts[tets[:, 0]]
cross = np.cross(verts[tets[:, 1]] - v0, verts[tets[:, 2]] - v0)
vol = np.einsum("ij,ij->i", cross, verts[tets[:, 3]] - v0) / 6.0

# Remove degenerate (volume < 0.001 mm³)
good = vol > 0.001
removed = int(np.sum(~good))
tets = tets[good]

# Compact: remove unused vertices
used_verts = np.unique(tets.ravel())
vert_map = np.full(len(verts), -1, dtype=np.int32)
vert_map[used_verts] = np.arange(len(used_verts), dtype=np.int32)
verts = verts[used_verts]
tets = vert_map[tets]
# Remap anchors
anchors = np.array([vert_map[a] for a in anchors if vert_map[a] >= 0])
print(f"{len(tets)} tets, {len(verts)} verts, removed {removed} degenerate, "
      f"{len(anchors)} anchors", flush=True)

mesh = tetmesh(verts, tets)
label_surface(mesh)
label_triangle_orient(mesh)

snh = StableNeoHookean()
moduli = ElasticModuli.youngs_poisson(0.5 * kPa, 0.40)
snh.apply_to(mesh, moduli, mass_density=1060.0)
print("apply_to OK", flush=True)

spc = SoftPositionConstraint()
spc.apply_to(mesh, 1e4)

engine = Engine("cuda")
world = World(engine)
config = Scene.default_config()
config["dt"] = 0.01
config["gravity"] = [[0.0], [0.0], [0.0]]
config["sanity_check"] = {"enable": False}
scene = Scene(config)
scene.contact_tabular().default_model(0.0, 1e9)

obj = scene.objects().create("muscle")
geo_slot, _ = obj.geometries().create(mesh)

# Fix anchor vertices
def animate(info: Animation.UpdateInfo):
    geo = info.geo_slots()[0].geometry()
    rest_geo = info.rest_geo_slots()[0].geometry()
    cv = view(geo.vertices().find(builtin.is_constrained))
    av = view(geo.vertices().find(builtin.aim_position))
    rv = rest_geo.positions().view()
    for idx in anchors:
        if idx < len(cv):
            cv[idx] = 1
            av[idx] = rv[idx]

scene.animator().insert(obj, animate)
world.init(scene)
print("Init OK", flush=True)

for i in range(3):
    world.advance()
    world.retrieve()
    geo = geo_slot.geometry()
    pos = np.array(geo.positions().view()).reshape(-1, 3)

    # Check inversions on solved mesh
    v0c = pos[tets[:, 0]]
    crossc = np.cross(pos[tets[:, 1]] - v0c, pos[tets[:, 2]] - v0c)
    volc = np.einsum("ij,ij->i", crossc, pos[tets[:, 3]] - v0c) / 6.0
    n_inv = int(np.sum(volc <= 0))
    print(f"Frame {i}: inv={n_inv}/{len(tets)}, "
          f"vol=[{volc.min():.4f},{volc.max():.4f}] mm³", flush=True)

print("SUCCESS", flush=True)
