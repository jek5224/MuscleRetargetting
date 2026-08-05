#!/usr/bin/env python3
"""Build viewer-loadable DTI subject meshes from Models/Geometry VTP files.

The source VTP files are in millimeters in the DTI/OpenSim frame.
The viewer MeshLoader multiplies OBJ vertices by 0.01, so this script writes
centimeter-like OBJ coordinates after applying an axis-aligned registration
fit from DTI VTP mesh centers to existing Zygote mesh centers:

    viewer_x = -0.0011057564 * raw_z - 0.0076650302
    viewer_y =  0.0010472816 * raw_y + 0.8893543347
    viewer_z =  0.0006579641 * raw_x + 0.0072980839

Then divided by 0.01 for MeshLoader's scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import vtk


FIT_X_FROM_Z = (-0.0011057564427281645, -0.007665030191720255)
FIT_Y_FROM_Y = (0.0010472816196888395, 0.8893543347025546)
FIT_Z_FROM_X = (0.0006579641046280251, 0.007298083940845146)
VIEWER_TO_OBJ = 100.0  # MeshLoader later multiplies OBJ vertices by 0.01.


NAME_MAP = {
    "AB": "Adductor_Brevis",
    "AL": "Adductor_Longus",
    "AM": "Adductor_Magnus",
    "BFL": "Biceps_Femoris_Long_Head",
    "BFS": "Biceps_Femoris_Short_Head",
    "EDL": "Extensor_Digitorum_Longus",
    "EHL": "Extensor_Hallucis_Longus",
    "FDB": "Flexor_Digitorum_Brevis",
    "FDL": "Flexor_Digitorum_Longus",
    "FHB": "Flexor_Hallucis_Brevis",
    "FHL": "Flexor_Hallucis",
    "GRA": "Gracilis",
    "LG": "Gastrocnemius_Lateral",
    "MG": "Gastrocnemius_Medial",
    "PB": "Peroneus_Brevis",
    "PL": "Peroneus_Longus",
    "POP": "Popliteus",
    "QP": "Quadratus_Plantae",
    "RF": "Rectus_Femoris",
    "SAR": "Sartorius",
    "SM": "Semimembranosus",
    "SOL": "Soleus",
    "ST": "Semitendinosus",
    "TA": "Tibialis_Anterior",
    "TP": "Tibialis_Posterior",
    "VI": "Vastus_Intermedius",
    "VL": "Vastus_Lateralis",
    "VM": "Vastus_Medialis",
    "AHB": "Abductor_Hallucis",
    "ADM": "Abductor_Digiti_Minimi_Foot",
    "Abductor_hallucis": "Abductor_Hallucis",
    "abductor_hallucis": "Abductor_Hallucis",
}


SKIP_PREFIXES = (
    "Karl_",
    "pelvis",
    "femur",
    "tibia",
    "fibula",
    "patella",
    "talus",
    "Talus",
    "Calc",
    "calcaneus",
    "Forefoot",
    "forefoot",
    "Midfoot",
    "midfoot",
    "Toes",
    "toes",
    "Thigh_flesh",
    "thigh_flesh",
    "Leg_flesh",
    "Leg flesh",
    "hip_flesh",
    "TaloCalc",
)


def split_source_name(stem: str) -> tuple[str, str] | None:
    side = "R"
    base = stem
    if stem.endswith("_L"):
        side = "L"
        base = stem[:-2]
    elif stem.endswith("_2"):
        side = "R"
        base = stem[:-2]
    elif stem.endswith("_2_1"):
        side = "L"
        base = stem[:-4]

    if base.endswith("_tendon"):
        return None
    if base not in NAME_MAP:
        return None
    return side, NAME_MAP[base]


def read_polydata(path: Path) -> vtk.vtkPolyData:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputConnection(reader.GetOutputPort())
    cleaner.Update()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(cleaner.GetOutputPort())
    tri.Update()
    return tri.GetOutput()


def transformed_points(poly: vtk.vtkPolyData) -> list[tuple[float, float, float]]:
    pts = poly.GetPoints()
    out = []
    for i in range(pts.GetNumberOfPoints()):
        x, y, z = pts.GetPoint(i)
        vx = FIT_X_FROM_Z[0] * z + FIT_X_FROM_Z[1]
        vy = FIT_Y_FROM_Y[0] * y + FIT_Y_FROM_Y[1]
        vz = FIT_Z_FROM_X[0] * x + FIT_Z_FROM_X[1]
        out.append((vx * VIEWER_TO_OBJ, vy * VIEWER_TO_OBJ, vz * VIEWER_TO_OBJ))
    return out


def faces(poly: vtk.vtkPolyData) -> list[tuple[int, int, int]]:
    result = []
    polys = poly.GetPolys()
    polys.InitTraversal()
    ids = vtk.vtkIdList()
    while polys.GetNextCell(ids):
        if ids.GetNumberOfIds() != 3:
            continue
        result.append((ids.GetId(0) + 1, ids.GetId(1) + 1, ids.GetId(2) + 1))
    return result


def write_obj(path: Path, verts: list[tuple[float, float, float]], tris: list[tuple[int, int, int]]) -> None:
    with path.open("w", encoding="ascii") as f:
        f.write("# Built from DTI Subject 01 Models/Geometry VTP\n")
        for x, y, z in verts:
            f.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for a, b, c in tris:
            f.write(f"f {a} {b} {c}\n")


def load_transformed_mesh(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    poly = read_polydata(path)
    return transformed_points(poly), faces(poly)


def write_merged(src: Path, out: Path, side: str, dst_name: str, parts: list[str]) -> str | None:
    all_verts: list[tuple[float, float, float]] = []
    all_faces: list[tuple[int, int, int]] = []
    for part in parts:
        suffix = "_L" if side == "L" else ""
        path = src / f"{part}{suffix}.vtp"
        if side == "R" and not path.exists():
            alt = {"RF": "RF_2", "MG": "MG_2"}.get(part)
            if alt:
                path = src / f"{alt}.vtp"
        if not path.exists():
            return None
        verts, tris = load_transformed_mesh(path)
        offset = len(all_verts)
        all_verts.extend(verts)
        all_faces.extend((a + offset, b + offset, c + offset) for a, b, c in tris)
    dst = out / f"{side}_{dst_name}.obj"
    write_obj(dst, all_verts, all_faces)
    return dst.name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="DTI/Subject 01/Models/Geometry")
    parser.add_argument("--out", default="DTI/Subject 01/BuiltMeshes")
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    written = []
    for vtp in sorted(src.glob("*.vtp")):
        stem = vtp.stem
        if stem.startswith(SKIP_PREFIXES):
            continue
        mapped = split_source_name(stem)
        if mapped is None:
            continue
        side, zygote_name = mapped
        dst = out / f"{side}_{zygote_name}.obj"
        poly = read_polydata(vtp)
        verts = transformed_points(poly)
        tris = faces(poly)
        if not verts or not tris:
            continue
        write_obj(dst, verts, tris)
        written.append(dst.name)

    for side in ("L", "R"):
        for merged in (
            write_merged(src, out, side, "Biceps_Femoris", ["BFL", "BFS"]),
            write_merged(src, out, side, "Gastrocnemius", ["LG", "MG"]),
        ):
            if merged is not None:
                written.append(merged)

    for name in sorted(set(written)):
        print(name)
    print(f"Wrote {len(set(written))} meshes to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
