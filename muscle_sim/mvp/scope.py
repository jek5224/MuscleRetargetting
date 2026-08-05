"""Scope and best-effort policies for the left upper-leg MVP."""
from pathlib import Path

import numpy as np


LEFT_UPPER_LEG = (
    "L_Adductor_Brevis", "L_Adductor_Longus", "L_Adductor_Magnus",
    "L_Biceps_Femoris", "L_Gluteus_Maximus", "L_Gluteus_Medius",
    "L_Gluteus_Minimus", "L_Gracilis", "L_Iliacus",
    "L_Inferior_Gemellus", "L_Obturator_Externus",
    "L_Obturator_Internus", "L_Pectineus", "L_Piriformis",
    "L_Popliteus", "L_Quadratus_Femoris", "L_Rectus_Femoris",
    "L_Sartorius", "L_Semimembranosus", "L_Semitendinosus",
    "L_Superior_Gemellus", "L_Tensor_Fascia_Lata",
    "L_Vastus_Intermedius", "L_Vastus_Lateralis", "L_Vastus_Medialis")


def discover_upper_leg(manifest, muscle_root):
    manifest_rows = {row["muscle_name"]: row for row in manifest["muscles"]}
    disk = {path.name.replace("_tet.npz", ""): path
            for path in Path(muscle_root).glob("L_*_tet.npz")}
    rows = []
    for name in LEFT_UPPER_LEG:
        row = dict(manifest_rows.get(name, {}))
        path = disk.get(name)
        row.update({
            "canonical_name": name,
            "source_asset_path": str(path) if path else row.get(
                "tet_mesh_path"),
            "load_status": "DISCOVERED" if path else "MISSING",
            "surface_extraction_status": (
                "AVAILABLE" if row.get("surface_triangle_count", 0) else
                "NOT_VALIDATED"),
            "attachment_source": (
                "ASSET_METADATA" if row.get("attachment_vertex_count", 0)
                and row.get("attachment_bones") else
                "AUTOMATIC_ENDPOINT_FALLBACK"),
            "centerline_source": (
                "VALID_FIBER_BUNDLE_OR_ENDPOINT_FALLBACK"
                if row.get("available_fiber_data") else
                "PRINCIPAL_AXIS_ENDPOINT_FALLBACK"),
            "joint_crossing": classify_joint_crossing(name),
            "wrapping_regions": wrapping_regions(name),
            "preview_eligibility": bool(
                path and row.get("usable_for_mvp", True)),
            "exclusion_reason": (
                None if path and row.get("usable_for_mvp", True)
                else "MISSING_OR_UNUSABLE_TET_ASSET")})
        rows.append(row)
    return rows


def classify_joint_crossing(name):
    if any(token in name for token in (
            "Gracilis", "Sartorius", "Semitendinosus",
            "Semimembranosus", "Biceps_Femoris", "Rectus_Femoris")):
        return "HIP_AND_KNEE"
    if any(token in name for token in (
            "Vastus_", "Popliteus")):
        return "KNEE_ONLY"
    if name.startswith("L_"):
        return "HIP_ONLY"
    return "UNKNOWN"


def wrapping_regions(name):
    result = []
    if any(token in name for token in (
            "Gracilis", "Sartorius", "Semitendinosus")):
        result.append("medial_knee")
    if any(token in name for token in (
            "Rectus_Femoris", "Vastus_")):
        result.append("anterior_knee")
    if any(token in name for token in (
            "Biceps_Femoris", "Semimembranosus", "Popliteus")):
        result.append("posterior_knee")
    if "Tensor_Fascia_Lata" in name:
        result.append("lateral_knee")
    if classify_joint_crossing(name) in ("HIP_ONLY", "HIP_AND_KNEE"):
        result.append("hip")
    return result


def infer_attachment_sets(vertices, surface_ids, count=16):
    vertices = np.asarray(vertices)
    ids = np.asarray(surface_ids, dtype=np.int64)
    cloud = vertices[ids]
    _, _, vt = np.linalg.svd(cloud - cloud.mean(axis=0), full_matrices=False)
    coordinate = (cloud - cloud.mean(axis=0)) @ vt[0]
    count = max(1, min(count, len(ids) // 4))
    order = np.argsort(coordinate)
    return ids[order[:count]], ids[order[-count:]]


def build_neighbor_pairs(muscles, padding=.012):
    boxes = {}
    for muscle in muscles:
        vertices = np.asarray(muscle["rest_vertices"])
        boxes[muscle["name"]] = (vertices.min(axis=0), vertices.max(axis=0))
    pairs = []
    names = sorted(boxes)
    for i, first in enumerate(names):
        amin, amax = boxes[first]
        for second in names[i + 1:]:
            bmin, bmax = boxes[second]
            separated = np.any(
                amax + padding < bmin) or np.any(bmax + padding < amin)
            if not separated:
                pairs.append((first, second))
    return pairs

