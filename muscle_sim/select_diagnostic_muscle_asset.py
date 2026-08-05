"""Select a pre-C3 fiber-compatible diagnostic asset by validation."""
import argparse
from pathlib import Path

import yaml

from muscle_sim.diagnostic_original_asset import inspect_asset, sha256
from muscle_sim.fiber_repair import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--muscle", required=True)
    parser.add_argument("--search-roots", nargs="+", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    settings = config["diagnostic_original_asset"]
    side_prefix = (
        "L_" if args.muscle.startswith("Left_") else
        "R_" if args.muscle.startswith("Right_") else "")
    muscle_stem = args.muscle.split("_", 1)[-1]
    candidates = []
    for root_text in args.search_roots:
        root = Path(root_text)
        if not root.exists():
            candidates.append({
                "tet_mesh_path": str(root),
                "available": False, "rejection_reason": "MISSING_SEARCH_ROOT"})
            continue
        paths = ([root] if root.is_file() else sorted(
            root.rglob("*Semitendinosus*tet.npz")))
        for path in paths:
            if (side_prefix and not path.name.startswith(side_prefix)
                    or muscle_stem not in path.name):
                continue
            try:
                row = inspect_asset(
                    path, int(settings["expected_fiber_count"]),
                    int(settings["expected_source_sample_count"]))
                row["available"] = True
            except Exception as error:
                row = {
                    "tet_mesh_path": str(path), "available": True,
                    "valid_for_diagnostic_fem": False,
                    "rejection_reason": f"{type(error).__name__}: {error}"}
            candidates.append(row)
    ranked = sorted(candidates, key=lambda row: (
        not row.get("valid_for_diagnostic_fem", False),
        not row.get("fiber_signature_matches", False),
        not row.get("attachment_metadata_exists", False),
        row.get("outside_sample_count", 10 ** 9),
        row.get("tets_below_two_degrees", 10 ** 9),
        row.get("tet_mesh_path", "")))
    selected = ranked[0] if ranked else None
    approved = bool(selected and selected.get("valid_for_diagnostic_fem"))
    material = Path(settings["material_metadata_path"])
    report = {
        "muscle": args.muscle,
        "selection_policy": settings["asset_selection"],
        "selected_tet_mesh_path": (
            selected["tet_mesh_path"] if selected else None),
        "selected_fiber_path": (
            selected.get("fiber_path") if selected else None),
        "selected_attachment_metadata_path": (
            selected.get("attachment_metadata_path") if selected else None),
        "selected_material_metadata_path": str(material),
        "material_metadata_sha256": sha256(material),
        "approved_for_diagnostic_fem": approved,
        "selection_failure_classification": (
            None if approved else
            "NO_FIBER_COMPATIBLE_DIAGNOSTIC_ASSET"),
        "selected_asset": selected, "candidates": ranked}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "diagnostic_asset_selection.json", report)
    print(output / "diagnostic_asset_selection.json")
    if not approved:
        raise ValueError(
            "no candidate preserves the required fibers and supports FEM")


if __name__ == "__main__":
    main()
