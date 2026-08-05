"""Independent validation gate before isolated diagnostic FEM."""
import argparse
import json
from pathlib import Path

import yaml

from muscle_sim.diagnostic_original_asset import (
    diagnostic_decision, inspect_asset)
from muscle_sim.fiber_repair import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    selection = json.loads(Path(args.selection).read_text())
    config = yaml.safe_load(Path(args.config).read_text())
    settings = config["diagnostic_original_asset"]
    path = selection["selected_tet_mesh_path"]
    validation = inspect_asset(
        path, int(settings["expected_fiber_count"]),
        int(settings["expected_source_sample_count"]))
    hard = float(settings["validation"]["hard_minimum_dihedral_degrees"])
    validation["low_quality_gate"] = (
        validation["minimum_dihedral_degrees"] >= hard)
    validation["isolated_flip_allowed"] = (
        validation["sliver_classification"] == "ISOLATED_SLIVER"
        and settings["allow_isolated_interior_flips"])
    passed = bool(
        selection["approved_for_diagnostic_fem"]
        and validation["valid_for_diagnostic_fem"]
        and (validation["low_quality_gate"]
             or validation["isolated_flip_allowed"]))
    report = {
        "passed": passed, "validation": validation,
        "blocking_defect": None if passed else (
            "GLOBAL_LOW_QUALITY_AND_FIBER_EMBEDDING_FAILURE"
            if validation["sliver_classification"] == "GLOBAL_LOW_QUALITY"
            and not validation["embedding_success"]
            else "DIAGNOSTIC_ASSET_VALIDATION_FAILURE"),
        "decision": diagnostic_decision(
            isolated_failure=None if passed else "asset_validation"),
        "isolated_fem": {
            "run": False,
            "reason": None if passed else
            "BLOCKED_BY_DIAGNOSTIC_ASSET_VALIDATION"},
        "diagnostic_experiment_F_D0": {
            "run": False,
            "reason": None if passed else
            "BLOCKED_BY_DIAGNOSTIC_ASSET_VALIDATION"},
        "solver_or_material_parameters_changed": False}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "diagnostic_asset_validation.json", report)
    print(output / "diagnostic_asset_validation.json")
    if not passed:
        raise ValueError(report["blocking_defect"])


if __name__ == "__main__":
    main()
