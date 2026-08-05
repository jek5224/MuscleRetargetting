"""Transactional gate for diagnostic-only isolated interior flips."""
import argparse
import json
from pathlib import Path

import yaml

from muscle_sim.diagnostic_original_asset import inspect_asset
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
    report = inspect_asset(
        selection["selected_tet_mesh_path"],
        int(settings["expected_fiber_count"]),
        int(settings["expected_source_sample_count"]))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if report["sliver_classification"] != "ISOLATED_SLIVER":
        result = {
            "modified": False, "input_asset": report["tet_mesh_path"],
            "failure_classification": (
                "LOCAL_FLIP_NOT_ALLOWED_FOR_"
                + report["sliver_classification"]),
            "reason": ("Only one isolated interior sliver may be repaired; "
                       "the input asset was retained byte-identically.")}
        write_json(output / "sliver_repair_report.json", result)
        raise ValueError(result["failure_classification"])
    raise NotImplementedError(
        "isolated candidate requires transactional 2-3/3-2/4-4 evaluation")


if __name__ == "__main__":
    main()
