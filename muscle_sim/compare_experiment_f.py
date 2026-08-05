"""Load and compare original/repaired Experiment F reports."""
import argparse
import json
from pathlib import Path


FIELDS = (
    "initial_exact_penetration", "final_exact_penetration",
    "solver_proxy_penetration", "mechanical_residual",
    "complementarity_residual", "minimum_J", "active_contact_count",
    "maximum_correction_displacement", "rms_correction_displacement",
    "volume_ratio", "attachment_error", "fiber_stretch",
    "fiber_curvature")


def load_result(directory):
    directory = Path(directory)
    report = json.loads((directory / "rest_correction_report.json").read_text())
    result = {"path": str(directory), "accepted": bool(report.get("accepted"))}
    for field in FIELDS:
        result[field] = report.get(field)
    result["acceptance_criteria"] = report.get("acceptance_criteria", {})
    return result


def compare(original, repaired):
    first, second = load_result(original), load_result(repaired)
    if first["acceptance_criteria"] != second["acceptance_criteria"]:
        raise ValueError("Experiment F acceptance thresholds differ")
    return {"original": first, "repaired": second,
            "same_acceptance_thresholds": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--repaired", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(
        compare(args.original, args.repaired), indent=2))


if __name__ == "__main__":
    main()
