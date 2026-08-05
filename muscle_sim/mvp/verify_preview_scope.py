"""Verify that a full MVP preview accounts for every eligible muscle."""
import argparse
import json
from pathlib import Path

from muscle_sim.mvp.scope import discover_upper_leg


def verify(manifest, preview, require_all=False):
    summary = json.loads((Path(preview) / "mvp_preview_summary.json").read_text())
    expected = {
        row["canonical_name"] for row in discover_upper_leg(
            manifest, summary["muscle_root"]) if row["preview_eligibility"]}
    statuses = summary["muscle_status"]
    accounted = {
        name for name, row in statuses.items()
        if row["status"] in ("SUCCESS", "APPROXIMATE", "SKIPPED", "FAILED")}
    exported = {
        name for name, row in statuses.items()
        if row["status"] in ("SUCCESS", "APPROXIMATE")}
    missing = sorted(expected - accounted)
    result = {
        "eligible_count": len(expected), "exported_count": len(exported),
        "accounted_count": len(accounted & expected),
        "missing_without_status": missing,
        "passed": not missing}
    if require_all and missing:
        raise ValueError("eligible muscles missing without status: "
                         + ", ".join(missing))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--preview", required=True)
    parser.add_argument("--require-all-eligible-muscles",
                        action="store_true")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    result = verify(
        manifest, args.preview, args.require_all_eligible_muscles)
    path = Path(args.preview) / "preview_scope_verification.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()
