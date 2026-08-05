import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from muscle_sim.mvp.io import write_json, write_scene_obj
from muscle_sim.mvp.scope import (
    LEFT_UPPER_LEG, build_neighbor_pairs, classify_joint_crossing,
    discover_upper_leg, infer_attachment_sets, wrapping_regions)
from muscle_sim.mvp.verify_preview_scope import verify


class MuscleAnimationMVPScopeTests(unittest.TestCase):
    def manifest(self):
        return {"muscles": [{
            "muscle_name": name, "usable_for_mvp": True,
            "surface_triangle_count": 1, "attachment_vertex_count": 2,
            "attachment_bones": ["pelvis", "femur"]}
            for name in LEFT_UPPER_LEG]}

    def test_full_left_upper_leg_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in LEFT_UPPER_LEG:
                (Path(directory) / f"{name}_tet.npz").touch()
            rows = discover_upper_leg(self.manifest(), directory)
            self.assertEqual(len(rows), 25)
            self.assertTrue(all(row["preview_eligibility"] for row in rows))

    def test_no_three_muscle_allowlist(self):
        self.assertIn("L_Rectus_Femoris", LEFT_UPPER_LEG)
        self.assertIn("L_Adductor_Magnus", LEFT_UPPER_LEG)
        self.assertGreater(len(LEFT_UPPER_LEG), 3)

    def test_per_muscle_failure_status_is_accountable(self):
        statuses = {"a": {"status": "SUCCESS"}, "b": {"status": "FAILED"}}
        self.assertEqual(set(statuses), {"a", "b"})

    def test_automatic_attachment_fallback(self):
        points = np.column_stack((np.linspace(0., 1., 20),
                                  np.zeros((20, 2))))
        first, last = infer_attachment_sets(points, np.arange(20), 3)
        self.assertLess(points[first, 0].max(), points[last, 0].min())

    def test_automatic_centerline_fallback_contract(self):
        first, last = infer_attachment_sets(
            np.column_stack((np.arange(8), np.zeros((8, 2)))),
            np.arange(8), 2)
        self.assertTrue(len(first) and len(last))

    def test_joint_crossing_classification(self):
        self.assertEqual(classify_joint_crossing("L_Rectus_Femoris"),
                         "HIP_AND_KNEE")
        self.assertEqual(classify_joint_crossing("L_Vastus_Lateralis"),
                         "KNEE_ONLY")
        self.assertEqual(classify_joint_crossing("L_Pectineus"), "HIP_ONLY")

    def test_region_specific_wrapping(self):
        self.assertIn("anterior_knee",
                      wrapping_regions("L_Rectus_Femoris"))
        self.assertNotIn("medial_knee", wrapping_regions("L_Pectineus"))

    def test_neighbor_pair_construction(self):
        muscles = [
            {"name": "a", "rest_vertices": np.array([[0, 0, 0], [1, 1, 1]])},
            {"name": "b", "rest_vertices": np.array([[.5, 0, 0], [2, 1, 1]])},
            {"name": "c", "rest_vertices": np.array([[9, 9, 9], [10, 10, 10]])}]
        self.assertEqual(build_neighbor_pairs(muscles, 0.), [("a", "b")])

    def test_full_scene_export(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.obj"
            write_scene_obj(path, [("muscle", np.zeros((3, 3)),
                                    [[0, 1, 2]], [])])
            self.assertIn("o muscle", path.read_text())

    def test_preview_scope_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in LEFT_UPPER_LEG:
                (root / f"{name}_tet.npz").touch()
            preview = root / "preview"
            write_json(preview / "mvp_preview_summary.json", {
                "muscle_root": str(root),
                "muscle_status": {
                    name: {"status": "SUCCESS"} for name in LEFT_UPPER_LEG}})
            result = verify(self.manifest(), preview, require_all=True)
            self.assertTrue(result["passed"])
            self.assertEqual(result["exported_count"], len(LEFT_UPPER_LEG))


if __name__ == "__main__":
    unittest.main()
