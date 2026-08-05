"""Export deterministic skeleton/bone previews for the configured poses."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from core.bvhparser import MyBVH
from tools.bake_contour_sim import (
    _detect_bvh_tframe, build_bone_trimeshes, compute_bone_rest_transforms,
    load_bone_meshes, load_skeleton)
from muscle_sim.mvp.io import write_json, write_scene_obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    config = yaml.safe_load(Path(args.config).read_text())[
        "muscle_animation_mvp"]
    skeleton, bvh_info, _ = load_skeleton()
    motion = MyBVH(manifest["bvh_path"], bvh_info, skeleton,
                   T_frame=_detect_bvh_tframe(manifest["bvh_path"]))
    bone_data = load_bone_meshes("L")
    rest = compute_bone_rest_transforms(skeleton, bone_data)
    output = Path(args.output)
    summary = []
    for label, frame in config["preview_frames"].items():
        frame = min(int(frame), len(motion.mocap_refs) - 1)
        skeleton.setPositions(motion.mocap_refs[frame].copy())
        bones = build_bone_trimeshes(skeleton, bone_data, rest)
        objects = [(f"bone_{index}", mesh.vertices, mesh.faces, [])
                   for index, mesh in enumerate(bones)]
        joints = np.asarray([
            skeleton.getBodyNode(i).getWorldTransform().translation()
            for i in range(skeleton.getNumBodyNodes())])
        objects.append(("joint_positions", joints, [], []))
        write_scene_obj(output / f"skeleton_{label}.obj", objects)
        summary.append({"pose": label, "bvh_frame": frame,
                        "joint_count": len(joints),
                        "bone_surface_count": len(bones)})
    write_json(output / "skeleton_preview_summary.json", {
        "bvh": manifest["bvh_path"], "poses": summary})
    print(output)


if __name__ == "__main__":
    main()

