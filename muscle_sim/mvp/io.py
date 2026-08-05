"""MVP export helpers."""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_obj(path, vertices, faces=(), lines=(), object_name=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if object_name:
        rows.append("o " + object_name)
    rows.extend("v %.9g %.9g %.9g" % tuple(point)
                for point in np.asarray(vertices))
    rows.extend("f %d %d %d" % tuple(np.asarray(face) + 1)
                for face in np.asarray(faces, dtype=int))
    rows.extend("l " + " ".join(str(int(index) + 1) for index in line)
                for line in lines)
    path.write_text("\n".join(rows) + "\n")


def write_scene_obj(path, objects):
    """Write [(name, vertices, faces, lines), ...] into one inspectable OBJ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, offset = [], 0
    for name, vertices, faces, lines in objects:
        vertices = np.asarray(vertices)
        rows.append("o " + name)
        rows.extend("v %.9g %.9g %.9g" % tuple(point) for point in vertices)
        rows.extend("f %d %d %d" % tuple(np.asarray(face) + 1 + offset)
                    for face in np.asarray(faces, dtype=int))
        rows.extend("l " + " ".join(
            str(int(index) + 1 + offset) for index in line)
            for line in lines)
        offset += len(vertices)
    path.write_text("\n".join(rows) + "\n")

