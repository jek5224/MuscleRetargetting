#!/usr/bin/env python3
"""Inventory DICOM series without external DICOM dependencies.

This is intentionally small: it reads explicit/implicit little-endian headers
well enough for the Siemens MR files in this workspace and stops before pixel
data. It is not a general replacement for pydicom/dcm2niix.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path


LONG_VR = {"OB", "OD", "OF", "OL", "OW", "SQ", "UC", "UR", "UT", "UN"}

TAGS = {
    (0x0008, 0x0008): "image_type",
    (0x0008, 0x1030): "study_description",
    (0x0008, 0x103E): "series_description",
    (0x0018, 0x0020): "scanning_sequence",
    (0x0018, 0x0021): "sequence_variant",
    (0x0018, 0x0023): "mr_acquisition_type",
    (0x0018, 0x0024): "sequence_name",
    (0x0018, 0x0050): "slice_thickness",
    (0x0018, 0x0080): "repetition_time",
    (0x0018, 0x0081): "echo_time",
    (0x0018, 0x0087): "magnetic_field_strength",
    (0x0018, 0x1020): "software_versions",
    (0x0018, 0x1030): "protocol_name",
    (0x0018, 0x1314): "flip_angle",
    (0x0018, 0x1312): "in_plane_phase_encoding_direction",
    (0x0018, 0x5100): "patient_position",
    (0x0020, 0x000D): "study_instance_uid",
    (0x0020, 0x000E): "series_instance_uid",
    (0x0020, 0x0011): "series_number",
    (0x0020, 0x0013): "instance_number",
    (0x0020, 0x0032): "image_position_patient",
    (0x0020, 0x0037): "image_orientation_patient",
    (0x0020, 0x1041): "slice_location",
    (0x0028, 0x0010): "rows",
    (0x0028, 0x0011): "columns",
    (0x0028, 0x0030): "pixel_spacing",
    (0x0051, 0x100B): "siemens_mosaic_matrix",
}


def clean_text(raw: bytes) -> str:
    return raw.rstrip(b" \0").decode("latin1", errors="replace")


def printable_text(data: bytes) -> str:
    chunks = re.findall(rb"[ -~]{3,}", data)
    return "\n".join(chunk.decode("latin1", errors="replace") for chunk in chunks)


def parse_numbers(value: str) -> list[float]:
    out = []
    for part in value.replace(",", "\\").split("\\"):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            pass
    return out


def value_to_python(vr: str | None, raw: bytes):
    if vr in {"US"} and len(raw) >= 2:
        vals = [struct.unpack_from("<H", raw, i)[0] for i in range(0, len(raw) - 1, 2)]
        return vals[0] if len(vals) == 1 else vals
    if vr in {"SS"} and len(raw) >= 2:
        vals = [struct.unpack_from("<h", raw, i)[0] for i in range(0, len(raw) - 1, 2)]
        return vals[0] if len(vals) == 1 else vals
    if vr in {"UL"} and len(raw) >= 4:
        vals = [struct.unpack_from("<I", raw, i)[0] for i in range(0, len(raw) - 3, 4)]
        return vals[0] if len(vals) == 1 else vals
    text = clean_text(raw)
    if vr in {"DS", "IS"}:
        nums = parse_numbers(text)
        return nums[0] if len(nums) == 1 else nums
    return text


def extract_siemens_diffusion_meta(text: str) -> dict:
    """Pull common Siemens CSA diffusion fields from a printable text dump."""
    out = {}
    head = text.split("### ASCCONV BEGIN", 1)[0]
    lines = [line.strip() for line in head.splitlines()]

    def numbers_after_key(key: str, n: int, max_scan: int = 12) -> list[float]:
        for i, line in enumerate(lines):
            if line != key:
                continue
            values = []
            for candidate in lines[i + 1:i + 1 + max_scan]:
                try:
                    values.append(float(candidate))
                except ValueError:
                    continue
                if len(values) == n:
                    return values
        return []

    b_value = numbers_after_key("B_value", 1)
    if b_value:
        value = b_value[0]
        out["b_value"] = int(value) if value.is_integer() else value

    mosaic = numbers_after_key("NumberOfImagesInMosaic", 1)
    if mosaic:
        out["number_of_images_in_mosaic"] = int(mosaic[0])

    grad = numbers_after_key("DiffusionGradientDirection", 3)
    if grad:
        out["diffusion_gradient_direction"] = grad
    return out


def parse_dicom(path: Path) -> dict:
    data = path.read_bytes()
    pos = 132 if data[128:132] == b"DICM" else 0
    meta = {}
    explicit = True

    while pos + 8 <= len(data):
        group, elem = struct.unpack_from("<HH", data, pos)
        pos += 4
        if (group, elem) == (0x7FE0, 0x0010):
            break

        vr = None
        if explicit:
            maybe_vr = data[pos:pos + 2].decode("ascii", errors="ignore")
            if maybe_vr.isalpha():
                vr = maybe_vr
                pos += 2
                if vr in LONG_VR:
                    pos += 2
                    if pos + 4 > len(data):
                        break
                    length = struct.unpack_from("<I", data, pos)[0]
                    pos += 4
                else:
                    if pos + 2 > len(data):
                        break
                    length = struct.unpack_from("<H", data, pos)[0]
                    pos += 2
            else:
                explicit = False
                if pos + 4 > len(data):
                    break
                length = struct.unpack_from("<I", data, pos)[0]
                pos += 4
        else:
            if pos + 4 > len(data):
                break
            length = struct.unpack_from("<I", data, pos)[0]
            pos += 4

        if length == 0xFFFFFFFF:
            break
        if pos + length > len(data):
            break
        raw = data[pos:pos + length]
        pos += length

        key = TAGS.get((group, elem))
        if key is not None:
            meta[key] = value_to_python(vr, raw)
        if group == 0x0029 and elem >= 0x1000:
            text = clean_text(raw)
            if (
                "B_value" in text
                or "DiffusionGradientDirection" in text
                or "NumberOfImagesInMosaic" in text
            ):
                meta.setdefault("siemens_csa_text", "")
                meta["siemens_csa_text"] += "\n" + text[:20000]

    csa_text = meta.get("siemens_csa_text", "")
    diffusion_meta = extract_siemens_diffusion_meta(csa_text) if csa_text else {}
    if "b_value" not in diffusion_meta and b"B_value" in data:
        csa_text = printable_text(data) + "\n" + csa_text
        diffusion_meta = extract_siemens_diffusion_meta(csa_text)
    if csa_text:
        meta["siemens_csa_text"] = csa_text
        meta.update(diffusion_meta)
    if meta.get("b_value") == 0 and "diffusion_gradient_direction" in meta:
        meta.pop("diffusion_gradient_direction", None)
    meta["path"] = str(path)
    return meta


def normal_from_orientation(orientation):
    if not isinstance(orientation, list) or len(orientation) < 6:
        return None
    r = orientation[:3]
    c = orientation[3:6]
    n = [
        r[1] * c[2] - r[2] * c[1],
        r[2] * c[0] - r[0] * c[2],
        r[0] * c[1] - r[1] * c[0],
    ]
    norm = math.sqrt(sum(x * x for x in n))
    if norm <= 1e-12:
        return None
    return [x / norm for x in n]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def summarize(items):
    first = items[0]
    positions = [x.get("image_position_patient") for x in items if isinstance(x.get("image_position_patient"), list)]
    normal = normal_from_orientation(first.get("image_orientation_patient"))
    slice_coords = []
    if normal is not None:
        for p in positions:
            if len(p) >= 3:
                slice_coords.append(dot(p[:3], normal))
    uniq_slices = sorted({round(x, 4) for x in slice_coords})
    diffs = [round(uniq_slices[i + 1] - uniq_slices[i], 4) for i in range(len(uniq_slices) - 1)]

    paths = [x["path"] for x in items]
    image_type = str(first.get("image_type", ""))
    protocol = str(first.get("protocol_name", ""))
    desc = str(first.get("series_description", ""))
    is_diffusion = any(s in (image_type + protocol + desc).upper() for s in ["DIFFUSION", "DIFF", "EP2D"])
    b_values = [x.get("b_value") for x in items if x.get("b_value") is not None]
    b_counts = Counter(str(int(x) if isinstance(x, float) and x.is_integer() else x) for x in b_values)
    gradients = [
        tuple(round(v, 6) for v in x.get("diffusion_gradient_direction", []))
        for x in items
        if x.get("diffusion_gradient_direction") and x.get("b_value", 0) > 0
    ]
    gradient_counts = Counter(gradients)
    mosaic_counts = Counter(
        str(x.get("number_of_images_in_mosaic"))
        for x in items
        if x.get("number_of_images_in_mosaic") is not None
    )

    return {
        "series_uid": first.get("series_instance_uid", "UNKNOWN"),
        "series_number": first.get("series_number"),
        "description": desc,
        "protocol": protocol,
        "sequence_name": first.get("sequence_name"),
        "image_type": image_type,
        "n_files": len(items),
        "rows": first.get("rows"),
        "columns": first.get("columns"),
        "pixel_spacing": first.get("pixel_spacing"),
        "slice_thickness": first.get("slice_thickness"),
        "unique_slice_positions": len(uniq_slices),
        "slice_step_candidates": sorted(set(diffs))[:10],
        "orientation": first.get("image_orientation_patient"),
        "normal": normal,
        "patient_position": first.get("patient_position"),
        "is_diffusion_like": is_diffusion,
        "b_value_counts": dict(sorted(b_counts.items())),
        "unique_diffusion_gradients": len(gradient_counts),
        "diffusion_gradient_counts": {" ".join(map(str, k)): v for k, v in gradient_counts.most_common()},
        "mosaic_image_counts": dict(sorted(mosaic_counts.items())),
        "sample_file": paths[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()

    files = []
    for arg in args.paths:
        p = Path(arg)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.dcm")))
        else:
            files.append(p)

    groups = defaultdict(list)
    for path in files:
        try:
            meta = parse_dicom(path)
            groups[meta.get("series_instance_uid", f"UNKNOWN:{path.parent}")].append(meta)
        except Exception as exc:
            print(f"SKIP {path}: {exc}")

    summaries = [summarize(v) for _, v in sorted(groups.items(), key=lambda kv: (str(kv[1][0].get("series_number")), kv[0]))]

    for s in summaries:
        print(f"\nSeries {s['series_number']}  files={s['n_files']}  diffusion={s['is_diffusion_like']}")
        print(f"  desc/protocol: {s['description']} / {s['protocol']}")
        print(f"  dims: {s['columns']} x {s['rows']}  pixel_spacing={s['pixel_spacing']}  thickness={s['slice_thickness']}")
        print(f"  unique slices: {s['unique_slice_positions']}  slice steps={s['slice_step_candidates']}")
        print(f"  orient={s['orientation']} normal={s['normal']} patient={s['patient_position']}")
        if s["is_diffusion_like"]:
            print(f"  b-values: {s['b_value_counts']}  gradients={s['unique_diffusion_gradients']}  mosaic={s['mosaic_image_counts']}")
        print(f"  sample={s['sample_file']}")

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
