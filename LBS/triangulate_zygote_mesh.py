import numpy as np
import os

THRESHOLD_AREA = 1  # Adjust as needed
THRESHOLD_EDGE_LENGTH = 0.5  # Adjust this based on the longest edge constraint

def compute_edge_length(vertices, v1, v2):
    """Computes the length of an edge between two vertices."""
    return np.linalg.norm(np.array(vertices[v1]) - np.array(vertices[v2]))

def compute_triangle_area(vertices, v1, v2, v3):
    """Calculate the area of a triangle given its vertex indices."""
    p1 = np.array(vertices[v1])
    p2 = np.array(vertices[v2])
    p3 = np.array(vertices[v3])
    return 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))

def parse_face(face):
    """Parses a face and returns vertex, texture, and normal indices."""
    v_idx, vt_idx, vn_idx = [], [], []
    
    for component in face:
        parts = component.split('/')
        v_idx.append(int(parts[0]) - 1)
        vt_idx.append(int(parts[1]) - 1 if len(parts) > 1 and parts[1] else -1)
        vn_idx.append(int(parts[2]) - 1 if len(parts) > 2 and parts[2] else -1)

    return v_idx, vt_idx, vn_idx

def get_midpoint(vertices, textures, normals, v1, v2, vt1, vt2, vn1, vn2, vertex_dict):
    """Computes the midpoint and stores it to avoid duplication."""
    edge_key = tuple(sorted((v1, v2)))
    if edge_key in vertex_dict:
        return vertex_dict[edge_key]

    mid_v = (np.array(vertices[v1]) + np.array(vertices[v2])) / 2.0
    new_v_idx = len(vertices)
    vertices.append(mid_v.tolist())

    mid_vt_idx = -1
    if vt1 != -1 and vt2 != -1:
        mid_vt = (np.array(textures[vt1]) + np.array(textures[vt2])) / 2.0
        mid_vt_idx = len(textures)
        textures.append(mid_vt.tolist())

    mid_vn_idx = -1
    if vn1 != -1 and vn2 != -1:
        mid_vn = (np.array(normals[vn1]) + np.array(normals[vn2])) / 2.0
        mid_vn /= np.linalg.norm(mid_vn)  # Normalize normal
        mid_vn_idx = len(normals)
        normals.append(mid_vn.tolist())

    vertex_dict[edge_key] = (new_v_idx, mid_vt_idx, mid_vn_idx)
    return new_v_idx, mid_vt_idx, mid_vn_idx

def triangulate_obj(input_filepath, output_filepath):
    """Reads an OBJ file, triangulates non-triangle faces, and iteratively subdivides large triangles."""
    with open(input_filepath, 'r') as file:
        lines = file.readlines()

    vertices, textures, normals, faces = [], [], [], []

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == 'v':
            vertices.append([float(x) for x in parts[1:]])
        elif parts[0] == 'vt':
            textures.append([float(x) for x in parts[1:]])
        elif parts[0] == 'vn':
            normals.append([float(x) for x in parts[1:]])
        elif parts[0] == 'f':
            faces.append(parts[1:])

    new_faces = []
    for face in faces:
        if len(face) == 3:
            new_faces.append(face)  # Keep triangles as they are
        else:
            v0 = face[0]
            for i in range(1, len(face) - 1):
                new_faces.append([v0, face[i], face[i + 1]])

    # Subdivision loop until all triangles are small enough
    subdivided_faces = new_faces
    vertex_dict = {}

    while True:
        next_faces = []
        subdivided = False  # Flag to check if we need another iteration

        for face in subdivided_faces:
            v_indices, vt_indices, vn_indices = parse_face(face)

            # Compute area and edge lengths
            # area = compute_triangle_area(vertices, v_indices[0], v_indices[1], v_indices[2])
            edge_lengths = [
                (compute_edge_length(vertices, v_indices[0], v_indices[1]), 0, 1),
                (compute_edge_length(vertices, v_indices[1], v_indices[2]), 1, 2),
                (compute_edge_length(vertices, v_indices[2], v_indices[0]), 2, 0),
            ]

            # Find longest edge
            longest_edge_length, i1, i2 = max(edge_lengths, key=lambda x: x[0])

            # Check constraints
            # if area < THRESHOLD_AREA and longest_edge_length < THRESHOLD_EDGE_LENGTH:
            if longest_edge_length < THRESHOLD_EDGE_LENGTH:
                next_faces.append(face)  # Keep triangles that meet both conditions
                continue

            subdivided = True  # Mark that we need another pass

            # Get the third vertex (the one opposite the longest edge)
            i3 = 3 - (i1 + i2)

            # Compute midpoint of the longest edge
            mid_v, mid_vt, mid_vn = get_midpoint(
                vertices, textures, normals,
                v_indices[i1], v_indices[i2],
                vt_indices[i1], vt_indices[i2],
                vn_indices[i1], vn_indices[i2],
                vertex_dict
            )

            # Create two new triangles
            next_faces.extend([
                [f"{v_indices[i1]+1}/{vt_indices[i1]+1 if vt_indices[i1] != -1 else ''}/{vn_indices[i1]+1 if vn_indices[i1] != -1 else ''}",
                f"{mid_v+1}/{mid_vt+1 if mid_vt != -1 else ''}/{mid_vn+1 if mid_vn != -1 else ''}",
                f"{v_indices[i3]+1}/{vt_indices[i3]+1 if vt_indices[i3] != -1 else ''}/{vn_indices[i3]+1 if vn_indices[i3] != -1 else ''}"],

                [f"{mid_v+1}/{mid_vt+1 if mid_vt != -1 else ''}/{mid_vn+1 if mid_vn != -1 else ''}",
                f"{v_indices[i2]+1}/{vt_indices[i2]+1 if vt_indices[i2] != -1 else ''}/{vn_indices[i2]+1 if vn_indices[i2] != -1 else ''}",
                f"{v_indices[i3]+1}/{vt_indices[i3]+1 if vt_indices[i3] != -1 else ''}/{vn_indices[i3]+1 if vn_indices[i3] != -1 else ''}"]
            ])

        if not subdivided:
            break  # Stop when no more subdivisions are needed

        subdivided_faces = next_faces

    # Write the final OBJ file
    with open(output_filepath, 'w') as file:
        file.write("# Processed OBJ File\n")
        for v in vertices:
            file.write(f"v {' '.join(map(str, v))}\n")
        for vt in textures:
            file.write(f"vt {' '.join(map(str, vt))}\n")
        for vn in normals:
            file.write(f"vn {' '.join(map(str, vn))}\n")
        for face in subdivided_faces:
            file.write(f"f {' '.join(face)}\n")

    print(f"Processed mesh saved to {output_filepath}")

def process_directory(input_dir, output_dir):
    """Recursively processes all OBJ files in the directory."""
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.obj'):
                input_filepath = os.path.join(root, file)
                relative_path = os.path.relpath(input_filepath, input_dir)
                output_filepath = os.path.join(output_dir, relative_path)

                os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
                triangulate_obj(input_filepath, output_filepath)

input_dir = "../Zygote_Meshes_Revised"
output_dir = "../Zygote_Meshes_Revised_Subdivided"
os.makedirs(output_dir, exist_ok=True)

process_directory(input_dir, output_dir)
