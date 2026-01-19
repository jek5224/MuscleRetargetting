# Fiber Architecture operations for muscle mesh processing
# Extracted from mesh_loader.py for better organization

import numpy as np
from OpenGL.GL import *
from scipy.spatial import cKDTree, Delaunay

# Constants
MESH_SCALE = 0.01


# =============================================================================
# Radial Interpolation Helper Functions
# =============================================================================

def unit_circle_to_radial_params(x, y, N, first_vertex_angle=0):
    """
    Convert unit circle coordinates to radial interpolation parameters.

    Args:
        x, y: Point in [0,1]^2 (unit circle centered at 0.5, 0.5)
        N: Number of vertices in contour
        first_vertex_angle: Angle of contour vertex 0 from centroid (for alignment)

    Returns:
        (vertex_angle, radius) where:
        - vertex_angle in [0, N): continuous vertex index
        - radius in [0, 1): normalized radial distance
    """
    import math

    # Convert from [0,1]^2 to centered coordinates
    cx = x - 0.5
    cy = y - 0.5

    # Compute angle in [0, 2*pi)
    theta = math.atan2(cy, cx)
    if theta < 0:
        theta += 2 * math.pi

    # Adjust angle so that fiber at first_vertex_angle maps to vertex 0
    # This aligns unit circle reference frame with contour vertex ordering
    adjusted_theta = theta - first_vertex_angle
    if adjusted_theta < 0:
        adjusted_theta += 2 * math.pi

    # Map adjusted angle to vertex index
    vertex_angle = adjusted_theta * N / (2 * math.pi)

    # Compute normalized radius (capped at 0.99 for strict interior)
    euclidean_radius = math.sqrt(cx * cx + cy * cy)
    radius = min(euclidean_radius / 0.5, 0.99)  # Normalize and cap

    return vertex_angle, radius


def compute_radial_weights(vertex_angle, radius, N):
    """
    Compute vertex weights for radial interpolation.

    The waypoint is computed as:
        waypoint = (1 - r) * centroid + r * boundary_point

    Where:
        centroid = (1/N) * sum(vertices)
        boundary_point = (1-t) * V_i + t * V_{i+1}

    This expands to weights on vertices:
        w_k = (1-r)/N                for all k (centroid contribution)
        w_i += r*(1-t)               for boundary vertex i
        w_j += r*t                   for boundary vertex j = (i+1) % N

    Args:
        vertex_angle: Continuous index in [0, N)
        radius: Normalized distance [0, 1)
        N: Number of vertices

    Returns:
        (N,) array of weights that sum to 1
    """
    i = int(vertex_angle) % N
    t = vertex_angle - int(vertex_angle)
    j = (i + 1) % N

    # Base weight for all vertices (from centroid contribution)
    weights = np.full(N, (1 - radius) / N)

    # Add boundary contribution to vertices i and j
    weights[i] += radius * (1 - t)
    weights[j] += radius * t

    return weights


def find_angle_bracket(vertex_angles, target_angle):
    """
    Find two consecutive vertices (in contour order) that bracket target_angle.

    Args:
        vertex_angles: List of angles (radians) for each vertex from centroid
        target_angle: Target angle (radians) to find bracketing vertices for

    Returns:
        (i, j, t) where:
        - i, j are consecutive vertex indices in contour order
        - t in [0,1] is interpolation parameter: boundary = (1-t)*V[i] + t*V[j]
    """
    import math
    N = len(vertex_angles)

    if N < 2:
        return 0, 0, 0

    for idx in range(N):
        i = idx
        j = (idx + 1) % N
        a1 = vertex_angles[i]
        a2 = vertex_angles[j]

        # Handle wrap-around at 2π
        if a2 < a1:
            a2_check = a2 + 2 * math.pi
        else:
            a2_check = a2

        # Adjust target for wrap-around check
        if a2 < a1:  # Wrap-around case
            target_check = target_angle if target_angle >= a1 else target_angle + 2 * math.pi
        else:
            target_check = target_angle

        if a1 <= target_check <= a2_check:
            if abs(a2_check - a1) < 1e-10:
                t = 0
            else:
                t = (target_check - a1) / (a2_check - a1)
            return i, j, t

    # Fallback: return first two vertices
    return 0, 1, 0


# =============================================================================
# Angular Fiber Positioning Helper Functions
# =============================================================================

def create_angular_unit_circle_vertices(contour_vertices, geodesic_indices):
    """
    Create unit circle vertex angles matching contour vertices.

    1. Geodesic vertices get uniform angles: 0, 2π/N_geo, 4π/N_geo, ...
    2. Non-geodesic vertices between each geodesic pair get angles based on
       distance ratios from the contour.

    Args:
        contour_vertices: (N, 3) array of contour vertices in order
        geodesic_indices: List of indices into contour_vertices that are geodesic vertices

    Returns:
        unit_circle_angles: (N,) array of angles for each contour vertex
    """
    contour_vertices = np.array(contour_vertices)
    N = len(contour_vertices)
    N_geo = len(geodesic_indices)

    if N_geo == 0 or N_geo == 1:
        # Fallback: uniform distribution
        return np.linspace(0, 2 * np.pi, N, endpoint=False)

    unit_circle_angles = np.zeros(N)

    # Step 1: Assign uniform angles to geodesic vertices
    for g in range(N_geo):
        geo_idx = geodesic_indices[g]
        unit_circle_angles[geo_idx] = g * 2 * np.pi / N_geo

    # Step 2: For each segment between geodesic vertices, subdivide non-geodesic vertices
    for g in range(N_geo):
        start_geo_idx = geodesic_indices[g]
        end_geo_idx = geodesic_indices[(g + 1) % N_geo]

        start_angle = unit_circle_angles[start_geo_idx]
        end_angle = unit_circle_angles[end_geo_idx]

        # Handle angle wrap-around
        if end_angle <= start_angle:
            end_angle += 2 * np.pi

        angle_span = end_angle - start_angle

        # Build list of contour indices from start to end (exclusive of end for non-geodesic)
        if end_geo_idx <= start_geo_idx:
            # Wrap around
            indices = list(range(start_geo_idx, N)) + list(range(0, end_geo_idx))
        else:
            indices = list(range(start_geo_idx, end_geo_idx))

        # indices[0] is the start geodesic vertex (already assigned)
        # indices[1:] are the non-geodesic vertices to subdivide
        if len(indices) <= 1:
            continue

        # Compute cumulative distances from start geodesic vertex
        cumulative_dists = [0.0]
        for i in range(len(indices) - 1):
            idx_curr = indices[i]
            idx_next = indices[i + 1]
            d = np.linalg.norm(contour_vertices[idx_next] - contour_vertices[idx_curr])
            cumulative_dists.append(cumulative_dists[-1] + d)

        # Add distance to end geodesic vertex
        last_idx = indices[-1]
        d_to_end = np.linalg.norm(contour_vertices[end_geo_idx] - contour_vertices[last_idx])
        total_dist = cumulative_dists[-1] + d_to_end

        # Assign angles to non-geodesic vertices (skip indices[0] which is geodesic)
        for i in range(1, len(indices)):
            idx = indices[i]
            if total_dist > 1e-10:
                ratio = cumulative_dists[i] / total_dist
            else:
                ratio = i / len(indices)
            angle = start_angle + ratio * angle_span
            unit_circle_angles[idx] = angle % (2 * np.pi)

    return unit_circle_angles


def sample_fibers_angular(num_fibers, num_vertices, radius_range=(0.1, 0.9), unit_circle_angles=None):
    """
    Sample fiber positions on radial lines from origin to unit circle vertices.

    Each fiber is placed on a different radial line (cycling through vertices),
    with random radius along each line.

    Args:
        num_fibers: Number of fibers to sample
        num_vertices: Number of vertices in contour
        radius_range: (min, max) radius range for sampling
        unit_circle_angles: If provided, use these angles. Otherwise use uniform angles.

    Returns:
        fiber_samples_2d: (num_fibers, 2) array of (x, y) positions in unit circle space [0,1]^2
        fiber_samples_params: (num_fibers, 2) array of (vertex_idx, radius) for waypoint computation
    """
    fiber_samples_2d = np.zeros((num_fibers, 2))
    fiber_samples_params = np.zeros((num_fibers, 2))

    min_r, max_r = radius_range

    for i in range(num_fibers):
        # Cycle through vertices to distribute fibers evenly on radial lines
        vertex_idx = i % num_vertices

        # Random radius along the radial line
        radius = np.random.uniform(min_r, max_r)

        # Get angle for this vertex
        if unit_circle_angles is not None:
            angle = unit_circle_angles[vertex_idx]
        else:
            # Uniform angles
            angle = 2 * np.pi * vertex_idx / num_vertices

        # Compute 2D position on the radial line
        x = 0.5 + radius * 0.5 * np.cos(angle)
        y = 0.5 + radius * 0.5 * np.sin(angle)

        fiber_samples_2d[i] = [x, y]
        fiber_samples_params[i] = [vertex_idx, radius]

    return fiber_samples_2d, fiber_samples_params


def find_geodesic_vertex_indices(contour_vertices, num_geodesic_lines, geodesic_paths=None, mesh_vertices=None):
    """
    Find indices of contour vertices that correspond to geodesic line crossings.

    If geodesic_paths and mesh_vertices are provided, finds actual crossing points.
    Otherwise falls back to uniform spacing assumption.

    Each geodesic line gets a unique contour vertex - if the closest vertex is already
    assigned to another geodesic, we find the next closest unassigned vertex.

    Args:
        contour_vertices: (N, 3) array of contour vertices
        num_geodesic_lines: Number of geodesic lines
        geodesic_paths: List of geodesic path dicts with 'chain' (vertex indices)
        mesh_vertices: Mesh vertices array for computing geodesic edge positions

    Returns:
        geodesic_indices: List of contour vertex indices corresponding to geodesic crossings
                         Always has exactly num_geodesic_lines elements (one per geodesic)
    """
    contour_vertices = np.array(contour_vertices)
    N_contour = len(contour_vertices)

    if num_geodesic_lines == 0 or N_contour == 0:
        return []

    # If we have actual geodesic path info, find real crossing points
    if geodesic_paths is not None and mesh_vertices is not None and len(geodesic_paths) > 0:
        # First pass: compute distance from each contour vertex to each geodesic
        # geodesic_distances[g] = (best_dist, best_contour_idx) for geodesic g
        geodesic_candidates = []

        for path_info in geodesic_paths:
            chain = path_info['chain']
            if len(chain) < 2:
                geodesic_candidates.append((float('inf'), 0))
                continue

            # Compute distance from each contour vertex to this geodesic line
            min_dists = np.full(N_contour, float('inf'))
            for i in range(len(chain) - 1):
                v1 = mesh_vertices[chain[i]]
                v2 = mesh_vertices[chain[i + 1]]
                edge_mid = (v1 + v2) / 2

                # Distance from each contour vertex to this edge midpoint
                dists = np.linalg.norm(contour_vertices - edge_mid, axis=1)
                min_dists = np.minimum(min_dists, dists)

            # Store (sorted_indices, min_dists) for this geodesic
            sorted_indices = np.argsort(min_dists)
            geodesic_candidates.append((min_dists, sorted_indices))

        # Second pass: assign unique contour vertices to each geodesic
        # Process geodesics in order, assigning closest available vertex
        used_vertices = set()
        geodesic_indices = []

        for g in range(len(geodesic_candidates)):
            min_dists, sorted_indices = geodesic_candidates[g]

            # Find closest unassigned vertex
            assigned = False
            for idx in sorted_indices:
                if idx not in used_vertices:
                    geodesic_indices.append(idx)
                    used_vertices.add(idx)
                    assigned = True
                    break

            if not assigned:
                # All vertices used - this shouldn't happen if N_contour >= num_geodesic_lines
                # Fall back to closest vertex anyway
                geodesic_indices.append(sorted_indices[0])

        # Sort by contour order to maintain proper ordering
        geodesic_indices = sorted(geodesic_indices)
        return geodesic_indices

    # Fallback: assume uniform spacing (less accurate)
    verts_per_segment = N_contour // num_geodesic_lines
    geodesic_indices = [i * verts_per_segment for i in range(num_geodesic_lines)]

    return geodesic_indices


# =============================================================================
# Triangulation-Based Fiber Mapping Classes and Functions
# =============================================================================

class UnitCircleTriangulation:
    """
    Manages 2D triangulation of unit circle domain for fiber mapping.

    The triangulation consists of:
    - Boundary vertices on the unit circle (corresponding to contour vertices)
    - Interior grid points (concentric rings + center) for quality triangulation
    """

    def __init__(self, n_boundary, uc_angles, n_interior_rings=3):
        """
        Create triangulated unit circle mesh.

        Args:
            n_boundary: Number of boundary vertices
            uc_angles: (N,) array of angles for boundary vertices
            n_interior_rings: Number of concentric rings for interior points
        """
        self.center = np.array([0.5, 0.5])
        self.radius = 0.4
        self.n_boundary = n_boundary
        self.n_interior_rings = n_interior_rings

        # Create boundary vertices
        boundary_verts = []
        for angle in uc_angles:
            x = 0.5 + self.radius * np.cos(angle)
            y = 0.5 + self.radius * np.sin(angle)
            boundary_verts.append([x, y])
        boundary_verts = np.array(boundary_verts)

        # Create interior grid points (concentric rings + center)
        interior_verts = [[0.5, 0.5]]  # Center point
        for ring_idx in range(1, n_interior_rings + 1):
            r = self.radius * ring_idx / (n_interior_rings + 1)
            n_ring = max(6, int(n_boundary * ring_idx / (n_interior_rings + 1)))
            for i in range(n_ring):
                angle = 2 * np.pi * i / n_ring
                interior_verts.append([0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)])
        interior_verts = np.array(interior_verts)

        # Combine vertices
        self.vertices = np.vstack([boundary_verts, interior_verts])
        self.boundary_indices = list(range(n_boundary))
        self.interior_indices = list(range(n_boundary, len(self.vertices)))

        # Triangulate using Delaunay
        self.delaunay = Delaunay(self.vertices)
        self.triangles = self.delaunay.simplices

        # Extract edge structure for ARAP
        self._build_edge_structure()

    def _build_edge_structure(self):
        """Build edge and neighbor structure for ARAP solver."""
        edges_set = set()
        for tri in self.triangles:
            for i in range(3):
                e = tuple(sorted([tri[i], tri[(i + 1) % 3]]))
                edges_set.add(e)
        self.edges = list(edges_set)

        # Build neighbor lists
        n_verts = len(self.vertices)
        self.neighbors = [[] for _ in range(n_verts)]
        for i, j in self.edges:
            self.neighbors[i].append(j)
            self.neighbors[j].append(i)

        # Compute edge weights (uniform for simplicity)
        self.weights = {}
        for i, j in self.edges:
            self.weights[(i, j)] = 1.0
            self.weights[(j, i)] = 1.0

        # Rest edge vectors
        self.rest_edges = [{} for _ in range(n_verts)]
        for i, j in self.edges:
            self.rest_edges[i][j] = self.vertices[j] - self.vertices[i]
            self.rest_edges[j][i] = self.vertices[i] - self.vertices[j]


class DirectFiberTriangulation:
    """
    Triangulation with boundary vertices + fiber samples directly as interior vertices.

    No extra grid points - fibers ARE the interior vertices.
    This allows solving harmonic interpolation directly for fiber positions.
    """

    def __init__(self, n_boundary, uc_angles, fiber_samples):
        """
        Create triangulation with boundary + fibers as vertices.

        Args:
            n_boundary: Number of boundary vertices
            uc_angles: (N,) array of angles for boundary vertices
            fiber_samples: (M, 2) array of fiber positions in [0,1]^2
        """
        self.center = np.array([0.5, 0.5])
        self.radius = 0.4
        self.n_boundary = n_boundary
        self.n_fibers = len(fiber_samples)

        # Create boundary vertices
        boundary_verts = []
        for angle in uc_angles:
            x = 0.5 + self.radius * np.cos(angle)
            y = 0.5 + self.radius * np.sin(angle)
            boundary_verts.append([x, y])
        boundary_verts = np.array(boundary_verts)

        # Fibers are interior vertices
        fiber_verts = np.array(fiber_samples)

        # Combine: boundary first, then fibers
        self.vertices = np.vstack([boundary_verts, fiber_verts])
        self.boundary_indices = list(range(n_boundary))
        self.interior_indices = list(range(n_boundary, len(self.vertices)))  # These ARE the fibers
        self.fiber_vertex_indices = self.interior_indices  # Alias for clarity

        # Triangulate using Delaunay
        self.delaunay = Delaunay(self.vertices)
        self.triangles = self.delaunay.simplices

        # Build edge structure for harmonic solver
        self._build_edge_structure()

    def _build_edge_structure(self):
        """Build edge and neighbor structure for harmonic solver."""
        edges_set = set()
        for tri in self.triangles:
            for i in range(3):
                e = tuple(sorted([tri[i], tri[(i + 1) % 3]]))
                edges_set.add(e)
        self.edges = list(edges_set)

        # Build neighbor lists
        n_verts = len(self.vertices)
        self.neighbors = [[] for _ in range(n_verts)]
        for i, j in self.edges:
            self.neighbors[i].append(j)
            self.neighbors[j].append(i)

        # Uniform weights
        self.weights = {}
        for i, j in self.edges:
            self.weights[(i, j)] = 1.0
            self.weights[(j, i)] = 1.0


class GeodesicTriangulation:
    """
    Direct triangulation with geodesic crossing vertices as boundary and fibers as interior.

    No unit circle mapping - works directly in 2D contour space.
    Uses constrained Delaunay to preserve boundary edges.
    """

    def __init__(self, boundary_2d, fiber_samples_2d):
        """
        Create triangulation with geodesic boundary + fibers.

        Args:
            boundary_2d: (N_boundary, 2) geodesic crossing vertices in 2D contour space
            fiber_samples_2d: (M, 2) fiber positions in 2D contour space
        """
        import triangle

        self.n_boundary = len(boundary_2d)
        self.n_fibers = len(fiber_samples_2d)

        # Sort boundary vertices by angle from centroid to form proper polygon
        centroid = np.mean(boundary_2d, axis=0)
        rel = boundary_2d - centroid
        angles = np.arctan2(rel[:, 1], rel[:, 0])
        angle_order = np.argsort(angles)
        sorted_boundary = boundary_2d[angle_order]

        # Combine: sorted boundary first, then fibers
        self.vertices = np.vstack([sorted_boundary, fiber_samples_2d])
        self.boundary_indices = list(range(self.n_boundary))
        self.fiber_vertex_indices = list(range(self.n_boundary, len(self.vertices)))

        # Create boundary segments (closed polygon)
        segments = []
        for i in range(self.n_boundary):
            segments.append([i, (i + 1) % self.n_boundary])

        # Use constrained Delaunay triangulation
        tri_input = {
            'vertices': self.vertices,
            'segments': np.array(segments)
        }

        # 'p' = PSLG (planar straight line graph)
        # 'q' = quality mesh (optional, adds Steiner points)
        # We use just 'p' to preserve our exact vertices
        try:
            tri_output = triangle.triangulate(tri_input, 'p')
            self.triangles = tri_output['triangles']
        except Exception:
            # Fallback to unconstrained Delaunay if constrained fails
            from scipy.spatial import Delaunay
            delaunay = Delaunay(self.vertices)
            self.triangles = delaunay.simplices

        # Build edge structure
        self._build_edge_structure()

    def _build_edge_structure(self):
        """Build edge and neighbor structure."""
        edges_set = set()
        for tri in self.triangles:
            for i in range(3):
                e = tuple(sorted([tri[i], tri[(i + 1) % 3]]))
                edges_set.add(e)
        self.edges = list(edges_set)

        # Build neighbor lists
        n_verts = len(self.vertices)
        self.neighbors = [[] for _ in range(n_verts)]
        for i, j in self.edges:
            self.neighbors[i].append(j)
            self.neighbors[j].append(i)

        # Uniform weights
        self.weights = {}
        for i, j in self.edges:
            self.weights[(i, j)] = 1.0
            self.weights[(j, i)] = 1.0


class FiberTriangleEmbedding:
    """
    Stores fiber embeddings in unit circle triangulation.

    Each fiber sample is embedded in a triangle with barycentric coordinates,
    allowing position computation in deformed triangulation.
    """

    def __init__(self, fiber_samples, triangulation):
        """
        Embed fiber samples in triangulation.

        Args:
            fiber_samples: (M, 2) array of fiber positions in [0,1]^2
            triangulation: UnitCircleTriangulation object
        """
        self.fiber_samples = np.array(fiber_samples)
        self.n_fibers = len(fiber_samples)

        # Find containing triangle for each fiber
        self.triangle_ids = triangulation.delaunay.find_simplex(fiber_samples)
        self.is_external = np.zeros(self.n_fibers, dtype=bool)

        # Compute barycentric coordinates
        self.bary_coords = np.zeros((self.n_fibers, 3))
        for i in range(self.n_fibers):
            tri_idx = self.triangle_ids[i]
            is_external = tri_idx < 0

            if is_external:
                # Outside triangulation - find nearest triangle
                tri_idx = self._find_nearest_triangle(fiber_samples[i], triangulation)
                self.triangle_ids[i] = tri_idx
                self.is_external[i] = True

            tri_verts = triangulation.vertices[triangulation.triangles[tri_idx]]
            # Clamp barycentric coords for external fibers to keep waypoints inside contour
            self.bary_coords[i] = self._compute_2d_barycentric(fiber_samples[i], tri_verts, clamp=is_external)

    def _find_nearest_triangle(self, point, triangulation):
        """Find nearest triangle for a point outside the triangulation."""
        min_dist = float('inf')
        nearest_tri = 0
        for tri_idx, tri in enumerate(triangulation.triangles):
            tri_verts = triangulation.vertices[tri]
            centroid = np.mean(tri_verts, axis=0)
            dist = np.linalg.norm(point - centroid)
            if dist < min_dist:
                min_dist = dist
                nearest_tri = tri_idx
        return nearest_tri

    def _compute_2d_barycentric(self, p, tri_verts, clamp=False):
        """
        Compute barycentric coordinates of point p in triangle.

        Args:
            p: (2,) point
            tri_verts: (3, 2) triangle vertices
            clamp: If True, clamp coordinates to [0,1] and renormalize (for external points)

        Returns:
            (3,) barycentric coordinates (u, v, w) where u + v + w = 1
        """
        v0, v1, v2 = tri_verts

        # Vectors
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = p - v0

        # Dot products
        d00 = np.dot(v0v1, v0v1)
        d01 = np.dot(v0v1, v0v2)
        d11 = np.dot(v0v2, v0v2)
        d20 = np.dot(v0p, v0v1)
        d21 = np.dot(v0p, v0v2)

        # Barycentric coordinates
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-10:
            # Degenerate triangle
            return np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        bary = np.array([u, v, w])

        if clamp:
            # Clamp to [0, 1] and renormalize - projects external point to triangle boundary
            bary = np.clip(bary, 0.0, 1.0)
            total = np.sum(bary)
            if total > 1e-10:
                bary = bary / total
            else:
                bary = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

        return bary


def create_unit_circle_triangulation(n_boundary, uc_angles, n_interior_rings=3):
    """
    Create triangulated unit circle mesh.

    Args:
        n_boundary: Number of boundary vertices
        uc_angles: (N,) array of angles for boundary vertices
        n_interior_rings: Number of concentric rings for interior points

    Returns:
        UnitCircleTriangulation object
    """
    return UnitCircleTriangulation(n_boundary, uc_angles, n_interior_rings)


def embed_fibers_in_triangulation(fiber_samples, triangulation):
    """
    Embed fiber samples in triangulation.

    Args:
        fiber_samples: (M, 2) array of fiber positions in [0,1]^2
        triangulation: UnitCircleTriangulation object

    Returns:
        FiberTriangleEmbedding object
    """
    return FiberTriangleEmbedding(fiber_samples, triangulation)


def create_direct_fiber_triangulation(n_boundary, uc_angles, fiber_samples):
    """
    Create triangulation with boundary + fibers directly (no extra grid points).

    Args:
        n_boundary: Number of boundary vertices
        uc_angles: (N,) array of angles for boundary vertices
        fiber_samples: (M, 2) array of fiber positions in [0,1]^2

    Returns:
        DirectFiberTriangulation object
    """
    return DirectFiberTriangulation(n_boundary, uc_angles, fiber_samples)


def find_waypoints_harmonic_direct(bounding_plane_info, fiber_samples, triangulation):
    """
    Compute waypoints using direct harmonic interpolation.

    Fibers ARE the interior vertices - their positions are solved directly
    via harmonic interpolation from boundary (contour) vertices.

    Args:
        bounding_plane_info: Dictionary with contour info
        fiber_samples: (M, 2) array of fiber positions (for reference)
        triangulation: DirectFiberTriangulation object

    Returns:
        (deformed_2d, waypoints_3d): Deformed 2D vertices and 3D waypoints
    """
    # Get contour vertices
    Ps = bounding_plane_info.get('contour_vertices')
    if Ps is None:
        contour_match = bounding_plane_info.get('contour_match')
        if contour_match is not None:
            Ps = np.array([pair[0] for pair in contour_match])
        else:
            return None, np.array([])

    Ps = np.array(Ps)
    n_contour = len(Ps)

    if n_contour != triangulation.n_boundary:
        return None, np.array([])

    # Get projection basis
    basis_x = bounding_plane_info.get('basis_x')
    basis_y = bounding_plane_info.get('basis_y')
    center_3d = bounding_plane_info.get('mean', np.mean(Ps, axis=0))

    # Project contour to 2D (target boundary positions)
    target_boundary_2d = np.zeros((n_contour, 2))
    for i, p in enumerate(Ps):
        rel = p - center_3d
        if basis_x is not None and basis_y is not None:
            target_boundary_2d[i] = [np.dot(rel, basis_x), np.dot(rel, basis_y)]
        else:
            target_boundary_2d[i] = [rel[0], rel[1]]

    # Solve harmonic interpolation - fiber positions are solved directly
    deformed_2d = solve_interior_vertices_harmonic(triangulation, target_boundary_2d)

    # Extract fiber positions (interior vertices = fibers)
    fiber_positions_2d = deformed_2d[triangulation.fiber_vertex_indices]

    # Convert to 3D waypoints
    waypoints = np.zeros((len(fiber_positions_2d), 3))
    for i, p2d in enumerate(fiber_positions_2d):
        if basis_x is not None and basis_y is not None:
            waypoints[i] = center_3d + p2d[0] * basis_x + p2d[1] * basis_y
        else:
            waypoints[i] = center_3d + np.array([p2d[0], p2d[1], 0])

    return deformed_2d, waypoints


# =============================================================================
# Flip-Free Deformation Helper Functions
# =============================================================================

def compute_signed_area_2d(v0, v1, v2):
    """
    Compute signed area of 2D triangle.

    Args:
        v0, v1, v2: (2,) vertex positions

    Returns:
        Signed area (positive = CCW, negative = CW/flipped)
    """
    return 0.5 * ((v1[0] - v0[0]) * (v2[1] - v0[1]) -
                  (v2[0] - v0[0]) * (v1[1] - v0[1]))


def get_original_triangle_areas(triangulation):
    """
    Compute signed areas of all triangles in original triangulation.

    Args:
        triangulation: UnitCircleTriangulation or DirectFiberTriangulation

    Returns:
        List of signed areas for each triangle
    """
    areas = []
    for tri in triangulation.triangles:
        v0 = triangulation.vertices[tri[0]]
        v1 = triangulation.vertices[tri[1]]
        v2 = triangulation.vertices[tri[2]]
        areas.append(compute_signed_area_2d(v0, v1, v2))
    return areas


def build_vertex_to_triangles(triangulation):
    """
    Build mapping from vertex index to containing triangle indices.

    Args:
        triangulation: UnitCircleTriangulation or DirectFiberTriangulation

    Returns:
        List of lists: vertex_triangles[i] = [tri_idx1, tri_idx2, ...]
    """
    n_verts = len(triangulation.vertices)
    vertex_triangles = [[] for _ in range(n_verts)]
    for tri_idx, tri in enumerate(triangulation.triangles):
        for vi in tri:
            vertex_triangles[vi].append(tri_idx)
    return vertex_triangles


def causes_flip_for_vertex(positions, triangulation, vertex_idx, original_areas, vertex_triangles):
    """
    Check if current vertex position causes any adjacent triangle to flip.

    Args:
        positions: (N, 2) current vertex positions
        triangulation: UnitCircleTriangulation or DirectFiberTriangulation
        vertex_idx: Index of vertex to check
        original_areas: List of original triangle signed areas
        vertex_triangles: Vertex-to-triangles mapping

    Returns:
        True if any adjacent triangle has flipped orientation
    """
    min_area_threshold = 1e-10  # Treat very small areas as flipped

    for tri_idx in vertex_triangles[vertex_idx]:
        tri = triangulation.triangles[tri_idx]
        area = compute_signed_area_2d(positions[tri[0]], positions[tri[1]], positions[tri[2]])

        # Check for flip (sign change) or degenerate (near-zero area)
        if original_areas[tri_idx] > 0:
            if area <= min_area_threshold:
                return True
        else:
            if area >= -min_area_threshold:
                return True

    return False


def find_uc_segment(angle, uc_boundary_angles):
    """
    Find which unit circle boundary segment contains the given angle.

    The unit circle boundary angles ARE ordered (created from uc_angles parameter).
    This finds which segment (i, i+1) brackets the target angle.

    Args:
        angle: Target angle in radians (-pi to pi)
        uc_boundary_angles: (N,) array of unit circle boundary vertex angles (ordered)

    Returns:
        (segment_idx, t): Segment index and interpolation parameter (0-1)
    """
    n = len(uc_boundary_angles)

    # Normalize to [0, 2*pi)
    target = angle % (2 * np.pi)
    norm_angles = uc_boundary_angles % (2 * np.pi)

    # Find bracketing segment
    for i in range(n):
        j = (i + 1) % n
        a1 = norm_angles[i]
        a2 = norm_angles[j]

        # Handle wrap-around case (last segment crosses 0/2pi)
        if a2 < a1:  # Wraps around
            if target >= a1 or target <= a2:
                # Compute interpolation parameter
                segment_span = (2 * np.pi - a1) + a2
                if target >= a1:
                    t = (target - a1) / segment_span
                else:
                    t = (2 * np.pi - a1 + target) / segment_span
                return i, t
        else:
            if a1 <= target <= a2:
                segment_span = a2 - a1
                if segment_span < 1e-10:
                    t = 0.5
                else:
                    t = (target - a1) / segment_span
                return i, t

    # Fallback: find closest angle (shouldn't happen with proper coverage)
    diffs = np.abs(np.arctan2(np.sin(target - norm_angles), np.cos(target - norm_angles)))
    closest = np.argmin(diffs)
    return closest, 0.0


def _solve_harmonic_direct(triangulation, target_boundary_2d, fixed_mask, free_indices):
    """
    Internal helper: solve harmonic interpolation directly (may have flips).

    Args:
        triangulation: Triangulation object with edge structure
        target_boundary_2d: (N_boundary, 2) target boundary positions
        fixed_mask: Boolean mask for fixed vertices
        free_indices: List of free (interior) vertex indices

    Returns:
        positions: (V, 2) solved vertex positions
    """
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.linalg import spsolve

    n_verts = len(triangulation.vertices)
    n_free = len(free_indices)

    # Initialize with boundary positions
    positions = np.zeros((n_verts, 2))
    for i, bi in enumerate(triangulation.boundary_indices):
        positions[bi] = target_boundary_2d[i]

    if n_free == 0:
        return positions

    # Map from global index to free index
    free_map = {gi: fi for fi, gi in enumerate(free_indices)}

    # Build Laplacian system
    L = lil_matrix((n_free, n_free))
    b = np.zeros((n_free, 2))

    for fi, i in enumerate(free_indices):
        total_weight = 0.0
        for j in triangulation.neighbors[i]:
            w = triangulation.weights.get((i, j), 1.0)
            total_weight += w

            if fixed_mask[j]:
                b[fi] += w * positions[j]
            else:
                fj = free_map[j]
                L[fi, fj] = -w

        L[fi, fi] = total_weight

    # Solve
    L_csr = csr_matrix(L)
    for dim in range(2):
        x = spsolve(L_csr, b[:, dim])
        for fi, i in enumerate(free_indices):
            positions[i, dim] = x[fi]

    return positions


def solve_interior_vertices_harmonic(triangulation, target_boundary_2d):
    """
    Solve for interior vertex positions using unit circle angle correspondence.

    Algorithm: For each interior point in unit circle:
    1. Find which UC boundary segment contains its angle
    2. Get the CORRESPONDING contour boundary positions (same indices)
    3. Interpolate between contour positions and scale radially

    This preserves the 1:1 correspondence between UC boundary and contour vertices.

    Args:
        triangulation: UnitCircleTriangulation with edge structure
        target_boundary_2d: (N_boundary, 2) target positions for boundary vertices

    Returns:
        deformed_vertices: (V, 2) array of solved vertex positions
    """
    n_verts = len(triangulation.vertices)
    positions = np.zeros((n_verts, 2))

    # 1. Set boundary vertices to contour positions
    for i, bi in enumerate(triangulation.boundary_indices):
        positions[bi] = target_boundary_2d[i]

    # 2. Compute unit circle boundary angles (these ARE ordered)
    uc_center = triangulation.center  # [0.5, 0.5]
    uc_radius = triangulation.radius  # 0.4
    n_boundary = triangulation.n_boundary

    uc_boundary_angles = np.zeros(n_boundary)
    for i, bi in enumerate(triangulation.boundary_indices):
        rel = triangulation.vertices[bi] - uc_center
        uc_boundary_angles[i] = np.arctan2(rel[1], rel[0])

    # 3. Compute contour center
    contour_center = np.mean(target_boundary_2d, axis=0)

    # 4. Find interior vertices
    fixed_mask = np.zeros(n_verts, dtype=bool)
    fixed_mask[triangulation.boundary_indices] = True
    interior_indices = [i for i in range(n_verts) if not fixed_mask[i]]

    if len(interior_indices) == 0:
        return positions

    # 5. Map each interior vertex
    for i in interior_indices:
        uc_pos = triangulation.vertices[i]

        # Get angle and radius in unit circle space
        rel = uc_pos - uc_center
        angle = np.arctan2(rel[1], rel[0])
        uc_radius_at_point = np.linalg.norm(rel)

        # Normalized radius (0=center, 1=boundary)
        if uc_radius > 1e-10:
            t_radial = uc_radius_at_point / uc_radius
        else:
            t_radial = 0.0

        # Find which UC boundary segment contains this angle
        segment_idx, t_angular = find_uc_segment(angle, uc_boundary_angles)

        # Get CORRESPONDING contour boundary positions (same indices!)
        p0 = target_boundary_2d[segment_idx]
        p1 = target_boundary_2d[(segment_idx + 1) % n_boundary]

        # Interpolate boundary position
        boundary_pos = (1 - t_angular) * p0 + t_angular * p1

        # Scale radially from contour center
        positions[i] = contour_center + t_radial * (boundary_pos - contour_center)

    # Apply area ratio preservation refinement
    positions = refine_for_area_preservation(triangulation, positions)

    return positions


def triangle_area_2d(p0, p1, p2):
    """Compute signed area of 2D triangle."""
    return 0.5 * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))


def refine_for_area_preservation(triangulation, positions, n_iterations=10, blend=0.5):
    """
    Refine interior vertex positions to preserve triangle area ratios from unit circle.

    Uses iterative Laplacian smoothing weighted by local area ratios.
    Vertices in regions with too-small triangles move outward, too-large move inward.

    Args:
        triangulation: UnitCircleTriangulation with reference areas
        positions: (V, 2) current vertex positions
        n_iterations: Number of refinement iterations
        blend: How much to blend area correction (0=none, 1=full)

    Returns:
        refined_positions: (V, 2) refined vertex positions
    """
    positions = positions.copy()
    n_verts = len(positions)
    triangles = triangulation.triangles

    # Identify interior vertices (not boundary)
    fixed_mask = np.zeros(n_verts, dtype=bool)
    fixed_mask[triangulation.boundary_indices] = True

    interior_indices = [i for i in range(n_verts) if not fixed_mask[i]]
    if len(interior_indices) == 0:
        return positions

    # Compute reference areas in unit circle space
    uc_verts = triangulation.vertices
    ref_areas = np.zeros(len(triangles))
    for ti, tri in enumerate(triangles):
        ref_areas[ti] = abs(triangle_area_2d(uc_verts[tri[0]], uc_verts[tri[1]], uc_verts[tri[2]]))

    total_ref_area = np.sum(ref_areas)
    if total_ref_area < 1e-10:
        return positions

    # Build vertex-to-triangle adjacency and neighbor lists
    vert_triangles = [[] for _ in range(n_verts)]
    neighbors = [set() for _ in range(n_verts)]
    for ti, tri in enumerate(triangles):
        for k in range(3):
            vi = tri[k]
            vert_triangles[vi].append(ti)
            neighbors[vi].add(tri[(k + 1) % 3])
            neighbors[vi].add(tri[(k + 2) % 3])

    # Compute contour center
    contour_center = np.mean(positions[triangulation.boundary_indices], axis=0)

    # Iterative refinement
    for iteration in range(n_iterations):
        # Compute current areas
        cur_areas = np.zeros(len(triangles))
        for ti, tri in enumerate(triangles):
            cur_areas[ti] = abs(triangle_area_2d(positions[tri[0]], positions[tri[1]], positions[tri[2]]))

        total_cur_area = np.sum(cur_areas)
        if total_cur_area < 1e-10:
            break

        # Compute area scale factors for each triangle
        # scale > 1 means triangle is too small, needs to expand
        # scale < 1 means triangle is too large, needs to shrink
        area_scales = np.ones(len(triangles))
        for ti in range(len(triangles)):
            if cur_areas[ti] > 1e-12:
                target_area = ref_areas[ti] * total_cur_area / total_ref_area
                area_scales[ti] = np.sqrt(target_area / cur_areas[ti])
                # Clamp to prevent extreme adjustments
                area_scales[ti] = np.clip(area_scales[ti], 0.8, 1.25)

        # Update each interior vertex
        new_positions = positions.copy()
        for vi in interior_indices:
            # Compute average scale factor from adjacent triangles
            avg_scale = 1.0
            if len(vert_triangles[vi]) > 0:
                avg_scale = np.mean([area_scales[ti] for ti in vert_triangles[vi]])

            # Current position relative to center
            rel = positions[vi] - contour_center

            # Scale radially to adjust local area
            scaled_pos = contour_center + rel * avg_scale

            # Blend with Laplacian smooth (average of neighbors)
            if len(neighbors[vi]) > 0:
                neighbor_avg = np.mean([positions[ni] for ni in neighbors[vi]], axis=0)
                # Blend: area-scaled position with neighbor average for smoothness
                new_positions[vi] = (1 - blend * 0.3) * scaled_pos + blend * 0.3 * neighbor_avg
            else:
                new_positions[vi] = scaled_pos

        positions = new_positions

    return positions


def compute_waypoints_triangulated(embedding, triangulation, deformed_verts_3d):
    """
    Compute 3D waypoints using barycentric interpolation in deformed triangles.

    Args:
        embedding: FiberTriangleEmbedding
        triangulation: UnitCircleTriangulation
        deformed_verts_3d: (V, 3) deformed vertex positions in 3D

    Returns:
        waypoints: (M, 3) array of waypoint positions
    """
    waypoints = np.zeros((embedding.n_fibers, 3))

    for i in range(embedding.n_fibers):
        tri_idx = embedding.triangle_ids[i]
        bary = embedding.bary_coords[i]

        tri_vert_indices = triangulation.triangles[tri_idx]
        v0 = deformed_verts_3d[tri_vert_indices[0]]
        v1 = deformed_verts_3d[tri_vert_indices[1]]
        v2 = deformed_verts_3d[tri_vert_indices[2]]

        waypoints[i] = bary[0] * v0 + bary[1] * v1 + bary[2] * v2

    return waypoints


def find_waypoints_triangulated(bounding_plane_info, fiber_samples, triangulation, embedding):
    """
    Compute waypoints using triangulation + harmonic interpolation.

    Boundary vertices of the unit circle triangulation are "stitched" to contour vertices.
    Interior vertices are smoothly interpolated using Laplacian/harmonic coordinates.

    Args:
        bounding_plane_info: Dictionary with contour info ('contour_vertices', 'basis_x', 'basis_y', 'mean')
        fiber_samples: (M, 2) array of fiber positions (for reference)
        triangulation: UnitCircleTriangulation object
        embedding: FiberTriangleEmbedding object

    Returns:
        (deformed_2d, waypoints_3d): Deformed 2D vertices and 3D waypoints
    """
    # Get contour vertices
    Ps = bounding_plane_info.get('contour_vertices')
    if Ps is None:
        contour_match = bounding_plane_info.get('contour_match')
        if contour_match is not None:
            Ps = np.array([pair[0] for pair in contour_match])
        else:
            return None, np.array([])

    Ps = np.array(Ps)
    n_contour = len(Ps)

    if n_contour != triangulation.n_boundary:
        # Contour vertex count mismatch - fall back
        return None, np.array([])

    # Get projection basis
    basis_x = bounding_plane_info.get('basis_x')
    basis_y = bounding_plane_info.get('basis_y')
    center_3d = bounding_plane_info.get('mean', np.mean(Ps, axis=0))

    # Project contour to 2D (target boundary positions)
    target_boundary_2d = np.zeros((n_contour, 2))
    for i, p in enumerate(Ps):
        rel = p - center_3d
        if basis_x is not None and basis_y is not None:
            target_boundary_2d[i] = [np.dot(rel, basis_x), np.dot(rel, basis_y)]
        else:
            target_boundary_2d[i] = [rel[0], rel[1]]

    # Solve harmonic interpolation for interior vertices (simple stitching)
    deformed_2d = solve_interior_vertices_harmonic(triangulation, target_boundary_2d)

    # Convert deformed 2D to 3D
    deformed_3d = np.zeros((len(deformed_2d), 3))
    for i, p2d in enumerate(deformed_2d):
        if basis_x is not None and basis_y is not None:
            deformed_3d[i] = center_3d + p2d[0] * basis_x + p2d[1] * basis_y
        else:
            deformed_3d[i] = center_3d + np.array([p2d[0], p2d[1], 0])

    # Compute waypoints using barycentric interpolation
    waypoints = compute_waypoints_triangulated(embedding, triangulation, deformed_3d)

    return deformed_2d, waypoints


def find_waypoints_geodesic(bounding_plane_info, fiber_samples, geodesic_paths=None, mesh_vertices=None):
    """
    Compute waypoints using direct geodesic triangulation.

    Uses geodesic crossing vertices as boundary and fibers directly as interior points.
    No unit circle mapping - works directly in contour 2D space.

    Args:
        bounding_plane_info: Dictionary with contour info ('contour_vertices', 'basis_x', 'basis_y', 'mean')
        fiber_samples: (M, 2) array of fiber positions in [0,1]^2
        geodesic_paths: List of geodesic path dicts with 'chain' (vertex indices)
        mesh_vertices: Mesh vertices array for computing geodesic positions

    Returns:
        (triangulation, waypoints_3d): GeodesicTriangulation object and 3D waypoints
    """
    # Get contour vertices
    Ps = bounding_plane_info.get('contour_vertices')
    if Ps is None:
        contour_match = bounding_plane_info.get('contour_match')
        if contour_match is not None:
            Ps = np.array([pair[0] for pair in contour_match])
        else:
            return None, np.array([])

    Ps = np.array(Ps)
    n_contour = len(Ps)

    # Get projection basis
    basis_x = bounding_plane_info.get('basis_x')
    basis_y = bounding_plane_info.get('basis_y')
    center_3d = bounding_plane_info.get('mean', np.mean(Ps, axis=0))

    # Project all contour vertices to 2D
    contour_2d = np.zeros((n_contour, 2))
    for i, p in enumerate(Ps):
        rel = p - center_3d
        if basis_x is not None and basis_y is not None:
            contour_2d[i] = [np.dot(rel, basis_x), np.dot(rel, basis_y)]
        else:
            contour_2d[i] = [rel[0], rel[1]]

    # Get geodesic vertex indices
    num_geodesic = len(geodesic_paths) if geodesic_paths is not None else 8
    geodesic_indices = find_geodesic_vertex_indices(Ps, num_geodesic, geodesic_paths, mesh_vertices)

    if len(geodesic_indices) < 3:
        # Not enough geodesic vertices - fall back
        return None, np.array([])

    # Extract geodesic boundary vertices in 2D (these are already in contour order)
    geodesic_boundary_2d = contour_2d[geodesic_indices]

    # Map fiber samples from [0,1]^2 to geodesic polygon space
    # Use proportional angular mapping
    fiber_samples = np.array(fiber_samples)
    fiber_samples_2d = map_fibers_to_geodesic_polygon(
        fiber_samples, geodesic_boundary_2d
    )

    # Check for empty fibers
    if len(fiber_samples) == 0:
        return None, np.array([])

    # Create geodesic triangulation
    try:
        triangulation = GeodesicTriangulation(geodesic_boundary_2d, fiber_samples_2d)
    except Exception:
        return None, np.array([])

    # Fiber waypoints are at their positions in the triangulation
    # (vertices after boundary = fiber vertices)
    fiber_positions_2d = triangulation.vertices[triangulation.fiber_vertex_indices]

    # Convert 2D positions to 3D
    n_fibers = len(fiber_samples)
    waypoints_3d = np.zeros((n_fibers, 3))
    for i, p2d in enumerate(fiber_positions_2d):
        if basis_x is not None and basis_y is not None:
            waypoints_3d[i] = center_3d + p2d[0] * basis_x + p2d[1] * basis_y
        else:
            waypoints_3d[i] = center_3d + np.array([p2d[0], p2d[1], 0])

    return triangulation, waypoints_3d


def map_fibers_to_geodesic_polygon(fiber_samples, geodesic_boundary_2d):
    """
    Map fiber samples from [0,1]^2 unit square to geodesic polygon.

    Uses proportional angular mapping:
    - Fiber angle from (0.5, 0.5) → same angle in geodesic polygon from centroid
    - Fiber radial distance → proportional distance to boundary at that angle

    Args:
        fiber_samples: (M, 2) array of fiber positions in [0,1]^2
        geodesic_boundary_2d: (N, 2) geodesic boundary vertices in 2D

    Returns:
        fiber_positions_2d: (M, 2) fiber positions in geodesic polygon space
    """
    fiber_samples = np.array(fiber_samples)
    n_fibers = len(fiber_samples)
    n_boundary = len(geodesic_boundary_2d)

    # Unit square center and radius
    us_center = np.array([0.5, 0.5])
    us_radius = 0.5  # Max distance from center to edge in unit square

    # Geodesic polygon centroid
    geo_center = np.mean(geodesic_boundary_2d, axis=0)

    # Compute angles of geodesic boundary vertices from centroid
    geo_rel = geodesic_boundary_2d - geo_center
    geo_angles = np.arctan2(geo_rel[:, 1], geo_rel[:, 0])

    # Sort boundary vertices by angle for proper interpolation
    angle_order = np.argsort(geo_angles)
    sorted_boundary = geodesic_boundary_2d[angle_order]
    sorted_angles = geo_angles[angle_order]

    # Map each fiber
    fiber_positions_2d = np.zeros((n_fibers, 2))

    for i, fiber in enumerate(fiber_samples):
        # Fiber position relative to unit square center
        rel = fiber - us_center
        fiber_angle = np.arctan2(rel[1], rel[0])
        fiber_radius = np.linalg.norm(rel)

        # Normalized radius (0 = center, 1 = boundary of unit square)
        t_radial = min(fiber_radius / us_radius, 1.0)

        # Find geodesic boundary position at this angle (using sorted boundary)
        boundary_pos = interpolate_geodesic_boundary(
            sorted_boundary, geo_center, sorted_angles, fiber_angle
        )

        # Scale radially from geodesic centroid
        fiber_positions_2d[i] = geo_center + t_radial * (boundary_pos - geo_center)

    return fiber_positions_2d


def interpolate_geodesic_boundary(boundary_2d, center, boundary_angles, target_angle):
    """
    Find position on geodesic boundary at a given angle.

    Args:
        boundary_2d: (N, 2) boundary vertices
        center: (2,) centroid of boundary
        boundary_angles: (N,) angles of boundary vertices from center
        target_angle: Target angle in radians

    Returns:
        position: (2,) interpolated position on boundary
    """
    n = len(boundary_2d)

    # Normalize angles to [0, 2*pi)
    target = target_angle % (2 * np.pi)
    norm_angles = boundary_angles % (2 * np.pi)

    # Find bracketing segment
    for i in range(n):
        j = (i + 1) % n
        a1 = norm_angles[i]
        a2 = norm_angles[j]

        # Handle wrap-around
        if a2 < a1:  # Wraps around
            if target >= a1 or target <= a2:
                segment_span = (2 * np.pi - a1) + a2
                if target >= a1:
                    t = (target - a1) / segment_span
                else:
                    t = (2 * np.pi - a1 + target) / segment_span
                return (1 - t) * boundary_2d[i] + t * boundary_2d[j]
        else:
            if a1 <= target <= a2:
                segment_span = a2 - a1
                if segment_span < 1e-10:
                    t = 0.5
                else:
                    t = (target - a1) / segment_span
                return (1 - t) * boundary_2d[i] + t * boundary_2d[j]

    # Fallback: closest vertex
    diffs = np.abs(np.arctan2(np.sin(target - norm_angles), np.cos(target - norm_angles)))
    closest = np.argmin(diffs)
    return boundary_2d[closest].copy()


def create_shared_geodesic_triangulation(num_geo, fiber_samples, n_interior_rings=3):
    """
    Create a shared triangulation for geodesic mode.

    All contours will have the same number of boundary vertices (num_geo),
    so we can use ONE triangulation for all contours.

    Args:
        num_geo: Number of geodesic paths (= number of boundary vertices)
        fiber_samples: (M, 2) array of fiber positions in [0,1]^2
        n_interior_rings: Number of concentric rings for interior points

    Returns:
        (triangulation, embedding): UnitCircleTriangulation and FiberTriangleEmbedding
    """
    if num_geo < 3:
        return None, None

    # Evenly spaced angles on unit circle (geodesics provide consistent reference)
    uc_angles = np.linspace(0, 2 * np.pi, num_geo, endpoint=False)

    # Create triangulation
    triangulation = UnitCircleTriangulation(num_geo, uc_angles, n_interior_rings)

    # Embed fibers
    embedding = FiberTriangleEmbedding(fiber_samples, triangulation)

    return triangulation, embedding


def find_waypoints_geodesic_shared(bounding_plane_info, fiber_samples, triangulation, embedding,
                                    geodesic_paths, mesh_vertices):
    """
    Compute waypoints using shared geodesic triangulation.

    Uses geodesic crossing vertices as boundary positions, but the triangulation
    structure (connectivity) is shared across all contours.

    Args:
        bounding_plane_info: Dictionary with contour info
        fiber_samples: (M, 2) array of fiber positions in [0,1]^2
        triangulation: UnitCircleTriangulation (shared)
        embedding: FiberTriangleEmbedding (shared)
        geodesic_paths: List of geodesic path dicts
        mesh_vertices: Mesh vertices array

    Returns:
        (deformed_2d, waypoints_3d): Deformed 2D vertices and 3D waypoints
    """
    if triangulation is None or embedding is None:
        return None, np.array([])

    # Get contour vertices
    Ps = bounding_plane_info.get('contour_vertices')
    if Ps is None:
        contour_match = bounding_plane_info.get('contour_match')
        if contour_match is not None:
            Ps = np.array([pair[0] for pair in contour_match])
        else:
            return None, np.array([])

    Ps = np.array(Ps)
    num_geo = triangulation.n_boundary

    # Find geodesic crossing vertices for this contour
    geodesic_indices = find_geodesic_vertex_indices(Ps, num_geo, geodesic_paths, mesh_vertices)

    if len(geodesic_indices) != num_geo:
        # Mismatch - fallback
        return None, np.array([])

    # Get geodesic crossing points (boundary positions for this contour)
    geodesic_boundary_3d = Ps[geodesic_indices]

    # Get projection basis
    basis_x = bounding_plane_info.get('basis_x')
    basis_y = bounding_plane_info.get('basis_y')
    center_3d = bounding_plane_info.get('mean', np.mean(Ps, axis=0))

    # Project geodesic boundary to 2D (target boundary positions)
    target_boundary_2d = np.zeros((num_geo, 2))
    for i, p in enumerate(geodesic_boundary_3d):
        rel = p - center_3d
        if basis_x is not None and basis_y is not None:
            target_boundary_2d[i] = [np.dot(rel, basis_x), np.dot(rel, basis_y)]
        else:
            target_boundary_2d[i] = [rel[0], rel[1]]

    # Solve harmonic interpolation for interior vertices
    deformed_2d = solve_interior_vertices_harmonic(triangulation, target_boundary_2d)

    # Convert deformed 2D to 3D
    deformed_3d = np.zeros((len(deformed_2d), 3))
    for i, p2d in enumerate(deformed_2d):
        if basis_x is not None and basis_y is not None:
            deformed_3d[i] = center_3d + p2d[0] * basis_x + p2d[1] * basis_y
        else:
            deformed_3d[i] = center_3d + np.array([p2d[0], p2d[1], 0])

    # Compute waypoints using barycentric interpolation
    waypoints = compute_waypoints_triangulated(embedding, triangulation, deformed_3d)

    return deformed_2d, waypoints


class FiberArchitectureMixin:
    """
    Mixin class providing fiber/waypoint-related methods for MeshLoader.
    Handles waypoint computation, embedding, barycentric interpolation, and updates.
    """

    def _init_fiber_properties(self):
        """Initialize fiber-related properties. Call from MeshLoader.__init__."""
        # Fiber architecture
        self.fiber_architecture = None  # Will be set by main class
        self.is_draw_fiber_architecture = False
        self.is_one_fiber = False
        self.sampling_method = 'sobol_unit_square'  # 'sobol_unit_square' or 'sobol_min_contour'
        self.cutting_method = 'bp'  # 'bp', 'area_based', 'voronoi', 'angular', 'gradient', 'ratio', 'cumulative_area', or 'projected_area'

        # Waypoints
        self.waypoints = []
        self.waypoints_original = None
        self.waypoints_from_tet_sim = True  # If False, waypoints are imported and won't be updated by tet sim
        self.waypoint_bary_coords = []

        # VIPER waypoints
        self.viper_waypoints = []

        # Bone bounds for skeleton attachment
        self._bone_bounds = {}

    def find_contour_match(self, muscle_contour_orig, template_contour, prev_P0=None, preserve_order=False):
        """
        Find matching points between muscle contour and template contour (bounding plane corners).

        Uses ray-based intersection: for each BP corner, cast a ray from BP center through
        the corner and find where it intersects the contour. This provides consistent
        corner-to-contour correspondence across levels.

        Args:
            muscle_contour_orig: The muscle contour vertices (N x 3 array)
            template_contour: The bounding plane corners (4 x 3 array)
            prev_P0: Unused (kept for compatibility)
            preserve_order: If True, don't roll or reverse the muscle contour (keep normalized order)
        """
        muscle_contour = muscle_contour_orig.copy()
        template_contour = np.array(template_contour)

        # ===== Step 1: Compute BP coordinate system =====
        # BP corners are ordered: 0-1-2-3 forming a quadrilateral
        #   3 --- 2
        #   |     |
        #   0 --- 1
        bp_center_3d = np.mean(template_contour, axis=0)

        # Compute basis vectors from BP corners
        # Use edges to define the plane, then orthogonalize
        edge_01 = template_contour[1] - template_contour[0]
        edge_03 = template_contour[3] - template_contour[0]

        # Compute plane normal
        normal = np.cross(edge_01, edge_03)
        normal_norm = np.linalg.norm(normal)
        if normal_norm > 1e-10:
            normal = normal / normal_norm
        else:
            normal = np.array([0.0, 0.0, 1.0])

        # basis_x along edge_01 direction
        basis_x = edge_01.copy()
        basis_x_norm = np.linalg.norm(basis_x)
        if basis_x_norm > 1e-10:
            basis_x = basis_x / basis_x_norm
        else:
            basis_x = np.array([1.0, 0.0, 0.0])

        # basis_y perpendicular to both normal and basis_x (ensures orthogonality)
        basis_y = np.cross(normal, basis_x)
        basis_y_norm = np.linalg.norm(basis_y)
        if basis_y_norm > 1e-10:
            basis_y = basis_y / basis_y_norm
        else:
            basis_y = np.array([0.0, 1.0, 0.0])

        # ===== Step 2: Project to 2D =====
        def to_2d(pt_3d):
            diff = pt_3d - bp_center_3d
            return np.array([np.dot(diff, basis_x), np.dot(diff, basis_y)])

        def to_3d(pt_2d):
            return bp_center_3d + pt_2d[0] * basis_x + pt_2d[1] * basis_y

        # Project contour and BP corners to 2D
        contour_2d = np.array([to_2d(v) for v in muscle_contour])
        corners_2d = np.array([to_2d(v) for v in template_contour])
        center_2d = np.array([0.0, 0.0])  # BP center is origin in 2D

        # ===== Step 3: Find ray-contour intersections for each corner =====
        def ray_segment_intersection_2d(ray_origin, ray_dir, seg_start, seg_end):
            """
            Find intersection of ray with line segment in 2D.
            Returns (t_ray, t_seg) where intersection = ray_origin + t_ray * ray_dir
            and intersection = seg_start + t_seg * (seg_end - seg_start).
            Returns (None, None) if no valid intersection.
            """
            seg_dir = seg_end - seg_start
            # Solve: ray_origin + t_ray * ray_dir = seg_start + t_seg * seg_dir
            # [ray_dir, -seg_dir] * [t_ray, t_seg]^T = seg_start - ray_origin
            denom = ray_dir[0] * (-seg_dir[1]) - ray_dir[1] * (-seg_dir[0])
            if abs(denom) < 1e-10:
                return None, None  # Parallel

            diff = seg_start - ray_origin
            t_ray = (diff[0] * (-seg_dir[1]) - diff[1] * (-seg_dir[0])) / denom
            t_seg = (ray_dir[0] * diff[1] - ray_dir[1] * diff[0]) / denom

            # Valid if ray goes forward (t_ray > 0) and intersection is on segment (0 <= t_seg <= 1)
            if t_ray > 1e-10 and 0 <= t_seg <= 1:
                return t_ray, t_seg
            return None, None

        # For each corner, find intersection with contour
        corner_intersections = []  # List of (corner_idx, edge_idx, t_seg, t_ray)

        for corner_idx in range(4):
            corner_2d = corners_2d[corner_idx]
            ray_dir = corner_2d - center_2d
            ray_dir_norm = np.linalg.norm(ray_dir)
            if ray_dir_norm < 1e-10:
                continue
            ray_dir = ray_dir / ray_dir_norm

            best_intersection = None
            best_t_ray = float('inf')

            n_contour = len(contour_2d)
            for edge_idx in range(n_contour):
                seg_start = contour_2d[edge_idx]
                seg_end = contour_2d[(edge_idx + 1) % n_contour]

                t_ray, t_seg = ray_segment_intersection_2d(center_2d, ray_dir, seg_start, seg_end)

                if t_ray is not None:
                    # If multiple intersections, pick closest to corner (smallest t_ray that's past the corner)
                    # Corner is at t_ray = distance from center to corner
                    corner_dist = np.linalg.norm(corner_2d - center_2d)
                    # We want intersection close to the corner
                    dist_to_corner = abs(t_ray - corner_dist)
                    if best_intersection is None or dist_to_corner < abs(best_t_ray - corner_dist):
                        best_intersection = (corner_idx, edge_idx, t_seg)
                        best_t_ray = t_ray

            if best_intersection is not None:
                corner_intersections.append(best_intersection)

        # ===== Fallback: if ray-based method fails, use distance-based =====
        if len(corner_intersections) < 4:
            # Ray-based method didn't find all 4 intersections
            # Fall back to distance-based matching for missing corners
            found_corners = set(ci[0] for ci in corner_intersections)
            for corner_idx in range(4):
                if corner_idx not in found_corners:
                    # Find closest point on contour edges (original method)
                    v = template_contour[corner_idx]
                    min_distance = float("inf")
                    min_edge_idx = 0
                    min_t = 0.0
                    for edge_idx in range(len(muscle_contour)):
                        v0 = muscle_contour[edge_idx]
                        v1 = muscle_contour[(edge_idx + 1) % len(muscle_contour)]
                        edge_dir = v1 - v0
                        edge_len_sq = np.dot(edge_dir, edge_dir)
                        if edge_len_sq > 1e-10:
                            t = np.clip(np.dot(v - v0, edge_dir) / edge_len_sq, 0, 1)
                        else:
                            t = 0
                        closest_pt = v0 + t * edge_dir
                        dist = np.linalg.norm(v - closest_pt)
                        if dist < min_distance:
                            min_distance = dist
                            min_edge_idx = edge_idx
                            min_t = t
                    corner_intersections.append((corner_idx, min_edge_idx, min_t))

        # ===== Step 4: Insert vertices at intersections (in 3D) =====
        # First, compute all intersection 3D points and their contour positions
        intersection_data = []  # List of (corner_idx, edge_idx, t_seg, vertex_3d)

        for corner_idx, edge_idx, t_seg in corner_intersections:
            v0_3d = muscle_contour[edge_idx]
            v1_3d = muscle_contour[(edge_idx + 1) % len(muscle_contour)]
            vertex_3d = v0_3d + t_seg * (v1_3d - v0_3d)
            intersection_data.append((corner_idx, edge_idx, t_seg, vertex_3d))

        # Sort by contour position (edge_idx ascending, then t_seg ascending)
        intersection_data.sort(key=lambda x: (x[1], x[2]))

        # Insert vertices in order, tracking index offset
        corner_to_muscle_idx = {}
        index_offset = 0

        for corner_idx, edge_idx, t_seg, vertex_3d in intersection_data:
            adjusted_edge_idx = edge_idx + index_offset

            if 1e-6 < t_seg < 1 - 1e-6:
                # Intersection is on edge interior - insert new vertex
                insert_idx = adjusted_edge_idx + 1
                muscle_contour = np.insert(muscle_contour, insert_idx, vertex_3d, axis=0)
                corner_to_muscle_idx[corner_idx] = insert_idx
                index_offset += 1
            elif t_seg <= 1e-6:
                # Intersection at segment start
                corner_to_muscle_idx[corner_idx] = adjusted_edge_idx
            else:
                # Intersection at segment end
                corner_to_muscle_idx[corner_idx] = (adjusted_edge_idx + 1) % len(muscle_contour)

        # ===== Step 5: Build closest_muscle_index from corner intersections =====
        # Use ray-based indices where available, fallback to distance for missing corners
        closest_muscle_index = []
        for corner_idx in range(len(template_contour)):
            if corner_idx in corner_to_muscle_idx:
                closest_muscle_index.append(corner_to_muscle_idx[corner_idx])
            else:
                # Fallback: use distance-based matching
                distances = np.linalg.norm(muscle_contour - template_contour[corner_idx], axis=1)
                closest_muscle_index.append(np.argmin(distances))

        if not preserve_order:
            # Roll contours to align starting points
            P0_index = closest_muscle_index[0]

            muscle_contour = np.roll(muscle_contour, -P0_index, axis=0)

            # Adjust closest_muscle_index after rolling
            n = len(muscle_contour)
            closest_muscle_index = [(idx - P0_index) % n for idx in closest_muscle_index]

            # Check winding order - reverse if needed
            if len(closest_muscle_index) >= 3 and closest_muscle_index[2] < closest_muscle_index[1]:
                muscle_contour = np.roll(muscle_contour[::-1], 1, axis=0)
                # Adjust indices: after reverse and roll by 1
                # New index for old index i: (n - 1 - i + 1) % n = (n - i) % n
                n = len(muscle_contour)
                closest_muscle_index = [(n - idx) % n for idx in closest_muscle_index]
        # If preserve_order=True, keep indices as computed

        result_index = []
        result = []
        for i in range(len(template_contour)):
            result_index.append((closest_muscle_index[i], i))

        for i in range(len(result_index)):
            muscle_start = result_index[i][0]
            muscle_end = result_index[(i + 1) % len(result_index)][0]

            template_start_pos = template_contour[result_index[i][1]]
            template_end_pos = template_contour[result_index[(i + 1) % len(result_index)][1]]

            # Handle wrap-around: when muscle_end < muscle_start, we need to go around the contour
            if muscle_end >= muscle_start:
                muscle_list = list(range(muscle_start, muscle_end + 1))
            else:
                # Wrap around: go from muscle_start to end, then from 0 to muscle_end
                muscle_list = list(range(muscle_start, len(muscle_contour))) + list(range(0, muscle_end + 1))

            segment_t = []
            t_sum = 0
            for j in range(1, len(muscle_list)):
                t = np.linalg.norm(muscle_contour[muscle_list[j]] - muscle_contour[muscle_list[j - 1]])
                t_sum += t
                segment_t.append(t_sum)

            if t_sum > 0:
                segment_t = [t / t_sum for t in segment_t]

            for j in range(len(segment_t)):
                t = segment_t[j]
                template_pos = (1 - t) * template_start_pos + t * template_end_pos
                result.append((muscle_contour[muscle_list[j + 1]], template_pos))

        return muscle_contour, result

    def draw_fiber_architecture(self):
        """Draw fiber architecture visualization."""
        if self.fiber_architecture is None:
            return

        # Get inspector highlight state
        highlight_stream = getattr(self, 'inspector_highlight_stream', None)
        highlight_level = getattr(self, 'inspector_highlight_level', None)

        glDisable(GL_LIGHTING)

        # Draw waypoints
        for stream_idx, waypoint_group in enumerate(self.waypoints):
            # Check if stream should be drawn (with bounds check)
            if self.draw_contour_stream is not None and stream_idx < len(self.draw_contour_stream) and self.draw_contour_stream[stream_idx]:
                for level_idx, waypoints in enumerate(waypoint_group):
                    is_highlighted = (highlight_stream == stream_idx and highlight_level == level_idx)

                    if is_highlighted:
                        glPointSize(5)
                        glColor4f(0.3, 0.6, 0.9, 1)  # Subtle blue
                    else:
                        glPointSize(3)
                        glColor4f(1, 0, 0, 1)  # Red

                    glBegin(GL_POINTS)
                    for p in waypoints:
                        glVertex3fv(p)
                    glEnd()

        # Draw fiber lines
        glLineWidth(2)
        glColor4f(0.75, 0, 0, 1)  # Dark red
        for stream_idx, waypoint_group in enumerate(self.waypoints):
            # Check if stream should be drawn (with bounds check)
            if self.draw_contour_stream is not None and stream_idx < len(self.draw_contour_stream) and self.draw_contour_stream[stream_idx]:
                for contour_idx in range(len(waypoint_group) - 1):
                    glBegin(GL_LINES)
                    for p1, p2 in zip(waypoint_group[contour_idx], waypoint_group[contour_idx + 1]):
                        glVertex3fv(p1)
                        glVertex3fv(p2)
                    glEnd()

        glEnable(GL_LIGHTING)

    def find_waypoints(self, bounding_plane_info, fiber_architecture, is_origin=False):
        """
        Compute waypoints using Mean Value Coordinates (MVC) with original P->Q
        bounding plane correspondence.

        Uses Q points (normalized template positions in bounding plane) directly
        as the MVC polygon. MVC weights are computed from fiber samples to Q polygon,
        then applied to corresponding contour vertices (P).

        Args:
            bounding_plane_info: Dictionary with bounding plane data including 'contour_match'
            fiber_architecture: Fiber sampling points (inside unit square in 2D, [0,1] range)
            is_origin: Whether this is at the origin end

        Returns:
            (Qs_2d, waypoints, mvc_weights): Q positions (for visualization), computed waypoints,
                and MVC weights array (shape: num_fibers x num_vertices)
        """
        # Get contour match (P->Q correspondence)
        contour_match = bounding_plane_info.get('contour_match')
        if contour_match is None:
            return np.array([]), np.array([]), np.array([])

        # Extract P (contour vertices) and Q (bounding plane template positions)
        Ps = np.array([pair[0] for pair in contour_match])  # 3D contour points
        Qs = np.array([pair[1] for pair in contour_match])  # 2D/3D template points

        n_verts = len(Ps)
        if n_verts < 3:
            return np.array([]), np.array([]), np.array([])

        # Project Q to 2D if needed (Q should already be in bounding plane coords)
        basis_x = bounding_plane_info.get('basis_x')
        basis_y = bounding_plane_info.get('basis_y')
        center = bounding_plane_info.get('mean', np.mean(Qs, axis=0))
        bp = bounding_plane_info.get('bounding_plane')

        # Use bounding plane parametric coordinates to map Q to [0,1]^2
        # This matches how fiber samples are generated (Sobol points in unit square)
        # Formula: Q = bp[0] + u * (bp[1]-bp[0]) + v * (bp[3]-bp[0])
        # Solve for (u, v) using least squares
        if bp is not None and len(bp) >= 4:
            bp = [np.array(c) for c in bp[:4]]
            edge_x = bp[1] - bp[0]  # u direction
            edge_y = bp[3] - bp[0]  # v direction
            A = np.column_stack([edge_x, edge_y])

            Qs_normalized = np.zeros((n_verts, 2))
            for i, q in enumerate(Qs):
                q = np.array(q)
                rel_q = q - bp[0]
                result, _, _, _ = np.linalg.lstsq(A, rel_q, rcond=None)
                u, v = result[0], result[1]
                Qs_normalized[i] = [np.clip(u, 0, 1), np.clip(v, 0, 1)]
        else:
            # Fallback to min/max normalization if no bounding plane
            if Qs.shape[1] == 3:
                if basis_x is not None and basis_y is not None:
                    Qs_2d = np.zeros((n_verts, 2))
                    for i, q in enumerate(Qs):
                        rel = q - center
                        Qs_2d[i] = [np.dot(rel, basis_x), np.dot(rel, basis_y)]
                else:
                    Qs_2d = Qs[:, :2]
            else:
                Qs_2d = Qs

            q_min = Qs_2d.min(axis=0)
            q_max = Qs_2d.max(axis=0)
            q_range = q_max - q_min
            q_range[q_range < 1e-10] = 1.0
            Qs_normalized = (Qs_2d - q_min) / q_range

        # MVC computation using Q polygon
        fiber_samples = np.array(fiber_architecture)
        EPS = 1e-10
        mvc_polygon = Qs_normalized

        fs = []
        for v in fiber_samples:
            f_found = False
            s_v = [Q - v for Q in mvc_polygon]

            # Check for special cases
            for i in range(n_verts):
                i_plus = (i + 1) % n_verts
                r_i = np.linalg.norm(s_v[i])
                A_i = np.linalg.det(np.array([s_v[i], s_v[i_plus]])) / 2
                D_i = np.dot(s_v[i], s_v[i_plus])

                if r_i < EPS:  # Point coincides with vertex
                    f = np.zeros(n_verts)
                    f[i] = 1
                    fs.append(f)
                    f_found = True
                    break

                if abs(A_i) < EPS and D_i < 0:  # Point on edge
                    r_i_plus = np.linalg.norm(s_v[i_plus])
                    f_i = np.zeros(n_verts)
                    f_i[i] = 1
                    f_i_plus = np.zeros(n_verts)
                    f_i_plus[i_plus] = 1
                    denom = r_i + r_i_plus
                    if denom < EPS:
                        fs.append((f_i + f_i_plus) / 2)
                    else:
                        fs.append((r_i_plus * f_i + r_i * f_i_plus) / denom)
                    f_found = True
                    break

            if f_found:
                continue

            # General MVC computation
            f = np.zeros(n_verts)
            W = 0
            for i in range(n_verts):
                i_plus = (i + 1) % n_verts
                i_minus = (i - 1) % n_verts
                r_i = np.linalg.norm(s_v[i])
                w = 0

                if r_i < EPS:
                    continue

                A_i_minus = np.linalg.det(np.array([s_v[i_minus], s_v[i]])) / 2
                if abs(A_i_minus) > EPS:
                    r_i_minus = np.linalg.norm(s_v[i_minus])
                    D_i_minus = np.dot(s_v[i_minus], s_v[i])
                    w += (r_i_minus - D_i_minus / r_i) / A_i_minus

                A_i = np.linalg.det(np.array([s_v[i], s_v[i_plus]])) / 2
                if abs(A_i) > EPS:
                    r_i_plus = np.linalg.norm(s_v[i_plus])
                    D_i = np.dot(s_v[i], s_v[i_plus])
                    w += (r_i_plus - D_i / r_i) / A_i

                f[i] = w
                W += w

            if abs(W) < EPS:
                fs.append(np.ones(n_verts) / n_verts)
            else:
                fs.append(f / W)

        fs = np.array(fs)

        # Apply MVC weights to corresponding contour vertices
        waypoints = np.dot(fs, Ps)

        return Qs_normalized, waypoints, fs

    def find_waypoints_radial(self, bounding_plane_info, fiber_architecture):
        """
        Compute waypoints using radial interpolation from centroid.

        This method guarantees:
        1. All waypoints are inside the contour (even non-convex)
        2. Same topological position across all contours (index correspondence)

        The waypoint is computed as:
            waypoint = (1 - r) * centroid + r * boundary_point

        Where:
        - r is normalized radius [0, 1) from unit circle sample
        - boundary_point is interpolated between adjacent contour vertices

        Args:
            bounding_plane_info: Dictionary with 'contour_vertices' (N, 3) array
            fiber_architecture: (M, 2) array of (x, y) in [0,1]^2 unit circle

        Returns:
            (radial_2d, waypoints): 2D visualization coords and computed waypoints
        """
        # Get contour vertices
        Ps = bounding_plane_info.get('contour_vertices')
        if Ps is None:
            contour_match = bounding_plane_info.get('contour_match')
            if contour_match is not None:
                Ps = np.array([pair[0] for pair in contour_match])
            else:
                return np.array([]), np.array([])

        Ps = np.array(Ps)
        N = len(Ps)
        if N < 3:
            return np.array([]), np.array([])

        import math
        centroid = np.mean(Ps, axis=0)
        basis_x = bounding_plane_info.get('basis_x')
        basis_y = bounding_plane_info.get('basis_y')

        # Compute actual angles of each contour vertex from centroid
        vertex_angles = []
        if basis_x is not None and basis_y is not None:
            for v in Ps:
                rel = v - centroid
                x_2d = np.dot(rel, basis_x)
                y_2d = np.dot(rel, basis_y)
                angle = math.atan2(y_2d, x_2d)
                if angle < 0:
                    angle += 2 * math.pi
                vertex_angles.append(angle)
        else:
            # Fallback: assume uniform distribution
            vertex_angles = [2 * math.pi * i / N for i in range(N)]

        fiber_samples = np.array(fiber_architecture)
        M = len(fiber_samples)

        # Compute waypoints using actual vertex angles
        waypoints = np.zeros((M, 3))

        for m in range(M):
            x, y = fiber_samples[m]

            # Fiber angle on unit circle (centered at 0.5, 0.5)
            fiber_theta = math.atan2(y - 0.5, x - 0.5)
            if fiber_theta < 0:
                fiber_theta += 2 * math.pi

            # Fiber radius (0 = center, 1 = edge)
            fiber_radius = math.sqrt((x - 0.5)**2 + (y - 0.5)**2) / 0.5
            fiber_radius = min(fiber_radius, 0.99)  # Cap for strict interior

            # Find which two vertices bracket this angle
            i, j, t = find_angle_bracket(vertex_angles, fiber_theta)

            # Boundary point between vertex i and j
            boundary = (1 - t) * Ps[i] + t * Ps[j]

            # Waypoint = blend centroid and boundary by radius
            waypoints[m] = (1 - fiber_radius) * centroid + fiber_radius * boundary

        # For 2D visualization, return the fiber samples as-is (unit circle coords)
        radial_2d = fiber_samples

        return radial_2d, waypoints

    def find_waypoints_angular(self, bounding_plane_info, fiber_samples, num_geodesic_lines=None):
        """
        Compute waypoints using angular method with radial line sampling.

        Fibers are sampled on radial lines from centroid to contour vertices.
        Unit circle is divided by geodesic lines first (uniform),
        then subdivided by contour distance ratios.

        Args:
            bounding_plane_info: Dictionary with 'contour_vertices', 'mean' (centroid)
            fiber_samples: (M, 2) array of (vertex_idx, radius)
                          vertex_idx = which vertex's radial line
                          radius = distance along line (0=center, 1=on vertex)
            num_geodesic_lines: Number of geodesic lines (for subdivision)

        Returns:
            (angular_2d, waypoints): 2D visualization coords and computed waypoints
        """
        # Get contour vertices
        Ps = bounding_plane_info.get('contour_vertices')
        if Ps is None:
            contour_match = bounding_plane_info.get('contour_match')
            if contour_match is not None:
                Ps = np.array([pair[0] for pair in contour_match])
            else:
                return np.array([]), np.array([])

        Ps = np.array(Ps)
        N = len(Ps)
        if N < 3:
            return np.array([]), np.array([])

        # Get centroid (use 'mean' from normalize if available)
        centroid = bounding_plane_info.get('mean', np.mean(Ps, axis=0))

        # Get geodesic indices for subdivision
        geodesic_paths = None
        mesh_vertices = None
        if hasattr(self, '_geodesic_reference_paths') and self._geodesic_reference_paths:
            geodesic_paths = self._geodesic_reference_paths
            num_geodesic_lines = len(geodesic_paths)
        if hasattr(self, 'vertices'):
            mesh_vertices = self.vertices

        if num_geodesic_lines is None or num_geodesic_lines == 0:
            num_geodesic_lines = N  # Fallback: treat all vertices as geodesic

        geodesic_indices = find_geodesic_vertex_indices(Ps, num_geodesic_lines, geodesic_paths, mesh_vertices)

        # Create unit circle angles for each contour vertex
        unit_circle_angles = create_angular_unit_circle_vertices(Ps, geodesic_indices)

        fiber_samples = np.array(fiber_samples)
        M = len(fiber_samples)

        waypoints = np.zeros((M, 3))
        angular_2d = np.zeros((M, 2))

        for m in range(M):
            vertex_idx = int(fiber_samples[m, 0]) % N
            radius = fiber_samples[m, 1]  # 0 = center, 1 = on vertex

            # Get the vertex for this radial line
            V = Ps[vertex_idx]

            # Waypoint on radial line from centroid to vertex
            waypoints[m] = centroid + radius * (V - centroid)

            # For 2D visualization: compute position on unit circle
            angle = unit_circle_angles[vertex_idx]

            angular_2d[m] = [0.5 + radius * 0.5 * np.cos(angle),
                            0.5 + radius * 0.5 * np.sin(angle)]

        return angular_2d, waypoints

    def _compute_waypoint_barycentric_coords(self, skeleton_meshes=None, skeleton=None):
        """
        Compute barycentric coordinates for each waypoint within tetrahedra.
        Stores mapping for later use when updating waypoints after tet sim.

        Waypoint structure: self.waypoints[stream_idx][contour_idx] = array of shape (num_fibers, 3)

        Special handling for endpoints (first/last contour):
        - These are at the muscle caps and may be outside tetrahedra
        - They should follow their attached skeletons instead
        """
        if not hasattr(self, 'waypoints') or len(self.waypoints) == 0:
            return
        if not hasattr(self, 'tet_tetrahedra') or self.tet_tetrahedra is None:
            return
        if not hasattr(self, 'tet_vertices') or self.tet_vertices is None:
            return

        # Store original waypoints for reset
        if not hasattr(self, 'waypoints_original'):
            self.waypoints_original = [
                [np.array(wp).copy() for wp in stream]
                for stream in self.waypoints
            ]

        # waypoint_bary_coords[stream_idx][contour_idx] = list of (tet_idx, bary_coords) for each fiber
        # For endpoints: ('skeleton', body_name, local_pos) instead of (tet_idx, bary)
        self.waypoint_bary_coords = []

        tet_verts = np.array(self.tet_vertices)
        tetrahedra = np.array(self.tet_tetrahedra)

        # Get skeleton names for attachment lookup
        skeleton_names = list(skeleton_meshes.keys()) if skeleton_meshes else []

        # Pre-build bone bounding boxes for fast inside-bone checks
        self._bone_bounds = {}  # body_name -> (min_bound, max_bound, rotation, translation)
        if skeleton is not None and skeleton_meshes is not None:
            for mesh_name, skel_mesh in skeleton_meshes.items():
                if not hasattr(skel_mesh, 'vertices') or skel_mesh.vertices is None:
                    continue

                body_node = skeleton.getBodyNode(mesh_name)
                if body_node is None:
                    for i in range(skeleton.getNumBodyNodes()):
                        bn = skeleton.getBodyNode(i)
                        bn_name = bn.getName()
                        if mesh_name.lower() in bn_name.lower() or bn_name.lower() in mesh_name.lower():
                            body_node = bn
                            break

                if body_node is None:
                    continue

                try:
                    world_transform = body_node.getWorldTransform()
                    rotation = world_transform.rotation()
                    translation = world_transform.translation()
                    world_verts = (rotation @ (np.array(skel_mesh.vertices) * MESH_SCALE).T).T + translation
                    min_bound = np.min(world_verts, axis=0)
                    max_bound = np.max(world_verts, axis=0)
                    # Shrink bounds slightly to only catch points clearly inside
                    margin = (max_bound - min_bound) * 0.1
                    self._bone_bounds[body_node.getName()] = (min_bound + margin, max_bound - margin, rotation, translation)
                except Exception:
                    continue

        embedded_count = 0
        skeleton_count = 0
        total_count = 0
        clamped_count = 0

        for stream_idx, stream in enumerate(self.waypoints):
            stream_bary = []
            num_contours = len(stream)

            for contour_idx, contour_wps in enumerate(stream):
                contour_wps = np.array(contour_wps)
                if contour_wps.ndim == 1:
                    contour_wps = contour_wps.reshape(1, -1)

                if contour_wps.shape[-1] != 3:
                    stream_bary.append(None)
                    continue

                # Check if this is an endpoint (first or last contour)
                is_origin = (contour_idx == 0)
                is_insertion = (contour_idx == num_contours - 1)
                is_endpoint = is_origin or is_insertion

                contour_bary = []
                for fiber_idx in range(len(contour_wps)):
                    point = contour_wps[fiber_idx]
                    total_count += 1

                    # For endpoints, attach to skeleton instead of tetrahedra
                    if is_endpoint and skeleton is not None and hasattr(self, 'attach_skeletons'):
                        body_name = self._get_endpoint_body_name(
                            stream_idx, is_origin, skeleton_meshes, skeleton_names, skeleton
                        )
                        if body_name is not None:
                            body_node = skeleton.getBodyNode(body_name)
                            if body_node is not None:
                                # Compute local position in body frame
                                world_transform = body_node.getWorldTransform()
                                rotation = world_transform.rotation()
                                translation = world_transform.translation()
                                local_pos = rotation.T @ (point - translation)
                                contour_bary.append(('skeleton', body_name, local_pos.copy()))
                                skeleton_count += 1
                                continue

                    # Check if this waypoint is inside any bone bounding box
                    attached_to_bone = False
                    if skeleton is not None and hasattr(self, '_bone_bounds'):
                        for body_name, (min_b, max_b, rotation, translation) in self._bone_bounds.items():
                            if np.all(point >= min_b) and np.all(point <= max_b):
                                local_pos = rotation.T @ (point - translation)
                                contour_bary.append(('skeleton', body_name, local_pos.copy()))
                                skeleton_count += 1
                                attached_to_bone = True
                                break

                    if attached_to_bone:
                        continue

                    # For interior points not inside bones, use tetrahedra
                    tet_idx, bary, was_inside = self._find_containing_tet(point, tet_verts, tetrahedra)
                    if tet_idx is not None:
                        # Store whether point was truly inside (for stability during updates)
                        contour_bary.append(('tet', tet_idx, bary, was_inside))
                        embedded_count += 1
                        if not was_inside:
                            clamped_count += 1
                    else:
                        contour_bary.append(None)

                stream_bary.append(contour_bary)
            self.waypoint_bary_coords.append(stream_bary)

        failed_count = total_count - embedded_count - skeleton_count
        msg = f"  Waypoints: {embedded_count} in tetrahedra"
        if clamped_count > 0:
            msg += f" ({clamped_count} clamped/outside)"
        msg += f", {skeleton_count} attached to skeleton, {total_count} total"
        if failed_count > 0:
            msg += f", {failed_count} FAILED"
        print(msg)

    def _get_endpoint_body_name(self, stream_idx, is_origin, skeleton_meshes, skeleton_names, skeleton):
        """Get the body name for an endpoint based on attach_skeletons."""
        if not hasattr(self, 'attach_skeletons') or stream_idx >= len(self.attach_skeletons):
            return None

        attachments = self.attach_skeletons[stream_idx]
        skel_idx = attachments[0] if is_origin else attachments[1]

        if skel_idx >= len(skeleton_names):
            return None

        mesh_name = skeleton_names[skel_idx]

        # Try to find corresponding DART body node
        body_node = skeleton.getBodyNode(mesh_name)
        if body_node is not None:
            return mesh_name

        # Search by partial match
        for i in range(skeleton.getNumBodyNodes()):
            bn = skeleton.getBodyNode(i)
            bn_name = bn.getName()
            if mesh_name.lower() in bn_name.lower() or bn_name.lower() in mesh_name.lower():
                return bn_name

        return None

    def _find_containing_tet(self, point, tet_verts, tetrahedra):
        """
        Find the tetrahedron containing a point and compute barycentric coordinates.
        For points outside the mesh, finds the closest tetrahedron and uses clamped
        barycentric coordinates to preserve the original waypoint positions.

        Returns:
            (tet_idx, barycentric_coords, was_inside) - was_inside=True if point was truly inside
        """
        # For efficiency, first find nearest tetrahedra by centroid
        tet_centroids = np.mean(tet_verts[tetrahedra], axis=1)
        dists = np.linalg.norm(tet_centroids - point, axis=1)
        sorted_indices = np.argsort(dists)

        best_tet_idx = None
        best_bary = None
        best_min_coord = -float('inf')  # Track least-negative barycentric coord

        # Search through tetrahedra - check more if point is far outside
        num_to_check = min(500, len(tetrahedra))
        for tet_idx in sorted_indices[:num_to_check]:
            tet = tetrahedra[tet_idx]
            v0, v1, v2, v3 = tet_verts[tet[0]], tet_verts[tet[1]], tet_verts[tet[2]], tet_verts[tet[3]]

            bary = self._compute_barycentric(point, v0, v1, v2, v3)
            if bary is None:
                continue

            min_coord = np.min(bary)

            # Point is inside this tetrahedron (with small tolerance)
            if min_coord >= -0.01:
                return tet_idx, bary, True  # was_inside=True

            # Track the best candidate (least outside)
            if min_coord > best_min_coord:
                best_min_coord = min_coord
                best_tet_idx = tet_idx
                best_bary = bary

        # Use best candidate with clamped barycentric coords
        if best_tet_idx is not None:
            clamped_bary = np.maximum(best_bary, 0.0)
            clamped_bary /= np.sum(clamped_bary)  # Renormalize
            return best_tet_idx, clamped_bary, False  # was_inside=False

        # Fallback: use the closest tetrahedron by centroid
        closest_tet_idx = sorted_indices[0]
        tet = tetrahedra[closest_tet_idx]
        v0, v1, v2, v3 = tet_verts[tet[0]], tet_verts[tet[1]], tet_verts[tet[2]], tet_verts[tet[3]]
        bary = self._compute_barycentric(point, v0, v1, v2, v3)
        if bary is not None:
            clamped_bary = np.maximum(bary, 0.0)
            if np.sum(clamped_bary) > 0:
                clamped_bary /= np.sum(clamped_bary)
                return closest_tet_idx, clamped_bary, False  # was_inside=False

        # Ultimate fallback: assign to closest tet with equal weights
        return closest_tet_idx, np.array([0.25, 0.25, 0.25, 0.25]), False

    def _compute_barycentric(self, p, v0, v1, v2, v3):
        """
        Compute barycentric coordinates of point p in tetrahedron (v0, v1, v2, v3).

        Returns:
            Array of 4 barycentric coordinates, or None if degenerate
        """
        # Construct matrix for barycentric computation
        # [v1-v0, v2-v0, v3-v0]^T * [l1, l2, l3]^T = p - v0
        # l0 = 1 - l1 - l2 - l3
        T = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
        try:
            det = np.linalg.det(T)
            if abs(det) < 1e-10:
                return None
            bary_123 = np.linalg.solve(T, p - v0)
            bary_0 = 1.0 - np.sum(bary_123)
            return np.array([bary_0, bary_123[0], bary_123[1], bary_123[2]])
        except Exception:
            return None

    def _update_waypoints_from_tet(self, skeleton=None):
        """
        Update waypoint positions using deformed tetrahedra and skeleton transforms.

        Waypoint structure: self.waypoints[stream_idx][contour_idx] = array of shape (num_fibers, 3)

        Handles two types of waypoint attachments:
        - ('tet', tet_idx, bary): Use barycentric interpolation in deformed tetrahedra
        - ('skeleton', body_name, local_pos): Transform using skeleton body
        """
        if not hasattr(self, 'waypoint_bary_coords') or len(self.waypoint_bary_coords) == 0:
            return
        if not hasattr(self, 'tet_vertices') or self.tet_vertices is None:
            return
        if not hasattr(self, 'tet_tetrahedra') or self.tet_tetrahedra is None:
            return

        tet_verts = np.array(self.tet_vertices)
        tetrahedra = np.array(self.tet_tetrahedra)

        tet_count = 0
        skel_count = 0
        skipped_count = 0
        clamped_count = 0

        for stream_idx, stream_bary in enumerate(self.waypoint_bary_coords):
            if stream_idx >= len(self.waypoints):
                continue

            for contour_idx, contour_bary in enumerate(stream_bary):
                if contour_bary is None:
                    continue
                if contour_idx >= len(self.waypoints[stream_idx]):
                    continue

                contour_wps = np.array(self.waypoints[stream_idx][contour_idx])
                if contour_wps.ndim == 1:
                    contour_wps = contour_wps.reshape(1, -1)

                # Update each fiber point
                for fiber_idx, bary_data in enumerate(contour_bary):
                    if bary_data is None:
                        skipped_count += 1
                        continue
                    if fiber_idx >= len(contour_wps):
                        continue

                    if bary_data[0] == 'skeleton':
                        # Endpoint attached to skeleton
                        _, body_name, local_pos = bary_data
                        if skeleton is not None:
                            body_node = skeleton.getBodyNode(body_name)
                            if body_node is not None:
                                world_transform = body_node.getWorldTransform()
                                rotation = world_transform.rotation()
                                translation = world_transform.translation()
                                new_pos = rotation @ local_pos + translation
                                contour_wps[fiber_idx] = new_pos
                                skel_count += 1

                    elif bary_data[0] == 'tet':
                        # Interior point in tetrahedron
                        # Format: ('tet', tet_idx, bary, was_inside) or legacy ('tet', tet_idx, bary)
                        if len(bary_data) >= 4:
                            _, tet_idx, bary, was_inside = bary_data
                        else:
                            _, tet_idx, bary = bary_data
                            was_inside = True

                        if tet_idx >= len(tetrahedra):
                            skipped_count += 1
                            continue

                        # Get deformed tetrahedron vertices
                        tet = tetrahedra[tet_idx]
                        v0 = tet_verts[tet[0]]
                        v1 = tet_verts[tet[1]]
                        v2 = tet_verts[tet[2]]
                        v3 = tet_verts[tet[3]]

                        # Compute new position using barycentric interpolation
                        new_pos = bary[0] * v0 + bary[1] * v1 + bary[2] * v2 + bary[3] * v3

                        contour_wps[fiber_idx] = new_pos
                        tet_count += 1
                        if not was_inside:
                            clamped_count += 1

                # Update the waypoints array
                self.waypoints[stream_idx][contour_idx] = contour_wps

        if tet_count + skel_count > 0:
            msg = f"  Updated waypoints: {tet_count} from tetrahedra, {skel_count} from skeleton"
            if skipped_count > 0:
                msg += f", {skipped_count} SKIPPED (not embedded!)"
            if clamped_count > 0:
                msg += f", {clamped_count} clamped (outside mesh)"
            print(msg)

    def update_waypoints_from_viper(self):
        """Update viper_waypoints array from VIPER rod positions."""
        if self.viper_sim is None:
            return

        if not hasattr(self, 'viper_waypoints'):
            return

        # Get positions from VIPER rods
        positions = self.viper_sim.get_positions_flat()
        if positions is None:
            return

        # Update viper_waypoints structure
        idx = 0
        for rod_idx in range(len(self.viper_waypoints)):
            for point_idx in range(len(self.viper_waypoints[rod_idx])):
                if idx < len(positions):
                    self.viper_waypoints[rod_idx][point_idx] = positions[idx].copy()
                    idx += 1

    def _ensure_waypoints_inside_mesh(self, vertices_original, vertices_smoothed, faces):
        """
        Adjust smoothed mesh to ensure all waypoints remain inside.
        If smoothing moved surface inward past a waypoint, expand locally.

        Returns: adjusted vertices
        """
        if not hasattr(self, 'waypoints') or self.waypoints is None or len(self.waypoints) == 0:
            return vertices_smoothed

        vertices = vertices_smoothed.copy()

        # Collect all waypoint positions
        all_waypoints = []
        for stream in self.waypoints:
            for contour_wps in stream:
                if contour_wps is not None and len(contour_wps) > 0:
                    wps = np.array(contour_wps)
                    if wps.ndim == 1:
                        wps = wps.reshape(1, -1)
                    for wp in wps:
                        all_waypoints.append(wp)

        if len(all_waypoints) == 0:
            return vertices

        all_waypoints = np.array(all_waypoints)
        print(f"  Checking {len(all_waypoints)} waypoints against smoothed mesh...")

        # Build trimesh for inside/outside check
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # Find closest surface point for each waypoint
        closest_pts, distances, face_ids = mesh.nearest.on_surface(all_waypoints)

        # Check which waypoints are "outside" (smoothing moved surface past them)
        # Use dot product with face normal to determine inside/outside
        normals = mesh.face_normals[face_ids]
        to_waypoint = all_waypoints - closest_pts
        dots = np.einsum('ij,ij->i', to_waypoint, normals)

        # Find waypoints that need the mesh to be expanded
        margin = 0.001  # 1mm margin
        outside_mask = (dots > 0) & (distances < 0.01)  # Close but on wrong side

        if not np.any(outside_mask):
            print(f"  All waypoints inside smoothed mesh")
            return vertices

        outside_count = np.sum(outside_mask)
        print(f"  {outside_count} waypoints outside smoothed mesh, adjusting...")

        # For each outside waypoint, push nearby vertices outward
        vert_tree = cKDTree(vertices)

        adjusted_count = 0
        for i in np.where(outside_mask)[0]:
            wp = all_waypoints[i]
            closest_pt = closest_pts[i]
            normal = normals[i]

            # Find vertices near the closest point
            nearby_indices = vert_tree.query_ball_point(closest_pt, r=0.02)  # 2cm radius

            if len(nearby_indices) == 0:
                continue

            # Push these vertices outward along normal to include waypoint
            # Target: waypoint should be margin distance inside the surface
            wp_dist_along_normal = np.dot(wp - closest_pt, normal)
            push_amount = wp_dist_along_normal + margin

            if push_amount > 0:
                for vi in nearby_indices:
                    # Weight by distance to closest point
                    dist_to_closest = np.linalg.norm(vertices[vi] - closest_pt)
                    weight = max(0, 1 - dist_to_closest / 0.02)  # Linear falloff
                    vertices[vi] += normal * push_amount * weight
                    adjusted_count += 1

        print(f"  Adjusted {adjusted_count} vertex positions")
        return vertices

    def _update_contours_from_smoothed_mesh(self, vertices_original, vertices_smoothed):
        """
        Update self.contours to match the smoothed mesh vertices.
        This ensures waypoints computed from contours will be inside the tets.
        """
        if self.contours is None or len(self.contours) == 0:
            return

        # Build KD-tree from original mesh vertices
        tree = cKDTree(vertices_original)

        # Compute displacement for each original vertex
        displacements = vertices_smoothed - vertices_original

        updated_count = 0
        for stream_idx, stream in enumerate(self.contours):
            for level_idx, contour in enumerate(stream):
                if contour is None or len(contour) == 0:
                    continue

                new_contour = []
                for pt in contour:
                    pt = np.array(pt)
                    # Find nearest original mesh vertex
                    dist, idx = tree.query(pt)
                    if dist < 1e-4:  # Close enough - apply displacement
                        new_pt = pt + displacements[idx]
                        updated_count += 1
                    else:
                        # Not matched - keep original
                        new_pt = pt
                    new_contour.append(new_pt)

                self.contours[stream_idx][level_idx] = np.array(new_contour)

        print(f"  Updated {updated_count} contour points to match smoothed mesh")

    def _align_stream_contours(self, stream_contours):
        """
        Align consecutive contours in a stream to minimize twist.
        Uses minimum sum-of-squared-distances to find optimal rotation and orientation.
        """
        if len(stream_contours) < 2:
            return [np.array(c) for c in stream_contours]

        aligned = [np.array(stream_contours[0])]

        for i in range(1, len(stream_contours)):
            prev_contour = aligned[-1]
            curr_contour = np.array(stream_contours[i])

            # Find best alignment (considering both orientations and all rotations)
            aligned_contour = self._find_best_alignment(prev_contour, curr_contour)
            aligned.append(aligned_contour)

        return aligned

    def _find_best_alignment(self, ref_contour, target_contour):
        """
        Find the best alignment of target_contour to ref_contour.
        Tests all rotations and both forward/reverse orientations.
        Returns the aligned contour that minimizes sum of squared distances.
        """
        n_ref = len(ref_contour)
        n_target = len(target_contour)

        if n_target == 0:
            return target_contour

        best_contour = target_contour.copy()
        best_cost = float('inf')

        # Try both orientations
        for reverse in [False, True]:
            if reverse:
                test_contour = target_contour[::-1].copy()
            else:
                test_contour = target_contour.copy()

            # Try all rotations
            n = len(test_contour)
            for offset in range(n):
                rotated = np.roll(test_contour, -offset, axis=0)

                # Compute cost: sum of squared distances to corresponding points
                if n_ref == n:
                    # Same size: direct correspondence
                    cost = np.sum((rotated - ref_contour) ** 2)
                else:
                    # Different size: use interpolated correspondence
                    cost = self._compute_alignment_cost_variable(ref_contour, rotated)

                if cost < best_cost:
                    best_cost = cost
                    best_contour = rotated.copy()

        return best_contour

    def _compute_alignment_cost_variable(self, contour_a, contour_b):
        """
        Compute alignment cost between contours of different sizes.
        Maps each point in A to interpolated point in B.
        """
        n_a = len(contour_a)
        n_b = len(contour_b)

        total_cost = 0.0
        for i in range(n_a):
            # Map index in A to fractional index in B
            t = i * n_b / n_a
            j = int(t)
            frac = t - j
            j_next = (j + 1) % n_b

            # Interpolated point in B
            interp_point = (1 - frac) * contour_b[j % n_b] + frac * contour_b[j_next]
            total_cost += np.sum((contour_a[i] - interp_point) ** 2)

        return total_cost

    # ============================================================================
    # MVC-BASED WAYPOINT UPDATE FROM DEFORMED CONTOURS
    # ============================================================================

    def build_contour_vertex_mapping(self):
        """
        Map each contour vertex to its nearest tet vertex for deformation tracking.

        This builds a mapping structure:
        contour_to_tet_mapping[stream_idx][level_idx] = array of tet vertex indices

        After tet simulation, deformed contours can be retrieved using get_deformed_contours().
        """
        if not hasattr(self, 'tet_vertices') or self.tet_vertices is None:
            print("No tet vertices - cannot build contour mapping")
            return
        if not hasattr(self, 'contours') or self.contours is None:
            print("No contours - cannot build contour mapping")
            return

        tet_tree = cKDTree(np.array(self.tet_vertices))

        # contour_to_tet_mapping[stream_idx][level_idx] = array of tet vertex indices
        self.contour_to_tet_mapping = []

        total_mapped = 0
        for stream_idx, stream in enumerate(self.contours):
            stream_mapping = []
            for level_idx, contour in enumerate(stream):
                if contour is None or len(contour) == 0:
                    stream_mapping.append(np.array([], dtype=int))
                    continue

                contour_arr = np.array(contour)
                distances, indices = tet_tree.query(contour_arr)
                stream_mapping.append(indices)
                total_mapped += len(indices)
            self.contour_to_tet_mapping.append(stream_mapping)

        print(f"Built contour-to-tet mapping: {total_mapped} contour vertices mapped")

    def get_deformed_contours(self):
        """
        Get contours with positions updated from deformed tet mesh.

        Returns:
            List of streams, each containing lists of contour arrays with deformed positions.
            Same structure as self.contours but with positions from tet_vertices.
        """
        if not hasattr(self, 'contour_to_tet_mapping') or self.contour_to_tet_mapping is None:
            self.build_contour_vertex_mapping()

        if self.contour_to_tet_mapping is None:
            return self.contours  # Fallback to original contours

        if not hasattr(self, 'tet_vertices') or self.tet_vertices is None:
            return self.contours

        tet_verts = np.array(self.tet_vertices)
        deformed_contours = []

        for stream_idx, stream in enumerate(self.contours):
            deformed_stream = []
            for level_idx, contour in enumerate(stream):
                if contour is None or len(contour) == 0:
                    deformed_stream.append(np.array(contour) if contour is not None else None)
                    continue

                mapping = self.contour_to_tet_mapping[stream_idx][level_idx]
                if len(mapping) == 0:
                    deformed_stream.append(np.array(contour))
                    continue

                # Get deformed positions from tet vertices
                deformed = tet_verts[mapping]
                deformed_stream.append(deformed)
            deformed_contours.append(deformed_stream)

        return deformed_contours

    def _update_bounding_plane_from_deformed(self, bp_info, deformed_contour):
        """
        Update bounding plane info with deformed contour positions.

        Args:
            bp_info: Original bounding plane info dictionary
            deformed_contour: (N, 3) array of deformed contour positions

        Returns:
            Updated bounding plane info with deformed positions
        """
        bp_deformed = bp_info.copy()

        # Store deformed contour vertices directly
        bp_deformed['contour_vertices'] = np.array(deformed_contour).copy()

        # Update mean (centroid)
        bp_deformed['mean'] = np.mean(deformed_contour, axis=0)

        # Update contour_match with deformed positions
        old_match = bp_info.get('contour_match', [])
        old_match_orig = bp_info.get('contour_match_orig', old_match)

        # Build new contour_match using deformed positions
        new_match = []
        new_match_orig = []
        for i, (old_p, template_p) in enumerate(old_match):
            if i < len(deformed_contour):
                new_match.append((deformed_contour[i], template_p))
            else:
                new_match.append((old_p, template_p))

        for i, (old_p, template_p) in enumerate(old_match_orig):
            if i < len(deformed_contour):
                new_match_orig.append((deformed_contour[i], template_p))
            else:
                new_match_orig.append((old_p, template_p))

        bp_deformed['contour_match'] = new_match
        bp_deformed['contour_match_orig'] = new_match_orig

        # Recompute basis vectors from deformed contour (PCA)
        centered = deformed_contour - bp_deformed['mean']
        if len(centered) >= 3:
            try:
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                idx = np.argsort(eigenvalues)[::-1]
                bp_deformed['basis_x'] = eigenvectors[:, idx[0]]
                bp_deformed['basis_y'] = eigenvectors[:, idx[1]]
                if len(idx) > 2:
                    bp_deformed['basis_z'] = eigenvectors[:, idx[2]]
            except Exception as e:
                print(f"  Warning: Could not recompute basis vectors: {e}")

        # Update bounding plane corners if they exist
        bp_corners = bp_info.get('bounding_plane')
        if bp_corners is not None and len(bp_corners) == 4:
            # Recompute bounding plane from deformed contour
            mean = bp_deformed['mean']
            basis_x = bp_deformed.get('basis_x', np.array([1, 0, 0]))
            basis_y = bp_deformed.get('basis_y', np.array([0, 1, 0]))

            # Project deformed contour to 2D
            local_coords = []
            for p in deformed_contour:
                rel = p - mean
                x = np.dot(rel, basis_x)
                y = np.dot(rel, basis_y)
                local_coords.append([x, y])
            local_coords = np.array(local_coords)

            # Compute bounding box in local coordinates
            min_x, min_y = local_coords.min(axis=0)
            max_x, max_y = local_coords.max(axis=0)

            # Convert back to 3D
            v0 = mean + min_x * basis_x + min_y * basis_y
            v1 = mean + max_x * basis_x + min_y * basis_y
            v2 = mean + max_x * basis_x + max_y * basis_y
            v3 = mean + min_x * basis_x + max_y * basis_y
            bp_deformed['bounding_plane'] = [v0, v1, v2, v3]

        return bp_deformed

    def recompute_waypoints_from_deformed_contours(self):
        """
        Recompute all waypoints using MVC on deformed contours.

        This method uses the deformed contour positions (from tet mesh deformation)
        to recompute waypoints via Mean Value Coordinates, avoiding barycentric
        interpolation artifacts that can cause fiber bending.

        The fiber architecture samples (normalized 2D positions) are preserved
        from the initial computation; only the contour geometry changes.
        """
        if not hasattr(self, 'tet_vertices') or self.tet_vertices is None:
            print("No tet vertices - cannot recompute waypoints")
            return

        if not hasattr(self, 'bounding_planes') or self.bounding_planes is None:
            print("No bounding planes - cannot recompute waypoints")
            return

        if not hasattr(self, 'fiber_architecture') or self.fiber_architecture is None:
            print("No fiber architecture - cannot recompute waypoints")
            return

        if not hasattr(self, 'waypoints') or self.waypoints is None or len(self.waypoints) == 0:
            print("No waypoints - cannot recompute")
            return

        # Get deformed contours
        deformed_contours = self.get_deformed_contours()

        total_recomputed = 0

        for stream_idx, stream in enumerate(self.waypoints):
            if stream_idx >= len(self.bounding_planes):
                continue
            if stream_idx >= len(deformed_contours):
                continue

            for level_idx, level_waypoints in enumerate(stream):
                if level_idx >= len(self.bounding_planes[stream_idx]):
                    continue
                if level_idx >= len(deformed_contours[stream_idx]):
                    continue

                # Get deformed contour for this level
                deformed_contour = deformed_contours[stream_idx][level_idx]
                if deformed_contour is None or len(deformed_contour) == 0:
                    continue

                # Get original bounding plane info
                bp_info = self.bounding_planes[stream_idx][level_idx]

                # Update bounding plane with deformed contour positions
                bp_info_deformed = self._update_bounding_plane_from_deformed(
                    bp_info, deformed_contour
                )

                # Recompute waypoints using MVC on deformed contour
                is_origin = (level_idx == 0)
                try:
                    _, new_waypoints = self.find_waypoints(
                        bp_info_deformed,
                        self.fiber_architecture,
                        is_origin=is_origin
                    )

                    self.waypoints[stream_idx][level_idx] = new_waypoints
                    total_recomputed += len(new_waypoints)
                except Exception as e:
                    print(f"  Warning: Failed to recompute waypoints for stream {stream_idx} level {level_idx}: {e}")

        print(f"Recomputed {total_recomputed} waypoints from deformed contours using MVC")
