"""
ARAP (As-Rigid-As-Possible) solver backends.
Supports CPU (NumPy/SciPy) and GPU (PyTorch) implementations.
"""

import numpy as np
from abc import ABC, abstractmethod


class ARAPBackend(ABC):
    """Abstract base class for ARAP solver backends."""

    def __init__(self):
        self.name = "base"
        self.is_gpu = False

    @abstractmethod
    def build_system(self, num_vertices, neighbors, weights, fixed_mask, regularization=1e-6):
        """Build the ARAP system matrix."""
        pass

    @abstractmethod
    def local_step(self, positions, rest_positions, neighbors, weights, rest_edges):
        """Compute optimal rotations for each vertex."""
        pass

    @abstractmethod
    def global_step(self, rest_positions, neighbors, weights, rest_edges,
                    rotations, fixed_mask, fixed_targets, target_edges=None):
        """Solve for new positions given rotations."""
        pass

    @abstractmethod
    def solve(self, positions, rest_positions, neighbors, weights, rest_edges,
              fixed_mask, fixed_targets, max_iterations=20, tolerance=1e-4,
              target_edges=None, verbose=False):
        """Run full ARAP solve."""
        pass


class ARAPBackendCPU(ARAPBackend):
    """CPU backend using NumPy and SciPy sparse solvers."""

    def __init__(self):
        super().__init__()
        self.name = "cpu"
        self.is_gpu = False
        self.solver = None
        self.L = None

    def build_system(self, num_vertices, neighbors, weights, fixed_mask, regularization=1e-6):
        """Build and factorize the Laplacian system matrix."""
        import scipy.sparse
        import scipy.sparse.linalg

        n = num_vertices
        L = scipy.sparse.lil_matrix((n, n))

        for i in range(n):
            if fixed_mask[i]:
                L[i, i] = 1.0
            else:
                weight_sum = 0.0
                for j in neighbors[i]:
                    w = weights.get((i, j), weights.get((j, i), 1.0))
                    L[i, j] = -w
                    weight_sum += w
                L[i, i] = weight_sum + regularization

        self.L = L.tocsc()

        try:
            self.solver = scipy.sparse.linalg.factorized(self.L)
            return True
        except Exception as e:
            print(f"  CPU backend: factorization failed: {e}")
            self.solver = None
            return False

    def local_step(self, positions, rest_positions, neighbors, weights, rest_edges):
        """Compute optimal rotations using SVD."""
        n = len(positions)
        rotations = [np.eye(3) for _ in range(n)]

        for i in range(n):
            if len(neighbors[i]) == 0:
                continue

            # Build covariance matrix
            S = np.zeros((3, 3))
            for j in neighbors[i]:
                w = weights.get((i, j), weights.get((j, i), 1.0))
                # Current edge
                e_curr = positions[i] - positions[j]
                # Rest edge
                e_rest = rest_edges[i].get(j, rest_positions[i] - rest_positions[j])
                S += w * np.outer(e_curr, e_rest)

            # Check for degenerate case
            if np.linalg.norm(S, 'fro') < 1e-10:
                continue

            try:
                U, sigma, Vt = np.linalg.svd(S)
                if sigma[0] < 1e-10:
                    continue
                R = U @ Vt
                if np.linalg.det(R) < 0:
                    U[:, -1] *= -1
                    R = U @ Vt
                if np.isfinite(R).all():
                    rotations[i] = R
            except Exception:
                pass

        return rotations

    def global_step(self, rest_positions, neighbors, weights, rest_edges,
                    rotations, fixed_mask, fixed_targets, target_edges=None):
        """Solve linear system for new positions."""
        import scipy.sparse.linalg

        n = len(rest_positions)

        # Build RHS
        b = np.zeros((n, 3))
        regularization = 1e-6

        for i in range(n):
            if fixed_mask[i]:
                if fixed_targets is not None:
                    idx = np.where(np.arange(n)[fixed_mask] == i)[0]
                    if len(idx) > 0 and idx[0] < len(fixed_targets):
                        b[i] = fixed_targets[idx[0]]
                    else:
                        b[i] = rest_positions[i]
                else:
                    b[i] = rest_positions[i]
            else:
                rhs = regularization * rest_positions[i]
                for j in neighbors[i]:
                    w = weights.get((i, j), weights.get((j, i), 1.0))
                    if target_edges is not None and j in target_edges[i]:
                        e_rest = target_edges[i][j]
                    else:
                        e_rest = rest_edges[i].get(j, rest_positions[i] - rest_positions[j])

                    Ri = rotations[i]
                    Rj = rotations[j]
                    rhs += 0.5 * w * (Ri + Rj) @ e_rest
                b[i] = rhs

        # Solve
        if self.solver is not None:
            new_positions = np.zeros((n, 3))
            for dim in range(3):
                new_positions[:, dim] = self.solver(b[:, dim])
        else:
            new_positions = scipy.sparse.linalg.spsolve(self.L, b)

        return new_positions

    def solve(self, positions, rest_positions, neighbors, weights, rest_edges,
              fixed_mask, fixed_targets, max_iterations=20, tolerance=1e-4,
              target_edges=None, verbose=False):
        """Run full ARAP iteration."""
        positions = positions.copy()
        free_indices = np.where(~fixed_mask)[0]

        # Set fixed positions
        if fixed_targets is not None:
            fixed_indices = np.where(fixed_mask)[0]
            positions[fixed_indices] = fixed_targets

        for iteration in range(max_iterations):
            old_positions = positions.copy()

            # Local step
            rotations = self.local_step(positions, rest_positions, neighbors, weights, rest_edges)

            # Global step
            positions = self.global_step(rest_positions, neighbors, weights, rest_edges,
                                         rotations, fixed_mask, fixed_targets, target_edges)

            # Check for NaN
            if not np.isfinite(positions).all():
                if verbose:
                    print(f"  CPU: Non-finite at iteration {iteration}, reverting")
                positions = old_positions
                break

            # Re-enforce fixed
            if fixed_targets is not None:
                fixed_indices = np.where(fixed_mask)[0]
                positions[fixed_indices] = fixed_targets

            # Check convergence
            if len(free_indices) > 0:
                disp = np.linalg.norm(positions[free_indices] - old_positions[free_indices], axis=1)
                max_disp = np.max(disp)
            else:
                max_disp = 0.0

            if verbose and (iteration + 1) % 5 == 0:
                print(f"  Iter {iteration+1}: max_disp={max_disp:.2e}")

            if max_disp < tolerance:
                if verbose:
                    print(f"  Converged at iteration {iteration+1}, max_disp={max_disp:.2e}")
                return positions, iteration + 1, max_disp

        if verbose:
            print(f"  Max iterations reached, max_disp={max_disp:.2e}")
        return positions, max_iterations, max_disp


class ARAPBackendGPU(ARAPBackend):
    """GPU backend using PyTorch with CUDA."""

    def __init__(self, device=None):
        super().__init__()
        self.name = "gpu"
        self.is_gpu = True

        import torch
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        if self.device.type != 'cuda':
            print("  Warning: CUDA not available, GPU backend will use CPU")

        self.L_indices = None
        self.L_values = None
        self.L_shape = None
        self.num_vertices = None
        # Cached edge data for vectorized operations
        self._edge_i = None
        self._edge_j = None
        self._edge_w = None
        self._edge_rest = None

    def build_system(self, num_vertices, neighbors, weights, fixed_mask, regularization=1e-6):
        """Build sparse Laplacian matrix in PyTorch format."""
        import torch

        # Clear cached data when rebuilding system
        self._edge_i = None
        self._scipy_solver = None

        self.num_vertices = num_vertices
        n = num_vertices

        # Build COO format sparse matrix
        rows = []
        cols = []
        vals = []

        for i in range(n):
            if fixed_mask[i]:
                rows.append(i)
                cols.append(i)
                vals.append(1.0)
            else:
                weight_sum = 0.0
                for j in neighbors[i]:
                    w = weights.get((i, j), weights.get((j, i), 1.0))
                    rows.append(i)
                    cols.append(j)
                    vals.append(-w)
                    weight_sum += w
                rows.append(i)
                cols.append(i)
                vals.append(weight_sum + regularization)

        self.L_indices = torch.tensor([rows, cols], dtype=torch.long, device=self.device)
        self.L_values = torch.tensor(vals, dtype=torch.float64, device=self.device)
        self.L_shape = (n, n)

        # Store as sparse tensor
        self.L_sparse = torch.sparse_coo_tensor(
            self.L_indices, self.L_values, self.L_shape,
            dtype=torch.float64, device=self.device
        ).coalesce()

        # Convert to CSR for faster solve (PyTorch 2.0+)
        try:
            self.L_csr = self.L_sparse.to_sparse_csr()
            self.use_csr = True
        except Exception:
            self.use_csr = False

        return True

    def _precompute_edges(self, neighbors, weights, rest_edges):
        """Precompute edge data for vectorized operations."""
        import torch

        # Build flat edge arrays
        edge_i = []
        edge_j = []
        edge_w = []
        edge_rest = []

        for i, neighs in enumerate(neighbors):
            for j in neighs:
                edge_i.append(i)
                edge_j.append(j)
                edge_w.append(weights.get((i, j), weights.get((j, i), 1.0)))
                if j in rest_edges[i]:
                    edge_rest.append(rest_edges[i][j])
                else:
                    edge_rest.append([0, 0, 0])  # Will be computed from positions

        self._edge_i = torch.tensor(edge_i, dtype=torch.long, device=self.device)
        self._edge_j = torch.tensor(edge_j, dtype=torch.long, device=self.device)
        self._edge_w = torch.tensor(edge_w, dtype=torch.float64, device=self.device)
        self._edge_rest = torch.tensor(edge_rest, dtype=torch.float64, device=self.device)
        self._n_edges = len(edge_i)

    def _build_covariance_batched(self, positions, rest_positions, neighbors, weights, rest_edges, target_edges=None):
        """Build covariance matrices for all vertices using vectorized operations."""
        import torch

        n = len(positions)

        # Precompute edges if not done
        if not hasattr(self, '_edge_i') or self._edge_i is None:
            self._precompute_edges(neighbors, weights, rest_edges)

        # Convert to torch
        pos_t = torch.tensor(positions, dtype=torch.float64, device=self.device)
        rest_t = torch.tensor(rest_positions, dtype=torch.float64, device=self.device)

        # Compute all current edges at once: e_curr = p_j - p_i (matching viewer.py)
        e_curr = pos_t[self._edge_j] - pos_t[self._edge_i]  # (E, 3)

        # Use precomputed rest edges (p_j - p_i) or compute from rest positions
        e_rest = self._edge_rest.clone()
        # Fix zero rest edges (compute from rest positions)
        zero_mask = (e_rest.abs().sum(dim=1) < 1e-10)
        if zero_mask.any():
            e_rest[zero_mask] = rest_t[self._edge_j[zero_mask]] - rest_t[self._edge_i[zero_mask]]

        # Compute weighted outer products: w * e_curr @ e_rest^T
        # Shape: (E, 3, 3)
        weighted_outer = self._edge_w.unsqueeze(1).unsqueeze(2) * torch.bmm(
            e_curr.unsqueeze(2), e_rest.unsqueeze(1)
        )

        # Scatter-add to accumulate per vertex
        S_all = torch.zeros((n, 3, 3), dtype=torch.float64, device=self.device)
        idx_expand = self._edge_i.unsqueeze(1).unsqueeze(2).expand(-1, 3, 3)
        S_all.scatter_add_(0, idx_expand, weighted_outer)

        return S_all

    def local_step(self, positions, rest_positions, neighbors, weights, rest_edges, target_edges=None):
        """Compute optimal rotations using batched SVD on GPU."""
        import torch

        n = len(positions)

        # Build covariance matrices
        S_all = self._build_covariance_batched(positions, rest_positions, neighbors, weights, rest_edges, target_edges)

        # Compute norms to find non-degenerate matrices
        S_norms = torch.linalg.norm(S_all.reshape(n, -1), dim=1)
        valid_mask = S_norms > 1e-10

        # Initialize rotations to identity
        rotations_t = torch.eye(3, dtype=torch.float64, device=self.device).unsqueeze(0).expand(n, -1, -1).clone()

        # Batched SVD for valid matrices
        valid_indices = torch.where(valid_mask)[0]
        if len(valid_indices) > 0:
            S_valid = S_all[valid_indices]

            try:
                U, sigma, Vt = torch.linalg.svd(S_valid)

                # Check for degenerate singular values
                sigma_valid = sigma[:, 0] > 1e-10

                # Compute rotations
                R = torch.bmm(U, Vt)

                # Fix reflections
                det = torch.linalg.det(R)
                reflection_mask = det < 0
                if reflection_mask.any():
                    U_fixed = U.clone()
                    U_fixed[reflection_mask, :, -1] *= -1
                    R[reflection_mask] = torch.bmm(U_fixed[reflection_mask], Vt[reflection_mask])

                # Only use valid rotations (reshape to check all elements per rotation)
                R_finite = torch.isfinite(R).reshape(len(R), -1).all(dim=1)
                final_valid = sigma_valid & R_finite

                # Batch assign valid rotations
                for idx, i in enumerate(valid_indices):
                    if final_valid[idx]:
                        rotations_t[i] = R[idx]

            except Exception as e:
                print(f"  GPU SVD failed: {e}, falling back to identity rotations")

        # Convert back to numpy list format
        rotations = [rotations_t[i].cpu().numpy() for i in range(n)]
        return rotations

    def global_step(self, rest_positions, neighbors, weights, rest_edges,
                    rotations, fixed_mask, fixed_targets, target_edges=None):
        """Solve linear system on GPU using vectorized operations."""
        import torch

        n = len(rest_positions)
        regularization = 1e-6

        # Convert to tensors
        rest_t = torch.tensor(rest_positions, dtype=torch.float64, device=self.device)
        fixed_mask_t = torch.tensor(fixed_mask, dtype=torch.bool, device=self.device)

        # Stack rotations into tensor (n, 3, 3)
        rotations_t = torch.stack([torch.tensor(r, dtype=torch.float64, device=self.device) for r in rotations])

        # Build RHS vectorized using precomputed edges
        # For edge (i, j): e_rest[i][j] = p_j - p_i, but we need p_i - p_j for RHS
        # So contribution = 0.5 * w * (R_i + R_j) @ (-e_rest) = 0.5 * w * (R_i + R_j) @ (p_i - p_j)
        e_rest = self._edge_rest.clone()
        zero_mask = (e_rest.abs().sum(dim=1) < 1e-10)
        if zero_mask.any():
            # rest_edges stores p_j - p_i
            e_rest[zero_mask] = rest_t[self._edge_j[zero_mask]] - rest_t[self._edge_i[zero_mask]]

        # Negate to get (p_i - p_j) direction for RHS
        e_rest_neg = -e_rest

        # Get rotations for each edge endpoint
        R_i = rotations_t[self._edge_i]  # (E, 3, 3)
        R_j = rotations_t[self._edge_j]  # (E, 3, 3)
        R_avg = 0.5 * (R_i + R_j)  # (E, 3, 3)

        # Compute edge contributions: w * R_avg @ (p_i - p_j)
        # e_rest_neg is (E, 3), need to batch matmul: (E, 3, 3) @ (E, 3, 1) -> (E, 3, 1)
        edge_contrib = self._edge_w.unsqueeze(1) * torch.bmm(R_avg, e_rest_neg.unsqueeze(2)).squeeze(2)  # (E, 3)

        # Scatter-add contributions to vertices
        b = torch.zeros((n, 3), dtype=torch.float64, device=self.device)
        b.scatter_add_(0, self._edge_i.unsqueeze(1).expand(-1, 3), edge_contrib)

        # Add regularization term for free vertices
        b[~fixed_mask_t] += regularization * rest_t[~fixed_mask_t]

        # Set fixed vertex RHS
        fixed_indices = torch.where(fixed_mask_t)[0]
        if fixed_targets is not None and len(fixed_targets) > 0:
            fixed_targets_t = torch.tensor(fixed_targets, dtype=torch.float64, device=self.device)
            b[fixed_indices] = fixed_targets_t
        else:
            b[fixed_indices] = rest_t[fixed_indices]

        # Debug: check if edge contributions are symmetric/correct
        # At rest with R=I, b_free should equal what we get from L @ rest_positions
        # print(f"  DEBUG global_step: edge_contrib mean={edge_contrib.mean().item():.6f}, std={edge_contrib.std().item():.6f}")

        # Solve - use scipy sparse solver (more robust than PyTorch CG)
        # GPU speedup comes from batched SVD in local step and vectorized RHS building
        import scipy.sparse
        import scipy.sparse.linalg

        b_np = b.cpu().numpy()
        new_positions_np = np.zeros((n, 3))

        # Use cached scipy solver if available
        if not hasattr(self, '_scipy_solver') or self._scipy_solver is None:
            # Build scipy sparse matrix from PyTorch sparse tensor
            indices = self.L_indices.cpu().numpy()
            values = self.L_values.cpu().numpy()
            L_scipy = scipy.sparse.csr_matrix((values, (indices[0], indices[1])), shape=(n, n))
            try:
                self._scipy_solver = scipy.sparse.linalg.factorized(L_scipy)
            except Exception:
                self._scipy_solver = None
                self._L_scipy = L_scipy

        if self._scipy_solver is not None:
            for dim in range(3):
                new_positions_np[:, dim] = self._scipy_solver(b_np[:, dim])
        else:
            for dim in range(3):
                new_positions_np[:, dim] = scipy.sparse.linalg.spsolve(self._L_scipy, b_np[:, dim])

        return new_positions_np

    def _conjugate_gradient(self, matvec, b, x0=None, max_iter=100, tol=1e-6):
        """Conjugate gradient solver for Ax = b."""
        import torch

        n = len(b)
        if x0 is None:
            x = torch.zeros_like(b)
        else:
            x = x0.clone()

        r = b - matvec(x)
        p = r.clone()
        rs_old = torch.dot(r, r)

        for i in range(max_iter):
            Ap = matvec(p)
            alpha = rs_old / (torch.dot(p, Ap) + 1e-10)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.dot(r, r)

            if torch.sqrt(rs_new) < tol:
                return x, 0

            p = r + (rs_new / (rs_old + 1e-10)) * p
            rs_old = rs_new

        return x, 1  # Did not converge

    def solve(self, positions, rest_positions, neighbors, weights, rest_edges,
              fixed_mask, fixed_targets, max_iterations=20, tolerance=1e-4,
              target_edges=None, verbose=False):
        """Run full ARAP iteration on GPU."""
        import torch

        positions = positions.copy()
        free_indices = np.where(~fixed_mask)[0]

        # Set fixed positions
        if fixed_targets is not None:
            fixed_indices = np.where(fixed_mask)[0]
            positions[fixed_indices] = fixed_targets

        for iteration in range(max_iterations):
            old_positions = positions.copy()

            # Local step (on GPU)
            rotations = self.local_step(positions, rest_positions, neighbors, weights, rest_edges, target_edges)

            # Debug: check rotation deviation from identity
            if verbose and iteration == 0:
                rot_errors = [np.linalg.norm(r - np.eye(3)) for r in rotations]
                print(f"  DEBUG: max rotation deviation from I: {max(rot_errors):.6f}")

            # Global step (on GPU)
            positions = self.global_step(rest_positions, neighbors, weights, rest_edges,
                                         rotations, fixed_mask, fixed_targets, target_edges)

            # Debug: check position change at first iteration
            if verbose and iteration == 0:
                pos_diff = np.linalg.norm(positions - rest_positions, axis=1)
                print(f"  DEBUG: max position diff from rest after iter 0: {np.max(pos_diff):.6f}")

            # Check for NaN
            if not np.isfinite(positions).all():
                if verbose:
                    print(f"  GPU: Non-finite at iteration {iteration}, reverting")
                positions = old_positions
                break

            # Re-enforce fixed
            if fixed_targets is not None:
                fixed_indices = np.where(fixed_mask)[0]
                positions[fixed_indices] = fixed_targets

            # Check convergence
            if len(free_indices) > 0:
                disp = np.linalg.norm(positions[free_indices] - old_positions[free_indices], axis=1)
                max_disp = np.max(disp)
            else:
                max_disp = 0.0

            if verbose and (iteration + 1) % 5 == 0:
                print(f"  Iter {iteration+1}: max_disp={max_disp:.2e}")

            if max_disp < tolerance:
                if verbose:
                    print(f"  Converged at iteration {iteration+1}, max_disp={max_disp:.2e}")
                return positions, iteration + 1, max_disp

        if verbose:
            print(f"  Max iterations reached, max_disp={max_disp:.2e}")
        return positions, max_iterations, max_disp


class ARAPBackendTaichi(ARAPBackend):
    """Taichi backend using CUDA for parallel computation."""

    _initialized = False

    def __init__(self):
        super().__init__()
        self.name = "taichi"
        self.is_gpu = True

        # Initialize Taichi once
        if not ARAPBackendTaichi._initialized:
            import taichi as ti
            ti.init(arch=ti.cuda, offline_cache=True)
            ARAPBackendTaichi._initialized = True

        self.ti = __import__('taichi')

        # Fields will be allocated when needed
        self.n_verts = 0
        self.n_edges = 0
        self._fields_allocated = False

        # Scipy solver cache
        self._scipy_solver = None
        self._L_scipy = None

    def _allocate_fields(self, n_verts, n_edges):
        """Allocate Taichi fields for computation."""
        ti = self.ti

        if self._fields_allocated and self.n_verts == n_verts and self.n_edges == n_edges:
            return

        self.n_verts = n_verts
        self.n_edges = n_edges

        # Vertex data
        self.positions = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        self.rest_positions = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        self.S_matrices = ti.Matrix.field(3, 3, dtype=ti.f64, shape=n_verts)
        self.rhs = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)

        # Edge data
        self.edge_i = ti.field(dtype=ti.i32, shape=n_edges)
        self.edge_j = ti.field(dtype=ti.i32, shape=n_edges)
        self.edge_w = ti.field(dtype=ti.f64, shape=n_edges)
        self.edge_rest = ti.Vector.field(3, dtype=ti.f64, shape=n_edges)

        # Rotation matrices (stored as numpy, computed per iteration)
        self.rotations_np = np.zeros((n_verts, 3, 3))

        self._fields_allocated = True

    def build_system(self, num_vertices, neighbors, weights, fixed_mask, regularization=1e-6):
        """Build sparse Laplacian matrix."""
        import scipy.sparse

        self._scipy_solver = None  # Clear cache

        n = num_vertices
        self.num_vertices = n
        self.fixed_mask = np.array(fixed_mask)
        self.neighbors = neighbors
        self.weights = weights
        self.regularization = regularization

        # Build edge lists
        edge_i = []
        edge_j = []
        edge_w = []

        for i, neighs in enumerate(neighbors):
            for j in neighs:
                edge_i.append(i)
                edge_j.append(j)
                edge_w.append(weights.get((i, j), weights.get((j, i), 1.0)))

        self.edge_i_np = np.array(edge_i, dtype=np.int32)
        self.edge_j_np = np.array(edge_j, dtype=np.int32)
        self.edge_w_np = np.array(edge_w, dtype=np.float64)
        self.n_edges_total = len(edge_i)

        # Build scipy sparse matrix
        rows = []
        cols = []
        vals = []

        for i in range(n):
            if fixed_mask[i]:
                rows.append(i)
                cols.append(i)
                vals.append(1.0)
            else:
                weight_sum = 0.0
                for j in neighbors[i]:
                    w = weights.get((i, j), weights.get((j, i), 1.0))
                    rows.append(i)
                    cols.append(j)
                    vals.append(-w)
                    weight_sum += w
                rows.append(i)
                cols.append(i)
                vals.append(weight_sum + regularization)

        self._L_scipy = scipy.sparse.csr_matrix(
            (vals, (rows, cols)), shape=(n, n)
        )

        # Pre-factorize
        import scipy.sparse.linalg
        try:
            self._scipy_solver = scipy.sparse.linalg.factorized(self._L_scipy)
        except Exception:
            self._scipy_solver = None

        return True

    def _build_covariance_kernel(self):
        """Create Taichi kernel for covariance matrix computation."""
        ti = self.ti

        @ti.kernel
        def compute_covariance(
            positions: ti.template(),
            rest_positions: ti.template(),
            edge_i: ti.template(),
            edge_j: ti.template(),
            edge_w: ti.template(),
            edge_rest: ti.template(),
            S_matrices: ti.template(),
            n_edges: ti.i32
        ):
            # Clear S matrices
            for i in S_matrices:
                S_matrices[i] = ti.Matrix.zero(ti.f64, 3, 3)

            # Accumulate outer products
            for e in range(n_edges):
                i = edge_i[e]
                j = edge_j[e]
                w = edge_w[e]

                # Current edge: p_j - p_i
                e_curr = positions[j] - positions[i]
                # Rest edge: from precomputed
                e_rest_vec = edge_rest[e]

                # Outer product contribution
                for a in ti.static(range(3)):
                    for b in ti.static(range(3)):
                        S_matrices[i][a, b] += w * e_curr[a] * e_rest_vec[b]

        return compute_covariance

    def _build_svd_rotation_kernel(self):
        """Create Taichi kernel to compute rotations from covariance matrices using SVD."""
        ti = self.ti

        @ti.func
        def svd3x3(A: ti.template()) -> ti.template():
            """
            Compute SVD of 3x3 matrix using Jacobi iterations.
            Returns U, sigma (as vector), Vt such that A = U @ diag(sigma) @ Vt
            """
            # Initialize V = I, compute A^T @ A
            V = ti.Matrix.identity(ti.f64, 3)
            AtA = A.transpose() @ A

            # Jacobi iterations to diagonalize AtA -> V @ D @ V^T
            for _ in ti.static(range(10)):  # 10 iterations usually enough for 3x3
                # Off-diagonal elements
                for p, q in ti.static([(0, 1), (0, 2), (1, 2)]):
                    if ti.abs(AtA[p, q]) > 1e-10:
                        # Compute rotation angle
                        tau = ti.cast((AtA[q, q] - AtA[p, p]) / (2.0 * AtA[p, q]), ti.f64)
                        t = ti.cast(0.0, ti.f64)
                        if tau >= 0:
                            t = ti.cast(1.0, ti.f64) / (tau + ti.sqrt(ti.cast(1.0, ti.f64) + tau * tau))
                        else:
                            t = ti.cast(-1.0, ti.f64) / (-tau + ti.sqrt(ti.cast(1.0, ti.f64) + tau * tau))

                        c = ti.cast(1.0, ti.f64) / ti.sqrt(ti.cast(1.0, ti.f64) + t * t)
                        s = t * c

                        # Apply Givens rotation to AtA (from both sides)
                        # AtA = G^T @ AtA @ G
                        for k in ti.static(range(3)):
                            if k != p and k != q:
                                akp = AtA[k, p]
                                akq = AtA[k, q]
                                AtA[k, p] = c * akp - s * akq
                                AtA[p, k] = AtA[k, p]
                                AtA[k, q] = s * akp + c * akq
                                AtA[q, k] = AtA[k, q]

                        app = AtA[p, p]
                        aqq = AtA[q, q]
                        apq = AtA[p, q]

                        AtA[p, p] = c * c * app - 2 * c * s * apq + s * s * aqq
                        AtA[q, q] = s * s * app + 2 * c * s * apq + c * c * aqq
                        AtA[p, q] = ti.cast(0.0, ti.f64)
                        AtA[q, p] = ti.cast(0.0, ti.f64)

                        # Apply to V: V = V @ G
                        for k in ti.static(range(3)):
                            vkp = V[k, p]
                            vkq = V[k, q]
                            V[k, p] = c * vkp - s * vkq
                            V[k, q] = s * vkp + c * vkq

            # Singular values are sqrt of diagonal of AtA
            sigma = ti.Vector([
                ti.sqrt(ti.max(AtA[0, 0], ti.cast(0.0, ti.f64))),
                ti.sqrt(ti.max(AtA[1, 1], ti.cast(0.0, ti.f64))),
                ti.sqrt(ti.max(AtA[2, 2], ti.cast(0.0, ti.f64)))
            ], dt=ti.f64)

            # U = A @ V @ Sigma^-1
            U = ti.Matrix.identity(ti.f64, 3)
            AV = A @ V
            for j in ti.static(range(3)):
                if sigma[j] > 1e-10:
                    for i in ti.static(range(3)):
                        U[i, j] = AV[i, j] / sigma[j]

            return U, sigma, V.transpose()

        @ti.kernel
        def compute_rotations(
            S_matrices: ti.template(),
            rotations: ti.template(),
            n_verts: ti.i32
        ):
            for i in range(n_verts):
                S = S_matrices[i]

                # Check for degenerate matrix
                S_norm = ti.cast(0.0, ti.f64)
                for a in ti.static(range(3)):
                    for b in ti.static(range(3)):
                        S_norm = S_norm + S[a, b] * S[a, b]
                S_norm = ti.sqrt(S_norm)

                if S_norm < 1e-10:
                    # Identity rotation
                    rotations[i] = ti.Matrix.identity(ti.f64, 3)
                else:
                    U, sigma, Vt = svd3x3(S)

                    # R = U @ Vt
                    R = U @ Vt

                    # Check determinant and fix reflection
                    det = R.determinant()
                    if det < 0:
                        # Flip last column of U
                        for k in ti.static(range(3)):
                            U[k, 2] = -U[k, 2]
                        R = U @ Vt

                    rotations[i] = R

        return compute_rotations

    def _build_rhs_kernel(self):
        """Create Taichi kernel for RHS computation."""
        ti = self.ti

        @ti.kernel
        def compute_rhs(
            rest_positions: ti.template(),
            edge_i: ti.template(),
            edge_j: ti.template(),
            edge_w: ti.template(),
            edge_rest: ti.template(),
            rotations: ti.template(),
            rhs: ti.template(),
            regularization: ti.f64,
            n_verts: ti.i32,
            n_edges: ti.i32
        ):
            # Clear RHS
            for i in range(n_verts):
                rhs[i] = ti.Vector.zero(ti.f64, 3)

            # Add edge contributions
            for e in range(n_edges):
                i = edge_i[e]
                j = edge_j[e]
                w = edge_w[e]

                # Rest edge negated: p_i - p_j
                e_rest_neg = -edge_rest[e]

                # R_avg @ e_rest_neg
                R_i = rotations[i]
                R_j = rotations[j]
                R_avg = 0.5 * (R_i + R_j)

                contrib = w * (R_avg @ e_rest_neg)

                for d in ti.static(range(3)):
                    rhs[i][d] += contrib[d]

            # Add regularization term
            for i in range(n_verts):
                rhs[i] += regularization * rest_positions[i]

        return compute_rhs

    def local_step(self, positions, rest_positions, neighbors, weights, rest_edges, target_edges=None):
        """Compute optimal rotations using Taichi for covariance AND SVD (fully on GPU)."""
        ti = self.ti
        n = len(positions)

        # Allocate fields if needed
        if not self._fields_allocated or self.n_verts != n:
            # Build edge rest vectors
            edge_rest_list = []
            for i in range(len(self.edge_i_np)):
                ei = self.edge_i_np[i]
                ej = self.edge_j_np[i]
                if ej in rest_edges[ei]:
                    edge_rest_list.append(rest_edges[ei][ej])
                else:
                    edge_rest_list.append(rest_positions[ej] - rest_positions[ei])

            self.edge_rest_np = np.array(edge_rest_list, dtype=np.float64)
            self._allocate_fields(n, self.n_edges_total)

            # Copy edge data to Taichi fields
            self.edge_i.from_numpy(self.edge_i_np)
            self.edge_j.from_numpy(self.edge_j_np)
            self.edge_w.from_numpy(self.edge_w_np)
            self.edge_rest.from_numpy(self.edge_rest_np)

        # Allocate rotation field if needed
        if not hasattr(self, 'rotations_field_local') or self.rotations_field_local.shape[0] != n:
            self.rotations_field_local = ti.Matrix.field(3, 3, dtype=ti.f64, shape=n)

        # Copy positions to Taichi fields
        self.positions.from_numpy(positions.astype(np.float64))
        self.rest_positions.from_numpy(rest_positions.astype(np.float64))

        # Build kernels if not cached
        if not hasattr(self, '_covariance_kernel'):
            self._covariance_kernel = self._build_covariance_kernel()
        if not hasattr(self, '_svd_rotation_kernel'):
            self._svd_rotation_kernel = self._build_svd_rotation_kernel()

        # Compute covariance matrices on GPU
        self._covariance_kernel(
            self.positions, self.rest_positions,
            self.edge_i, self.edge_j, self.edge_w, self.edge_rest,
            self.S_matrices, self.n_edges_total
        )

        # Compute rotations on GPU using Taichi SVD
        self._svd_rotation_kernel(
            self.S_matrices,
            self.rotations_field_local,
            n
        )

        # Get rotations back to numpy (as list for compatibility)
        rotations_np = self.rotations_field_local.to_numpy()
        rotations = [rotations_np[i] for i in range(n)]

        return rotations

    def global_step(self, rest_positions, neighbors, weights, rest_edges,
                    rotations, fixed_mask, fixed_targets, target_edges=None):
        """Solve linear system using Taichi for RHS, scipy for solve."""
        import scipy.sparse.linalg
        ti = self.ti

        n = len(rest_positions)
        regularization = self.regularization

        # Store rotations in numpy array for Taichi
        rotations_np = np.array(rotations, dtype=np.float64)

        # Allocate rotation field if needed
        if not hasattr(self, 'rotations_field') or self.rotations_field.shape[0] != n:
            self.rotations_field = ti.Matrix.field(3, 3, dtype=ti.f64, shape=n)

        # Copy data to Taichi
        self.rest_positions.from_numpy(rest_positions.astype(np.float64))
        self.rotations_field.from_numpy(rotations_np)

        # Build RHS kernel if not cached
        if not hasattr(self, '_rhs_kernel'):
            self._rhs_kernel = self._build_rhs_kernel()

        # Compute RHS on GPU
        self._rhs_kernel(
            self.rest_positions,
            self.edge_i, self.edge_j, self.edge_w, self.edge_rest,
            self.rotations_field, self.rhs,
            regularization, n, self.n_edges_total
        )

        # Get RHS back to numpy
        b_np = self.rhs.to_numpy()

        # Set fixed vertex RHS
        fixed_indices = np.where(fixed_mask)[0]
        if fixed_targets is not None and len(fixed_targets) > 0:
            b_np[fixed_indices] = fixed_targets
        else:
            b_np[fixed_indices] = rest_positions[fixed_indices]

        # Solve using scipy
        new_positions_np = np.zeros((n, 3))

        if self._scipy_solver is not None:
            for dim in range(3):
                new_positions_np[:, dim] = self._scipy_solver(b_np[:, dim])
        else:
            for dim in range(3):
                new_positions_np[:, dim] = scipy.sparse.linalg.spsolve(
                    self._L_scipy, b_np[:, dim]
                )

        return new_positions_np

    def solve(self, positions, rest_positions, neighbors, weights, rest_edges,
              fixed_mask, fixed_targets, max_iterations=20, tolerance=1e-4,
              target_edges=None, verbose=False):
        """Run full ARAP iteration using Taichi."""
        positions = positions.copy()
        free_indices = np.where(~fixed_mask)[0]

        # Set fixed positions
        if fixed_targets is not None:
            fixed_indices = np.where(fixed_mask)[0]
            positions[fixed_indices] = fixed_targets

        for iteration in range(max_iterations):
            old_positions = positions.copy()

            # Local step
            rotations = self.local_step(positions, rest_positions, neighbors, weights, rest_edges, target_edges)

            # Debug output
            if verbose and iteration == 0:
                rot_errors = [np.linalg.norm(r - np.eye(3)) for r in rotations]
                print(f"  DEBUG: max rotation deviation from I: {max(rot_errors):.6f}")

            # Global step
            positions = self.global_step(rest_positions, neighbors, weights, rest_edges,
                                         rotations, fixed_mask, fixed_targets, target_edges)

            # Debug output
            if verbose and iteration == 0:
                pos_diff = np.linalg.norm(positions - rest_positions, axis=1)
                print(f"  DEBUG: max position diff from rest after iter 0: {np.max(pos_diff):.6f}")

            # Check for NaN
            if not np.isfinite(positions).all():
                if verbose:
                    print(f"  Taichi: Non-finite at iteration {iteration}, reverting")
                positions = old_positions
                break

            # Re-enforce fixed
            if fixed_targets is not None:
                fixed_indices = np.where(fixed_mask)[0]
                positions[fixed_indices] = fixed_targets

            # Check convergence
            if len(free_indices) > 0:
                disp = np.linalg.norm(positions[free_indices] - old_positions[free_indices], axis=1)
                max_disp = np.max(disp)
            else:
                max_disp = 0.0

            if verbose and (iteration + 1) % 5 == 0:
                print(f"  Iter {iteration+1}: max_disp={max_disp:.2e}")

            if max_disp < tolerance:
                if verbose:
                    print(f"  Converged at iteration {iteration+1}, max_disp={max_disp:.2e}")
                return positions, iteration + 1, max_disp

        if verbose:
            print(f"  Max iterations reached, max_disp={max_disp:.2e}")
        return positions, max_iterations, max_disp


def get_backend(name='cpu', device=None):
    """Factory function to get ARAP backend by name."""
    if name == 'cpu':
        return ARAPBackendCPU()
    elif name == 'gpu' or name == 'pytorch':
        return ARAPBackendGPU(device=device)
    elif name == 'taichi':
        return ARAPBackendTaichi()
    else:
        raise ValueError(f"Unknown backend: {name}. Available: cpu, gpu, taichi")


def check_gpu_available():
    """Check if GPU acceleration is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def check_taichi_available():
    """Check if Taichi is available."""
    try:
        import taichi as ti
        return True
    except ImportError:
        return False
