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
        self._fields_allocated = False
        self._data_stale = True  # True = CSR data needs re-upload to GPU

        # Scipy solver cache
        self._scipy_solver = None
        self._L_scipy = None

    def _allocate_fields(self, n_verts, n_csr_edges):
        """Allocate Taichi fields for computation."""
        ti = self.ti

        if self._fields_allocated and self.n_verts == n_verts:
            return

        self.n_verts = n_verts

        # Vertex data
        self.positions = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        self.rest_positions = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)
        self.rhs = ti.Vector.field(3, dtype=ti.f64, shape=n_verts)

        # CSR neighbor data (used by fused local kernel + RHS kernel)
        self.neighbor_offsets = ti.field(dtype=ti.i32, shape=n_verts + 1)
        self.neighbor_indices = ti.field(dtype=ti.i32, shape=n_csr_edges)
        self.neighbor_weights = ti.field(dtype=ti.f64, shape=n_csr_edges)
        self.neighbor_rest = ti.Vector.field(3, dtype=ti.f64, shape=n_csr_edges)

        self._fields_allocated = True

    def build_system(self, num_vertices, neighbors, weights, fixed_mask, regularization=1e-6):
        """Build sparse Laplacian matrix."""
        import scipy.sparse

        self._scipy_solver = None  # Clear cache
        self._data_stale = True  # Force re-upload of CSR data to GPU

        n = num_vertices
        self.num_vertices = n
        self.fixed_mask = np.array(fixed_mask)
        self.neighbors = neighbors
        self.weights = weights
        self.regularization = regularization

        # Build CSR structure for vertex-centric gather (used by fused local + RHS kernels)
        neighbor_offsets = np.zeros(n + 1, dtype=np.int32)
        for i, neighs in enumerate(neighbors):
            neighbor_offsets[i + 1] = neighbor_offsets[i] + len(neighs)
        n_csr_edges = neighbor_offsets[-1]
        neighbor_indices = np.zeros(n_csr_edges, dtype=np.int32)
        neighbor_weights_arr = np.zeros(n_csr_edges, dtype=np.float64)
        for i, neighs in enumerate(neighbors):
            start = neighbor_offsets[i]
            for k, j in enumerate(neighs):
                neighbor_indices[start + k] = j
                neighbor_weights_arr[start + k] = weights.get((i, j), weights.get((j, i), 1.0))
        self.neighbor_offsets_np = neighbor_offsets
        self.neighbor_indices_np = neighbor_indices
        self.neighbor_weights_csr_np = neighbor_weights_arr
        self.n_csr_edges = n_csr_edges

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

        # Extract free-DOF subblock and factorize with splu for batched solve
        import scipy.sparse.linalg
        free_idx = np.where(~np.array(fixed_mask))[0]
        fixed_idx = np.where(np.array(fixed_mask))[0]
        self._free_idx = free_idx
        self._fixed_idx = fixed_idx
        self._splu = None
        self._scipy_solver = None
        try:
            L_ff = self._L_scipy[np.ix_(free_idx, free_idx)].tocsc()
            self._L_fc = self._L_scipy[np.ix_(free_idx, fixed_idx)].tocsc()
            self._splu = scipy.sparse.linalg.splu(L_ff)
        except Exception:
            # Fallback: factorize full system
            try:
                self._scipy_solver = scipy.sparse.linalg.factorized(self._L_scipy)
            except Exception:
                pass

        # Build L CSR arrays for GPU CG solver
        L_csr = self._L_scipy.tocsr()
        self._L_csr_offsets_np = L_csr.indptr.astype(np.int32)
        self._L_csr_indices_np = L_csr.indices.astype(np.int32)
        self._L_csr_values_np = L_csr.data.astype(np.float64)
        self._L_nnz = len(self._L_csr_values_np)

        # Jacobi preconditioner: inverse diagonal
        diag = np.array(L_csr.diagonal(), dtype=np.float64)
        self._L_diag_inv_np = np.where(np.abs(diag) > 1e-15, 1.0 / diag, 1.0)

        self._cg_allocated = False

        return True

    def _build_fused_local_kernel(self):
        """Create fused Taichi kernel: CSR gather covariance + ti.svd rotation in one pass."""
        ti = self.ti

        @ti.kernel
        def fused_local_step(
            positions: ti.template(),
            neighbor_offsets: ti.template(),
            neighbor_indices: ti.template(),
            neighbor_weights: ti.template(),
            neighbor_rest: ti.template(),
            rotations: ti.template(),
            n_verts: ti.i32
        ):
            for i in range(n_verts):
                # Gather: build S matrix locally (registers, no global memory atomics)
                S = ti.Matrix.zero(ti.f64, 3, 3)
                start = neighbor_offsets[i]
                end = neighbor_offsets[i + 1]
                for idx in range(start, end):
                    j = neighbor_indices[idx]
                    w = neighbor_weights[idx]
                    e_curr = positions[j] - positions[i]
                    e_rest = neighbor_rest[idx]
                    for a in ti.static(range(3)):
                        for b in ti.static(range(3)):
                            S[a, b] += w * e_curr[a] * e_rest[b]

                # SVD + rotation (same thread, no intermediate global write)
                S_norm = S.norm()
                if S_norm < 1e-10:
                    rotations[i] = ti.Matrix.identity(ti.f64, 3)
                else:
                    U, sigma, V = ti.svd(S, ti.f64)
                    R = U @ V.transpose()
                    if R.determinant() < 0:
                        for k in ti.static(range(3)):
                            U[k, 2] = -U[k, 2]
                        R = U @ V.transpose()
                    rotations[i] = R

        return fused_local_step

    def _build_rhs_kernel(self):
        """Create Taichi kernel for RHS computation."""
        ti = self.ti

        @ti.kernel
        def compute_rhs(
            rest_positions: ti.template(),
            neighbor_offsets: ti.template(),
            neighbor_indices: ti.template(),
            neighbor_weights: ti.template(),
            neighbor_rest: ti.template(),
            rotations: ti.template(),
            rhs: ti.template(),
            regularization: ti.f64,
            n_verts: ti.i32
        ):
            for i in range(n_verts):
                R_i = rotations[i]
                rhs_val = regularization * rest_positions[i]
                start = neighbor_offsets[i]
                end = neighbor_offsets[i + 1]
                for idx in range(start, end):
                    j = neighbor_indices[idx]
                    w = neighbor_weights[idx]
                    # Rest edge negated: p_i - p_j (neighbor_rest stores p_j - p_i)
                    e_rest_neg = -neighbor_rest[idx]
                    R_avg = ti.cast(0.5, ti.f64) * (R_i + rotations[j])
                    rhs_val += w * (R_avg @ e_rest_neg)
                rhs[i] = rhs_val

        return compute_rhs

    def local_step(self, positions, rest_positions, neighbors, weights, rest_edges, target_edges=None):
        """Compute optimal rotations using fused CSR gather + ti.svd (fully on GPU)."""
        ti = self.ti
        n = len(positions)

        # Allocate fields if size changed
        if not self._fields_allocated or self.n_verts != n:
            # Build CSR rest edge vectors
            neighbor_rest_list = []
            for i, neighs in enumerate(neighbors):
                for j in neighs:
                    if j in rest_edges[i]:
                        neighbor_rest_list.append(rest_edges[i][j])
                    else:
                        neighbor_rest_list.append(rest_positions[j] - rest_positions[i])
            self.neighbor_rest_np = np.array(neighbor_rest_list, dtype=np.float64)

            self._allocate_fields(n, self.n_csr_edges)
            self._data_stale = True  # New fields need data upload

        # Re-upload CSR data if stale (build_system was called with new data)
        if self._data_stale:
            # Rebuild CSR rest edge vectors if not already done above
            if not hasattr(self, 'neighbor_rest_np') or len(self.neighbor_rest_np) != self.n_csr_edges:
                neighbor_rest_list = []
                for i, neighs in enumerate(neighbors):
                    for j in neighs:
                        if j in rest_edges[i]:
                            neighbor_rest_list.append(rest_edges[i][j])
                        else:
                            neighbor_rest_list.append(rest_positions[j] - rest_positions[i])
                self.neighbor_rest_np = np.array(neighbor_rest_list, dtype=np.float64)

            self.neighbor_offsets.from_numpy(self.neighbor_offsets_np)
            self.neighbor_indices.from_numpy(self.neighbor_indices_np)
            self.neighbor_weights.from_numpy(self.neighbor_weights_csr_np)
            self.neighbor_rest.from_numpy(self.neighbor_rest_np)
            self._data_stale = False

        # Allocate rotation field if needed
        if not hasattr(self, 'rotations_field_local') or self.rotations_field_local.shape[0] != n:
            self.rotations_field_local = ti.Matrix.field(3, 3, dtype=ti.f64, shape=n)

        # Copy positions to Taichi fields
        self.positions.from_numpy(positions if positions.dtype == np.float64 else positions.astype(np.float64))

        # Build fused kernel if not cached
        if not hasattr(self, '_fused_local_kernel'):
            self._fused_local_kernel = self._build_fused_local_kernel()

        # Fused covariance + SVD on GPU (one kernel, no atomics)
        self._fused_local_kernel(
            self.positions,
            self.neighbor_offsets, self.neighbor_indices,
            self.neighbor_weights, self.neighbor_rest,
            self.rotations_field_local, n
        )

        return None

    def global_step(self, rest_positions, neighbors, weights, rest_edges,
                    fixed_mask, fixed_targets, target_edges=None):
        """Solve linear system using Taichi for RHS, scipy for solve."""
        import scipy.sparse.linalg
        ti = self.ti

        n = len(rest_positions)
        regularization = self.regularization

        # Build RHS kernel if not cached
        if not hasattr(self, '_rhs_kernel'):
            self._rhs_kernel = self._build_rhs_kernel()

        # Compute RHS on GPU (CSR gather, no atomics)
        self._rhs_kernel(
            self.rest_positions,
            self.neighbor_offsets, self.neighbor_indices,
            self.neighbor_weights, self.neighbor_rest,
            self.rotations_field_local, self.rhs,
            regularization, n
        )

        # Get RHS back to numpy
        b_np = self.rhs.to_numpy()

        # Set fixed vertex RHS
        fixed_indices = np.where(fixed_mask)[0]
        if fixed_targets is not None and len(fixed_targets) > 0:
            b_np[fixed_indices] = fixed_targets
        else:
            b_np[fixed_indices] = rest_positions[fixed_indices]

        # Solve using reduced system (splu batched) or full system fallback
        new_positions_np = np.zeros((n, 3))

        if getattr(self, '_splu', None) is not None:
            fixed_pos = b_np[self._fixed_idx]
            b_free = b_np[self._free_idx] - self._L_fc.dot(fixed_pos)
            new_positions_np[self._free_idx] = self._splu.solve(b_free)
            new_positions_np[self._fixed_idx] = fixed_pos
        elif self._scipy_solver is not None:
            for dim in range(3):
                new_positions_np[:, dim] = self._scipy_solver(b_np[:, dim])
        else:
            for dim in range(3):
                new_positions_np[:, dim] = scipy.sparse.linalg.spsolve(
                    self._L_scipy, b_np[:, dim]
                )

        return new_positions_np

    # ── GPU CG solver ──────────────────────────────────────────────────

    def _allocate_cg_fields(self):
        """Allocate Taichi fields for GPU PCG solver."""
        ti = self.ti
        n = self.num_vertices
        nnz = self._L_nnz

        # L matrix CSR on GPU
        self.L_cg_offsets = ti.field(dtype=ti.i32, shape=n + 1)
        self.L_cg_indices = ti.field(dtype=ti.i32, shape=nnz)
        self.L_cg_values = ti.field(dtype=ti.f64, shape=nnz)
        self.cg_diag_inv = ti.field(dtype=ti.f64, shape=n)

        # CG work vectors (3-component to handle x/y/z simultaneously)
        self.cg_x = ti.Vector.field(3, dtype=ti.f64, shape=n)
        self.cg_r = ti.Vector.field(3, dtype=ti.f64, shape=n)
        self.cg_z = ti.Vector.field(3, dtype=ti.f64, shape=n)
        self.cg_p = ti.Vector.field(3, dtype=ti.f64, shape=n)
        self.cg_Ap = ti.Vector.field(3, dtype=ti.f64, shape=n)

        # Old positions for ARAP convergence check
        self._old_positions = ti.Vector.field(3, dtype=ti.f64, shape=n)

        # Reduction scalars
        self.cg_rz = ti.field(dtype=ti.f64, shape=())
        self.cg_pAp = ti.field(dtype=ti.f64, shape=())
        self.cg_rz_new = ti.field(dtype=ti.f64, shape=())
        self.cg_max_disp = ti.field(dtype=ti.f64, shape=())

        # Upload L data to GPU
        self.L_cg_offsets.from_numpy(self._L_csr_offsets_np)
        self.L_cg_indices.from_numpy(self._L_csr_indices_np)
        self.L_cg_values.from_numpy(self._L_csr_values_np)
        self.cg_diag_inv.from_numpy(self._L_diag_inv_np)

        self._cg_allocated = True

    def _build_cg_kernels(self):
        """Build all Taichi kernels for PCG solver."""
        ti = self.ti

        @ti.kernel
        def cg_spmv(offsets: ti.template(), indices: ti.template(),
                     values: ti.template(), x: ti.template(),
                     out: ti.template(), n: ti.i32):
            for i in range(n):
                val = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
                for idx in range(offsets[i], offsets[i + 1]):
                    j = indices[idx]
                    val += values[idx] * x[j]
                out[i] = val

        @ti.kernel
        def cg_dot(x: ti.template(), y: ti.template(),
                    out: ti.template(), n: ti.i32):
            out[None] = 0.0
            for i in range(n):
                ti.atomic_add(out[None], x[i].dot(y[i]))

        @ti.kernel
        def cg_residual(b: ti.template(), Ax: ti.template(),
                         r: ti.template(), n: ti.i32):
            for i in range(n):
                r[i] = b[i] - Ax[i]

        @ti.kernel
        def cg_precond(diag_inv: ti.template(), r: ti.template(),
                        z: ti.template(), n: ti.i32):
            for i in range(n):
                z[i] = diag_inv[i] * r[i]

        @ti.kernel
        def cg_copy(src: ti.template(), dst: ti.template(), n: ti.i32):
            for i in range(n):
                dst[i] = src[i]

        @ti.kernel
        def cg_step1(cg_x: ti.template(), cg_r: ti.template(),
                      cg_p: ti.template(), cg_Ap: ti.template(),
                      rz: ti.template(), pAp: ti.template(), n: ti.i32):
            """alpha = rz/pAp; x += alpha*p; r -= alpha*Ap"""
            alpha = rz[None] / pAp[None]
            for i in range(n):
                cg_x[i] += alpha * cg_p[i]
                cg_r[i] -= alpha * cg_Ap[i]

        @ti.kernel
        def cg_step2(cg_p: ti.template(), cg_z: ti.template(),
                      rz: ti.template(), rz_new: ti.template(), n: ti.i32):
            """beta = rz_new/rz; p = z + beta*p; rz = rz_new"""
            beta = rz_new[None] / rz[None]
            rz[None] = rz_new[None]
            for i in range(n):
                cg_p[i] = cg_z[i] + beta * cg_p[i]

        @ti.kernel
        def cg_set_fixed(field: ti.template(), indices: ti.template(),
                          targets: ti.template(), n_fixed: ti.i32):
            for k in range(n_fixed):
                i = indices[k]
                field[i] = targets[k]

        @ti.kernel
        def cg_convergence(new_pos: ti.template(), old_pos: ti.template(),
                            free_mask: ti.template(),
                            out: ti.template(), n: ti.i32):
            out[None] = 0.0
            for i in range(n):
                if free_mask[i] == 0:
                    diff = (new_pos[i] - old_pos[i]).norm()
                    ti.atomic_max(out[None], diff)

        self._cg_spmv_k = cg_spmv
        self._cg_dot_k = cg_dot
        self._cg_residual_k = cg_residual
        self._cg_precond_k = cg_precond
        self._cg_copy_k = cg_copy
        self._cg_step1_k = cg_step1
        self._cg_step2_k = cg_step2
        self._cg_set_fixed_k = cg_set_fixed
        self._cg_convergence_k = cg_convergence
        self._cg_kernels_built = True

    def _gpu_pcg_solve(self, n, max_iter=200, cg_tol=1e-10):
        """Run Jacobi-preconditioned CG on GPU: L @ x = rhs.

        Initial guess read from self.positions, RHS from self.rhs.
        Solution written to self.positions.
        Returns number of CG iterations.
        """
        # x = initial guess (copy positions)
        self._cg_copy_k(self.positions, self.cg_x, n)

        # Ap = L @ x0
        self._cg_spmv_k(self.L_cg_offsets, self.L_cg_indices, self.L_cg_values,
                         self.cg_x, self.cg_Ap, n)

        # r = b - Ax0
        self._cg_residual_k(self.rhs, self.cg_Ap, self.cg_r, n)

        # z = M^{-1} r
        self._cg_precond_k(self.cg_diag_inv, self.cg_r, self.cg_z, n)

        # p = z
        self._cg_copy_k(self.cg_z, self.cg_p, n)

        # rz = r . z
        self._cg_dot_k(self.cg_r, self.cg_z, self.cg_rz, n)

        # Read initial rz for relative convergence check
        rz0 = self.cg_rz[None]
        if rz0 < 1e-30:
            self._cg_copy_k(self.cg_x, self.positions, n)
            return 0

        cg_tol_sq = cg_tol * cg_tol
        iters = 0

        for k in range(max_iter):
            # Ap = L @ p
            self._cg_spmv_k(self.L_cg_offsets, self.L_cg_indices, self.L_cg_values,
                             self.cg_p, self.cg_Ap, n)

            # pAp = p . Ap
            self._cg_dot_k(self.cg_p, self.cg_Ap, self.cg_pAp, n)

            # alpha = rz/pAp, x += alpha*p, r -= alpha*Ap
            self._cg_step1_k(self.cg_x, self.cg_r, self.cg_p, self.cg_Ap,
                              self.cg_rz, self.cg_pAp, n)

            # z = M^{-1} r
            self._cg_precond_k(self.cg_diag_inv, self.cg_r, self.cg_z, n)

            # rz_new = r . z
            self._cg_dot_k(self.cg_r, self.cg_z, self.cg_rz_new, n)

            # beta = rz_new/rz, p = z + beta*p, rz = rz_new
            self._cg_step2_k(self.cg_p, self.cg_z, self.cg_rz, self.cg_rz_new, n)

            iters = k + 1

            # Check convergence periodically (one scalar download)
            if (k + 1) % 20 == 0 or k == max_iter - 1:
                rz_val = self.cg_rz[None]
                if rz_val < rz0 * cg_tol_sq:
                    break

        # Copy solution to positions
        self._cg_copy_k(self.cg_x, self.positions, n)
        return iters

    # ── Solve methods ────────────────────────────────────────────────

    def solve(self, positions, rest_positions, neighbors, weights, rest_edges,
              fixed_mask, fixed_targets, max_iterations=20, tolerance=1e-4,
              target_edges=None, verbose=False):
        """Run full ARAP iteration using Taichi.
        Uses GPU PCG for the linear solve when CG fields are allocated,
        keeping positions on GPU between iterations to eliminate transfers.
        Falls back to scipy otherwise.
        """
        n = len(positions)
        positions = positions.copy()
        free_indices = np.where(~fixed_mask)[0]
        fixed_indices = np.where(fixed_mask)[0]

        # Set fixed positions
        if fixed_targets is not None:
            positions[fixed_indices] = fixed_targets

        rest_pos_f64 = rest_positions if rest_positions.dtype == np.float64 else rest_positions.astype(np.float64)

        # GPU CG path disabled: iterative CG is slower than scipy's direct
        # LU forward/back-sub for this problem size (~11k verts).
        use_gpu_cg = False
        if use_gpu_cg and not getattr(self, '_cg_kernels_built', False):
            self._build_cg_kernels()

        if use_gpu_cg:
            return self._solve_gpu_cg(
                positions, rest_pos_f64, n, free_indices, fixed_indices,
                fixed_mask, fixed_targets, neighbors, weights, rest_edges,
                target_edges, max_iterations, tolerance, verbose)

        # Fallback: original scipy path
        rest_uploaded = False
        for iteration in range(max_iterations):
            self.local_step(positions, rest_positions, neighbors, weights, rest_edges, target_edges)
            if not rest_uploaded:
                self.rest_positions.from_numpy(rest_pos_f64)
                rest_uploaded = True
            new_positions = self.global_step(rest_positions, neighbors, weights, rest_edges,
                                             fixed_mask, fixed_targets, target_edges)
            if not np.isfinite(new_positions).all():
                if verbose:
                    print(f"  Taichi: Non-finite at iteration {iteration}, reverting")
                break
            if len(free_indices) > 0:
                disp = np.linalg.norm(new_positions[free_indices] - positions[free_indices], axis=1)
                max_disp = np.max(disp)
            else:
                max_disp = 0.0
            positions = new_positions
            if fixed_targets is not None:
                positions[fixed_indices] = fixed_targets
            if verbose and (iteration + 1) % 5 == 0:
                print(f"  Iter {iteration+1}: max_disp={max_disp:.2e}")
            if max_disp < tolerance:
                if verbose:
                    print(f"  Converged at iteration {iteration+1}, max_disp={max_disp:.2e}")
                return positions, iteration + 1, max_disp
        if verbose:
            print(f"  Max iterations reached, max_disp={max_disp:.2e}")
        return positions, max_iterations, max_disp

    def _solve_gpu_cg(self, positions, rest_pos_f64, n, free_indices, fixed_indices,
                       fixed_mask, fixed_targets, neighbors, weights, rest_edges,
                       target_edges, max_iterations, tolerance, verbose):
        """ARAP solve with GPU PCG: positions stay on GPU between iterations."""
        ti = self.ti

        # Ensure fields are allocated before any GPU uploads
        if self._data_stale:
            if not hasattr(self, 'neighbor_rest_np') or len(self.neighbor_rest_np) != self.n_csr_edges:
                neighbor_rest_list = []
                for i, neighs in enumerate(neighbors):
                    for j in neighs:
                        if j in rest_edges[i]:
                            neighbor_rest_list.append(rest_edges[i][j])
                        else:
                            neighbor_rest_list.append(rest_pos_f64[j] - rest_pos_f64[i])
                self.neighbor_rest_np = np.array(neighbor_rest_list, dtype=np.float64)
            if not self._fields_allocated or self.n_verts != n:
                self._allocate_fields(n, self.n_csr_edges)
            self.neighbor_offsets.from_numpy(self.neighbor_offsets_np)
            self.neighbor_indices.from_numpy(self.neighbor_indices_np)
            self.neighbor_weights.from_numpy(self.neighbor_weights_csr_np)
            self.neighbor_rest.from_numpy(self.neighbor_rest_np)
            self._data_stale = False
        elif not self._fields_allocated or self.n_verts != n:
            self._allocate_fields(n, self.n_csr_edges)

        # Allocate CG fields if needed
        if not self._cg_allocated:
            self._allocate_cg_fields()

        # Ensure kernels are built
        if not hasattr(self, '_fused_local_kernel'):
            self._fused_local_kernel = self._build_fused_local_kernel()
        if not hasattr(self, '_rhs_kernel'):
            self._rhs_kernel = self._build_rhs_kernel()
        if not hasattr(self, 'rotations_field_local') or self.rotations_field_local.shape[0] != n:
            self.rotations_field_local = ti.Matrix.field(3, 3, dtype=ti.f64, shape=n)

        # Upload all data to GPU once
        pos_f64 = positions if positions.dtype == np.float64 else positions.astype(np.float64)
        self.positions.from_numpy(pos_f64)
        self.rest_positions.from_numpy(rest_pos_f64)

        # Upload fixed target info to GPU
        n_fixed = len(fixed_indices)
        if n_fixed > 0:
            if not hasattr(self, '_fixed_idx_field') or self._fixed_idx_field.shape[0] != n_fixed:
                self._fixed_idx_field = ti.field(dtype=ti.i32, shape=n_fixed)
                self._fixed_tgt_field = ti.Vector.field(3, dtype=ti.f64, shape=n_fixed)
            self._fixed_idx_field.from_numpy(fixed_indices.astype(np.int32))
            if fixed_targets is not None:
                self._fixed_tgt_field.from_numpy(fixed_targets.astype(np.float64))
            else:
                self._fixed_tgt_field.from_numpy(rest_pos_f64[fixed_indices])

        # Upload fixed mask as int for convergence kernel
        if not hasattr(self, '_fixed_mask_field') or self._fixed_mask_field.shape[0] != n:
            self._fixed_mask_field = ti.field(dtype=ti.i32, shape=n)
        self._fixed_mask_field.from_numpy(fixed_mask.astype(np.int32))

        # Keep a numpy fallback in case of NaN
        fallback_positions = positions.copy()

        for iteration in range(max_iterations):
            # Save old positions for convergence check (GPU copy)
            self._cg_copy_k(self.positions, self._old_positions, n)

            # Local step: reads self.positions, writes self.rotations_field_local
            self._fused_local_kernel(
                self.positions,
                self.neighbor_offsets, self.neighbor_indices,
                self.neighbor_weights, self.neighbor_rest,
                self.rotations_field_local, n
            )

            # RHS kernel: reads rest_positions + rotations → writes self.rhs
            self._rhs_kernel(
                self.rest_positions,
                self.neighbor_offsets, self.neighbor_indices,
                self.neighbor_weights, self.neighbor_rest,
                self.rotations_field_local, self.rhs,
                self.regularization, n
            )

            # Set fixed vertex RHS on GPU
            if n_fixed > 0:
                self._cg_set_fixed_k(self.rhs, self._fixed_idx_field,
                                      self._fixed_tgt_field, n_fixed)

            # Solve L @ x = rhs using PCG on GPU
            cg_iters = self._gpu_pcg_solve(n)

            # Re-enforce fixed positions on GPU
            if n_fixed > 0:
                self._cg_set_fixed_k(self.positions, self._fixed_idx_field,
                                      self._fixed_tgt_field, n_fixed)

            # Convergence check on GPU (one scalar download)
            self._cg_convergence_k(self.positions, self._old_positions,
                                    self._fixed_mask_field, self.cg_max_disp, n)
            max_disp = self.cg_max_disp[None]

            if not np.isfinite(max_disp) or max_disp > 1e6:
                if verbose:
                    print(f"  GPU CG: Non-finite at iteration {iteration}, reverting")
                return fallback_positions, iteration, float('inf')

            if verbose and (iteration + 1) % 5 == 0:
                print(f"  Iter {iteration+1}: max_disp={max_disp:.2e} (CG:{cg_iters})")

            if max_disp < tolerance:
                if verbose:
                    print(f"  Converged at iteration {iteration+1}, max_disp={max_disp:.2e}")
                return self.positions.to_numpy(), iteration + 1, max_disp

            # Periodic numpy backup for NaN fallback
            if iteration % 20 == 0:
                fallback_positions = self.positions.to_numpy().copy()

        if verbose:
            print(f"  Max iterations reached, max_disp={max_disp:.2e}")
        return self.positions.to_numpy(), max_iterations, max_disp


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
