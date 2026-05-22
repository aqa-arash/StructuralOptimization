import numpy as np
import matplotlib.pyplot as plt
import feature_definition as sp 
import os



"""
Optimization layer (SciPy/IPOPT-friendly) for feature-based topology optimization.

Defines:
  - Primitive geometric constraints (segment min/max length) with value/Jacobian/Hessian.
  - Constraint combiner for stacking multiple constraints.
  - FeatureOptimizationProblemConstraints: objective, gradient, and Hessian assembly
    using the density/derivative fields from `pill` (imported as `sp`).
"""


class SegmentLengthConstraint:
    """
    Inequality constraint enforcing a minimum segment length ‖P-Q‖ ≥ l_min.

    Purpose
    -------
    Used to prevent degenerate capsules: ensures endpoints remain separated by
    at least `l_min`. Provides value, analytic Jacobian, and Hessian.
    """
    def __init__(self, l_min):
        """
        Parameters
        ----------
        l_min : float
            Minimum allowable segment length.
        """
        self.l_min = l_min


    def constraint(self, s):
        """
        Constraint value c(s) = ‖P-Q‖ - l_min ≥ 0.

        Parameters
        ----------
        s : ndarray, shape (>=4,)
            Design vector with first 4 entries [Px, Py, Qx, Qy].

        Returns
        -------
        c : ndarray, shape (1,)
            Scalar inequality value.
        """
        P = s[0:2]
        Q = s[2:4]
        value = np.linalg.norm(P - Q) - self.l_min
        return np.array([value], dtype=np.float64)

    def jacobian(self, s):
        """
        Jacobian ∂c/∂s.

        Parameters
        ----------
        s : ndarray, shape (>=4,)

        Returns
        -------
        grad : ndarray, shape (len(s),)
            Gradient w.r.t. design vector; only entries 0:4 are nonzero.
        """
        P = s[0:2]
        Q = s[2:4]
        v = P - Q
        l = np.linalg.norm(v)
        grad = np.zeros_like(s)
        if l == 0:
            return grad
        grad[0:2] = v / l
        grad[2:4] = -v / l
        return grad

    def hessian(self, s, lagrange_multiplier):
        """
        Hessian of the Lagrangian contribution for this constraint.

        Parameters
        ----------
        s : ndarray, shape (>=4,)
        lagrange_multiplier : float
            Constraint multiplier λ.

        Returns
        -------
        H : ndarray, shape (len(s), len(s))
            λ * ∂²c/∂s² with the standard 2×2 block structure on P/Q entries.
        """
        P = s[0:2]
        Q = s[2:4]
        v = P - Q
        l = np.linalg.norm(v)

        if l == 0:
            return np.zeros((len(s), len(s)))
        I = np.eye(2)
        outer = np.outer(v, v)
        block = (I / l) - (outer / l**3)

        H = np.zeros((len(s), len(s)))
        H[0:2, 0:2] = block
        H[2:4, 2:4] = block
        H[0:2, 2:4] = -block
        H[2:4, 0:2] = -block

        return lagrange_multiplier * H
    

class SegmentMaxLengthConstraint:
    """
    Inequality constraint enforcing a maximum segment length ‖P-Q‖ ≤ l_max.

    Purpose
    -------
    Prevents overly long features by bounding endpoint separation.
    """
    def __init__(self, l_max):
        self.l_max = l_max


    def constraint(self, s):
        """
        Constraint value c(s) = l_max - ‖P-Q‖ ≥ 0.

        Parameters
        ----------
        s : ndarray, shape (>=4,)

        Returns
        -------
        c : ndarray, shape (1,)
        """
        P = s[0:2]
        Q = s[2:4]
        value = self.l_max -np.linalg.norm(P - Q) 
        return np.array([value], dtype=np.float64)

    def jacobian(self, s):
        """
        Jacobian ∂c/∂s for the max-length constraint.

        Parameters
        ----------
        s : ndarray, shape (>=4,)

        Returns
        -------
        grad : ndarray, shape (len(s),)
            Negative of the min-length gradient.
        """
        P = s[0:2]
        Q = s[2:4]
        v = P - Q
        l = np.linalg.norm(v)
        grad = np.zeros_like(s)
        if l == 0:
            return grad
        grad[0:2] = v / l
        grad[2:4] = -v / l
        return -grad

    def hessian(self, s, lagrange_multiplier):
        """
        Hessian of the Lagrangian contribution (max-length).

        Parameters
        ----------
        s : ndarray
        lagrange_multiplier : float

        Returns
        -------
        H : ndarray, shape (len(s), len(s))
            λ * (−∂²‖P−Q‖/∂s²) with appropriate P/Q block signs.
        """
        P = s[0:2]
        Q = s[2:4]
        v = P - Q
        l = np.linalg.norm(v)

        if l == 0:
            return np.zeros((len(s), len(s)))

        I = np.eye(2)
        outer = np.outer(v, v)
        block = (I / l) - (outer / l**3)

        H = np.zeros((len(s), len(s)))
        H[0:2, 0:2] = block
        H[2:4, 2:4] = block
        H[0:2, 2:4] = -block
        H[2:4, 0:2] = -block

        return lagrange_multiplier * -H


class CombinedConstraints:
    """
    Constraint stacker for multiple inequality constraints.

    Purpose
    -------
    Concatenates individual constraint values, vertically stacks Jacobians,
    and sums Hessians weighted by their respective multipliers.
    """
    def __init__(self, constraints_list):
        """
        Parameters
        ----------
        constraints_list : list
            List of constraint objects exposing .constraint / .jacobian / .hessian.
        """
        self.constraints_list = constraints_list

    def constraint(self, s):
        """
        Concatenate all constraint values.

        Returns
        -------
        c : ndarray, shape (m,)
            m = Σ_i m_i (number of scalar constraints across all entries).
        """
        return np.concatenate([c.constraint(s) for c in self.constraints_list])

    def jacobian(self, s):
        """
        Stack all constraint Jacobians row-wise.

        Returns
        -------
        J : ndarray, shape (m, n)
            n = len(s). Matches the order of `constraint(s)`.
        """
        return np.vstack([c.jacobian(s) for c in self.constraints_list])

    def hessian(self, s, lagrange):
        """
        Sum of per-constraint Hessians weighted by their multipliers.

        Parameters
        ----------
        s : ndarray, shape (n,)
        lagrange : ndarray, shape (m,)
            Concatenated Lagrange multipliers aligned with `constraint(s)`.

        Returns
        -------
        H : ndarray, shape (n, n)
        """
        H_total = np.zeros((len(s), len(s)))
        offset = 0
        for c in self.constraints_list:
            n_c = len(c.constraint(s))
            λ_c = lagrange[offset:offset + n_c]
            H_total += c.hessian(s, λ_c)
            offset += n_c
        return H_total


class FeatureOptimizationProblemConstraints:
    """
    Objective and constraints wrapper for feature-based optimization.

    Purpose
    -------
    Provides:
      - objective(s): scalar cost based on target field S_Star and current density S(s),
      - gradient(s): analytic ∂objective/∂s using sp.ableitung,
      - hessian_matrix(s): analytic ∂²objective/∂s² using sp.hessian and Gauss-Newton terms,
      - IPOPT-compatible sparsity accessors and constraint interfaces.

    Notes
    -----
    Supports three objective modes:
      - default: least-squares fit to S_Star,
      - reward_only: maximize correlation with S_Star (negative dot),
      - reward_only2: saturating reward over Σρ with piecewise gradient.
    """
    def __init__(self, num_vars, S_Star, constraint_obj=None, plot = True, frame_dir=None, reward_only=False, reward_only2=False ):
        """
        Parameters
        ----------
        num_vars : int
            Dimension of s (5 × number_of_features).
        S_Star : ndarray, shape (ny*nx,) or (ny, nx), Fortran order implied
            Target density field (will be reshaped to (ny, nx) with order='F').
        constraint_obj : object, optional
            CombinedConstraints or single constraint object; may be None.
        plot : bool | list[str], default True
            If True: plot 'objective'. If list, allowed entries: ['objective','derivative'].
        frame_dir : str or None
            Directory to store frames if plotting is enabled.
        reward_only : bool, default False
            Use negative correlation objective.
        reward_only2 : bool, default False
            Use saturating reward objective.
        """
        
        self.num_vars = num_vars
        self.s = None
        self.S = None
        self.grad = None
        self.objective_value = None
        self.nx, self.ny = sp.glob.n
        self.n_points = self.nx * self.ny
        self.s_history = []  
        self.objective_history = []
        self.gradient_history = [] 
        self.hessian_history = []
        self.last_logged_s = None
        self.constraint_obj = constraint_obj
        if isinstance(plot, bool):
            self.plot = ["objective"] if plot else []
        elif isinstance(plot, list):
            self.plot = plot
        else:
            raise ValueError("plot must be a bool or list of strings like ['objective', 'derivative']")

        self.frame_dir = frame_dir
        self.reward_only = reward_only
        self.reward_only2 = reward_only2
        self.shape_2d = (self.ny, self.nx)
        self.S_Star = S_Star.reshape(self.shape_2d, order='F')



    def objective(self, s):
        """
        Objective value at s.

        Modes
        -----
        - default: sum((S_Star - S(s))^2)
        - reward_only: -⟨S_Star, S(s)⟩
        - reward_only2: -Σ (ρ / max(ρ, 1))

        Side Effects
        ------------
        - Updates histories (objective, s) and optionally writes plot frames.
        - Maintains last_logged_s to avoid excessive frame generation.

        Returns
        -------
        J : float
        """
        self.s = s.copy()
        S_current = sp.dichte(s).flatten(order='F')
        if self.reward_only:
            self.objective_value = -np.dot(self.S_Star.flatten(order='F'), S_current)
        elif self.reward_only2:
            rho_sum = S_current
            denom = np.maximum(rho_sum, 1.0)
            self.objective_value = -np.sum(rho_sum / denom)
        else:
            self.S = (self.S_Star.flatten(order='F') - S_current)**2
            self.objective_value = np.sum(self.S)
        if (
            self.last_logged_s is None or
            np.linalg.norm(s - self.last_logged_s) > 1e-4
        ):
            self.objective_history.append(self.objective_value)
            self.s_history.append(s.copy())
            self.last_logged_s = s.copy()
            if "objective" in self.plot:
                frame_idx = len(self.objective_history)
                S_target = self.S_Star.reshape(self.shape_2d, order='F')
                S_current = sp.dichte(s).reshape(self.shape_2d, order='F')


                plt.figure(figsize=(5, 5))
                plt.imshow(S_target, cmap='Reds', alpha=0.4, origin='lower',
                          extent=sp.glob.bounds["x"] + sp.glob.bounds["y"])
                plt.imshow(S_current, cmap='Blues', alpha=0.6, origin='lower',
                          extent=sp.glob.bounds["x"] + sp.glob.bounds["y"])
                if sp.glob.bounds["x"] != [0.0, 1.0] or sp.glob.bounds["y"] != [0.0, 1.0]:
                    square = plt.Rectangle((0, 0), 1, 1, linewidth=1.5, edgecolor='black',
                                          facecolor='none', linestyle='--', label='[0,1]²')
                    plt.gca().add_patch(square)

                plt.title(f"Iteration {frame_idx}")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.axis('scaled')
                plt.legend(loc='lower right', fontsize=8)
                plt.xlim(sp.glob.bounds["x"])
                plt.ylim(sp.glob.bounds["y"])
                plt.grid(True, linestyle='--', alpha=0.3)
                plt.tight_layout()

                if self.frame_dir is not None:
                    os.makedirs(self.frame_dir, exist_ok=True)
                    plt.savefig(os.path.join(self.frame_dir, f"frame_{frame_idx:03d}.png"), dpi=100)

                plt.close()

            if "derivative" in self.plot:
                grad_matrix = sp.ableitung(s).reshape((self.num_vars, self.n_points), order='C')
                max_abs = np.max(np.abs(grad_matrix))
                v_min = -max_abs
                v_max = max_abs
                grad_dir = os.path.join(self.frame_dir, "ableitungen")
                os.makedirs(grad_dir, exist_ok=True)

                for i in range(self.num_vars):
                    grad_current = grad_matrix[i].reshape(self.shape_2d, order='F')
                    plt.figure(figsize=(5, 5))
                    plt.imshow(grad_current, cmap='coolwarm', origin='lower', extent=[0, 1, 0, 1],
                              vmin=v_min, vmax=v_max)
                    plt.title(f"Ableitung Komponente {i} – Iteration {frame_idx}")
                    plt.axis('off')
                    plt.colorbar(fraction=0.046, pad=0.04)
                    plt.tight_layout()
                    plt.savefig(os.path.join(grad_dir, f"frame_{frame_idx:03d}_grad_{i}.png"), dpi=100)
                    plt.close()
                S = sp.dichte(s).flatten(order='F')
                residual = self.S_Star.flatten(order='F') - S
                grad_contributions = grad_matrix * residual 
                gradient_magnitude_per_point = np.sum(np.abs(grad_contributions), axis=0)
                heatmap = gradient_magnitude_per_point.reshape(self.shape_2d, order='F')
                
                max_val = np.max(np.abs(heatmap))
                vmin = 0 
                vmax = max_val
                full_gradient = -2 * np.dot(grad_matrix, residual)

                objgrad_dir = os.path.join(self.frame_dir, "zielfunktion")
                os.makedirs(objgrad_dir, exist_ok=True)

                for i in range(self.num_vars):
                    grad_contrib = -2*grad_contributions[i]
                    heatmap = grad_contrib.reshape(self.shape_2d, order='F')

                    max_val = np.max(np.abs(heatmap))
                    vmin = -max_val
                    vmax = max_val
                    scalar_grad_val = full_gradient[i]
                    plt.figure(figsize=(5, 5))
                    plt.imshow(heatmap, cmap='coolwarm', origin='lower', extent=[0, 1, 0, 1],
                              vmin=vmin, vmax=vmax)
                    plt.title(f"Zielfunktionsbeitrag Var {i} – Iteration {frame_idx}\n∇J[{i}] = {scalar_grad_val:.2e}")
                    plt.axis('off')
                    plt.colorbar(fraction=0.046, pad=0.04)
                    plt.tight_layout()
                    plt.savefig(os.path.join(objgrad_dir, f"frame_{frame_idx:03d}_grad_{i}.png"), dpi=100)
                    plt.close()


        return self.objective_value


    def gradient(self, s):
        """
        Objective gradient ∂J/∂s.

        Implementation
        --------------
        - default: -2 * A * (S_Star - S), with A = sp.ableitung(s) reshaped to (num_vars, nx*ny).
        - reward_only: -A * S_Star
        - reward_only2: piecewise derivative via indicator f'(ρ) = 1[ρ < 1].

        Returns
        -------
        g : ndarray, shape (num_vars,)
        """
        self.s = s
        S = sp.dichte(s).flatten(order='F')
        grad_matrix = sp.ableitung(s).reshape((self.num_vars, self.n_points), order='C')


        if self.reward_only:
            grad = -np.dot(grad_matrix, self.S_Star.flatten(order='F'))
        elif self.reward_only2:
            rho = sp.dichte(s).flatten(order='F') 
            f_prime = np.where(rho < 1.0, 1.0, 0.0)
            grad_matrix = sp.ableitung(s).reshape((self.num_vars, self.n_points), order='C')
            grad = -np.dot(grad_matrix, f_prime)
        else:
            residual = self.S_Star.flatten(order='F') - S
            grad = -2 * np.dot(grad_matrix, residual)
        self.gradient_history.append(grad.copy())

        return grad


    def hessian_matrix(self, s):
        """
        Objective Hessian ∂²J/∂s² (dense).

        Implementation
        --------------
        - default: Gauss–Newton (Σ dS_i dS_i^T) − second-order correction (Σ error_i * d²S_i),
          multiplied by 2 and symmetrized.
        - reward_only: contraction of per-point Hessians with S_Star, symmetrized.
        - reward_only2: explicit second derivatives of the saturating reward.

        Returns
        -------
        H : ndarray, shape (num_vars, num_vars)
        """
        self.s = s.copy()
        grad_matrix = sp.ableitung(s).reshape((self.num_vars, self.nx, self.ny))
        hess_tensor = sp.hessian(s).reshape((self.num_vars, self.num_vars, self.nx, self.ny))
        H4 = sp.hessian(s).reshape((self.num_vars, self.num_vars, self.nx, self.ny))

        H_total = np.zeros((self.num_vars, self.num_vars))
        if self.reward_only:
            Hsym = 0.5 * (H4 + H4.swapaxes(0, 1))
            Hsym_flat = Hsym.reshape(self.num_vars, self.num_vars, self.nx * self.ny, order='F')
            w = self.S_Star.flatten(order='F').astype(np.float64)
            H_total = -np.einsum('abk,k->ab', Hsym_flat, w)
            H_total = 0.5 * (H_total + H_total.T)
        elif self.reward_only2:
            rho = sp.dichte(s).flatten(order='F')
            rho_sum = np.sum(rho)
            denom = max(rho_sum, 1.0)
            H_total = np.zeros((self.num_vars, self.num_vars))
            for i in range(self.num_vars):
                for j in range(i + 1):
                    d2f_ij = 0.0
                    for k in range(self.n_points):
                        rho_k = rho[k]
                        d_rho_i = grad_matrix[i, k]
                        d_rho_j = grad_matrix[j, k]
                        d2_rho_ij = hess_tensor[i, j].flatten(order='F')[k]
                        d_rho_sum_i = np.sum(grad_matrix[i])
                        d_rho_sum_j = np.sum(grad_matrix[j])
                        d2_rho_sum_ij = np.sum(hess_tensor[i, j])

                        term1 = d2_rho_ij / denom
                        term2 = (d_rho_sum_i * d_rho_j + d_rho_sum_j * d_rho_i) / (denom**2)
                        term3 = rho_k * d2_rho_sum_ij / (denom**2)
                        term4 = 2 * rho_k * d_rho_sum_i * d_rho_sum_j / (denom**3)

                        d2f_ij_k = term1 - term2 - term3 + term4
                        d2f_ij += d2f_ij_k
                    H_total[i, j] = -d2f_ij
            H_total = H_total + np.tril(H_total, -1).T  
        else:
            error = self.S_Star.flatten(order='F') - sp.dichte(s).flatten(order='F')
            for i in range(self.num_vars):
                for j in range(i + 1):
                    dS_di = grad_matrix[i]
                    dS_dj = grad_matrix[j]
                    d2S_didj = hess_tensor[i, j].flatten(order='F')
                    first_term = np.sum(dS_di * dS_dj)
                    second_term = np.sum(error * d2S_didj)
                    H_total[i, j] = 2 * (first_term - second_term)
            H_total = H_total + np.tril(H_total, -1).T
        self.hessian_history.append(H_total.copy())
        return H_total

    def hessianstructure(self):
        """
        Lower-triangular sparsity pattern for IPOPT.

        Returns
        -------
        rows, cols : tuple[ndarray, ndarray]
            Indices (int32) of the lower triangle including diagonal.
        """
        rows, cols = np.tril_indices(self.num_vars)
        return rows.astype(np.int32), cols.astype(np.int32)

    def hessian(self, s, lagrange, obj_factor):
        """
        IPOPT-compatible Hessian callback (lower triangle, packed).

        Parameters
        ----------
        s : ndarray, shape (num_vars,)
        lagrange : ndarray
            Constraint multipliers (aligned with `constraints(s)`).
        obj_factor : float
            Scalar factor for the objective Hessian.

        Returns
        -------
        h_lower : ndarray, shape (nnz_lower,)
            Lower-triangle of H_total = obj_factor*H_obj + Σ λ_i H_constr_i, packed.
        """
        H_obj = self.hessian_matrix(s)
        H_total = obj_factor * H_obj

        if self.constraint_obj is not None:
            H_constr = self.constraint_obj.hessian(s, lagrange)  
            H_total += lagrange[0]* H_constr

        rows, cols = np.tril_indices(self.num_vars)
        return H_total[rows, cols].astype(np.float64)

    def constraints(self, s):
        """
        Concatenate constraint values for the current s.

        Returns
        -------
        c : ndarray, shape (m,)
            Empty array if no constraint object is attached.
        """
        if self.constraint_obj is None:
            return np.array([], dtype=np.float64)
        return self.constraint_obj.constraint(s)
    
    def jacobianstructure(self):
        """
        Dense Jacobian structure (row/col indices) for IPOPT.

        Returns
        -------
        rows, cols : tuple[ndarray, ndarray]
            All (i,j) positions for an m×n matrix when constraints exist; else ([], []).
        """
        if self.constraint_obj is None:
            return ([], [])
        s_dummy = np.zeros(self.num_vars, dtype=np.float64)
        m = len(self.constraint_obj.constraint(s_dummy))

        n = self.num_vars
        rows, cols = np.nonzero(np.ones((m, n)))
        return rows.astype(np.int32), cols.astype(np.int32)

    def jacobian(self, s):
        """
        Constraint Jacobian values in row-major (C) flattening.

        Parameters
        ----------
        s : ndarray, shape (num_vars,)

        Returns
        -------
        jac_flat : ndarray, shape (m*n,) or (0,)
            Flattened `constraint_obj.jacobian(s)` in C-order; empty if no constraints.
        """
        if self.constraint_obj is None:
            return np.array([], dtype=np.float64)
        jac = self.constraint_obj.jacobian(s)
        return jac.flatten(order="C")  