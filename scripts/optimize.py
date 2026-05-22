import numpy as np
import matplotlib.pyplot as plt
import feature_definition as sp 
import os
import json
from typing import Optional
from optimization_definition import FeatureOptimizationProblemConstraints
import sys
import time
from optimization_definition import SegmentLengthConstraint
from optimization_definition import CombinedConstraints
#from newton_optimization import NewtonOptimizer
import cyipopt
import imageio.v2 as imageio
import subprocess
import xml.etree.ElementTree as ET
from typing import Optional, Tuple
from scipy.optimize import minimize
import copy
import argparse
from skimage.transform import resize
from feature_definition import glob
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from optimization_definition import SegmentMaxLengthConstraint
import math 
from PIL import Image

"""
Staged, configurable optimization driver for feature-based topology optimization.

Responsibilities
---------------
- Load and apply JSON config to orchestrate multiple optimization stages.
- Provide wrappers for IPOPT / SciPy optimizers, logging, and visualization.
- Apply heuristics (pruning/merging) and greedy additive refinement.
- I/O helpers for densities, targets, and run artifacts.
"""



def run_ipopt_and_capture(nlp, s0, log_path):
    """
    Run an IPOPT problem while capturing stdout to a log file.

    Parameters
    ----------
    nlp : cyipopt.Problem
        Initialized IPOPT problem.
    s0 : array_like, shape (n,)
        Initial design vector.
    log_path : str
        File to write IPOPT's console output.

    Returns
    -------
    s_opt : np.ndarray, shape (n,)
        Optimized design.
    info : dict
        IPOPT solve info (e.g., 'status', 'obj_val').
    """
    sys.stdout.flush()
    original_stdout_fd = sys.stdout.fileno()
    saved_stdout_fd = os.dup(original_stdout_fd)

    with open(log_path, 'w') as log_file:
        os.dup2(log_file.fileno(), original_stdout_fd)
        try:
            s_opt, info = nlp.solve(s0)
        finally:
            os.dup2(saved_stdout_fd, original_stdout_fd)
            os.close(saved_stdout_fd)

    return s_opt, info


def load_existing_runs(path):
    """
    Load a JSON list of prior run results if it exists.

    Parameters
    ----------
    path : str

    Returns
    -------
    list
        Parsed JSON list or empty list if missing.
    """
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    else:
        return []


def plot_derivative_history(problem, out_dir):
    """
    Plot histories of gradients and Hessians recorded by the problem wrapper.

    Parameters
    ----------
    problem : FeatureOptimizationProblemConstraints
        Holds `gradient_history` and `hessian_history`.
    out_dir : str
        Output directory for PNG plots.

    Notes
    -----
    Respects `problem.plot` flags: 'derivative' and 'hessian'.
    """
    os.makedirs(out_dir, exist_ok=True)

    if "derivative" not in problem.plot and "hessian" not in problem.plot:
        log("ℹ️ Plots für Ableitungen/Hesse nicht aktiviert – überspringe.", level="info")
        return

    if "derivative" in problem.plot and problem.gradient_history:
        grad_array = np.array(problem.gradient_history)
        grad_iter, num_vars = grad_array.shape

        for i in range(num_vars):
            plt.figure()
            plt.plot(range(grad_iter), grad_array[:, i], label=f"∇J[{i}]")
            plt.xlabel("Iteration")
            plt.ylabel("Gradient")
            plt.title(f"Gradientverlauf ∇J[{i}]")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"gradient_{i}.png"), dpi=200)
            plt.close()

    if "hessian" in problem.plot and problem.hessian_history:
        hess_array = np.array(problem.hessian_history)
        if hess_array.ndim != 3:
            log(" Hesse-Historie fehlerhaft – kein Plot.", level="warning")
        else:
            hess_iter, num_vars, _ = hess_array.shape
            for i in range(num_vars):
                for j in range(num_vars):
                    plt.figure()
                    plt.plot(range(hess_iter), hess_array[:, i, j], label=f"H[{i},{j}]")
                    plt.xlabel("Iteration")
                    plt.ylabel("Hesse")
                    plt.title(f"Hesseverlauf H[{i},{j}]")
                    plt.grid(True, linestyle="--", alpha=0.5)
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f"hessian_{i}_{j}.png"), dpi=200)
                    plt.close()




def plot_convergence(results_per_setting, out_dir, trans_type, param):
    """
    Plot objective trajectories of multiple runs and highlight best/worst.

    Parameters
    ----------
    results_per_setting : list[dict]
        Each dict contains 'objective_values' for a run.
    out_dir : str
    trans_type : str
        Label for transition/boundary method.
    param : Any
        Parameter value shown in the title.
    """
    plt.figure(figsize=(8, 5))
    max_val = 1e-8
    best_val = float('inf')
    worst_val = -float('inf')
    best_idx = None
    worst_idx = None

    for i, run in enumerate(results_per_setting):
        vals = run["objective_values"]
        if not vals:
            continue
        final = vals[-1]
        if final < best_val:
            best_val = final
            best_idx = i
        if final > worst_val:
            worst_val = final
            worst_idx = i
        max_val = max(max_val, max(vals))

    for i, run in enumerate(results_per_setting):
        vals = run["objective_values"]
        if not vals:
            continue
        color = 'black'
        lw = 1
        if i == best_idx:
            color = 'green'
            lw = 2
        elif i == worst_idx:
            color = 'red'
            lw = 2
        plt.plot(range(len(vals)), vals, color=color, linewidth=lw)

    plt.title(f"{trans_type} (param={param}) – Zielfunktion pro Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Zielfunktionswert")
    plt.ylim(0, max_val * 1.1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "convergence_plot.jpeg"), dpi=300)
    plt.close()


def generate_gif_from_frames(out_path, frame_dir, duration=0.5):
    """
    Assemble a GIF from 'frame_*.png' images in a directory.

    Parameters
    ----------
    out_path : str
    frame_dir : str
    duration : float
        Duration per frame (seconds).
    """
    images = []
    files = sorted([f for f in os.listdir(frame_dir) if f.startswith("frame_") and f.endswith(".png")])
    for file in files:
        images.append(imageio.imread(os.path.join(frame_dir, file)))
    imageio.mimsave(out_path, images, duration=duration)


def create_s_star(P, Q, r):
    """
    Build a target density S* for a single Pill feature.

    Parameters
    ----------
    P, Q : array_like, shape (2,)
        Segment endpoints.
    r : float
        Width parameter p.

    Returns
    -------
    S_star : np.ndarray, shape (ny*nx,)
        Flattened (Fortran order) density field.
    """

    target_shape = sp.Pill(0, P, Q, r)
    sp.glob.shapes = [target_shape]
    return sp.dichte(target_shape.optvar()).flatten(order='F')

def configure_glob(
    *,
    r=0.1,
    bounds=None,
    extension=1.4,
    base_shape=None,
    set_n=False,
    boundary="smoothstep_extended",
    transition=0.1,
    bezier_order=5
):
    """
    Configure global grid/transition/boundary settings derived from a base shape.

    Parameters
    ----------
    r : float
        Feature width used to infer padding.
    bounds : dict or None
        {"x":[xmin,xmax], "y":[ymin,ymax]}; if None, auto-extends around [0,1]^2.
    extension : float
        Boundary extension used by feature boundary functions.
    base_shape : tuple[int,int]
        (nx, ny) of the *target* inner [0,1]^2 grid; required.
    set_n : bool
        If True, adapts `glob.n` to new bounds at same dx,dy resolution.
    boundary : str
        Boundary method name to set.
    transition : float
        Transition width parameter.
    """
    if base_shape is None:
        raise ValueError(" configure_glob: base_shape muss gesetzt sein (z.B. shape von S_star).")

    inner_nx, inner_ny = base_shape
    dx = 1.0 / inner_nx
    dy = 1.0 / inner_ny

    if bounds is None:
        xmin = -(r + extension)
        xmax = 1.0 + (r + extension)
        ymin = -(r + extension)
        ymax = 1.0 + (r + extension)
        bounds = {"x": [xmin, xmax], "y": [ymin, ymax]}

    sp.glob.bounds = bounds

    if set_n:
        Lx = bounds["x"][1] - bounds["x"][0]
        Ly = bounds["y"][1] - bounds["y"][0]
        nx = int(round(Lx / dx))
        ny = int(round(Ly / dy))
        sp.glob.n = [nx, ny]

    sp.glob.transition = transition
    sp.glob.boundary = boundary
    if boundary == "bezier":
        sp.glob.bezier_order(bezier_order)
    sp.glob.extension = extension
    sp.glob.update_grid_metrics()

def generate_testcase_matrix(n, test_type="T1", value=1.0, use_bigger_box=False):
    """
    Create synthetic target density patches for quick tests.

    Parameters
    ----------
    n : int
        Patch size in cells.
    test_type : {"T1","T2","T3"}
    value : float
        Patch intensity.
    use_bigger_box : bool
        If True, place patches in extended box coordinates.

    Returns
    -------
    S_star : list[float]
        Flattened row-major list (C-order) of shape (nx, ny).
    """
    nx, ny = sp.glob.n

    S_star = np.zeros((nx, ny))

    if use_bigger_box:
      if test_type == "T1":
          S_star[37:37+n, 61-n:61] = value
      elif test_type == "T2":
          S_star[37:37+n, 61-n:61] = value  
          S_star[61-n:61, 61-n:61] = value      
      elif test_type == "T3":
          S_star[37:37+n, 61-n:61] = 1.0
          S_star[61-n:61, 61-n:61] = 0.5
    else:
        
      if test_type == "T1":
          S_star[:n, -n:] = value
      elif test_type == "T2":
          S_star[:n, -n:] = value  
          S_star[-n:, -n:] = value      
      elif test_type == "T3":
          S_star[:n, -n:]  = 1.0
          S_star[-n:, -n:]  = 0.5

    return S_star.flatten(order="C").tolist()



def resize_density_field(
    S_padded_prev: np.ndarray,
    prev_bounds: dict,
    new_bounds: dict,
    new_n: tuple[int, int],
    center_box_bounds={"x": [0,1], "y": [0,1]},
):
    
    """
    Map a padded density field from old to new bounds while preserving the inner [0,1]^2 content.

    Parameters
    ----------
    S_padded_prev : np.ndarray, shape (ny_prev*nx_prev,)
        Flattened (Fortran order) prior stage density.
    prev_bounds, new_bounds : dict
        {"x":[xmin,xmax], "y":[ymin,ymax]} before/after.
    new_n : tuple[int,int]
        New grid (nx, ny).
    center_box_bounds : dict
        Box to preserve exactly (default: [0,1]^2).

    Returns
    -------
    S_resized : np.ndarray, shape (new_n_y*new_n_x,)
        Flattened (C-order) resized field.
    """
    aspect_ratio = (prev_bounds["x"][1] - prev_bounds["x"][0]) / (prev_bounds["y"][1] - prev_bounds["y"][0])

    n_total = S_padded_prev.size
    ny_prev = int(round(np.sqrt(n_total / aspect_ratio)))
    nx_prev = int(round(n_total / ny_prev))

    assert nx_prev * ny_prev == n_total, f"Inkompatible reshape-Größe: {nx_prev}×{ny_prev} ≠ {n_total}"
    shape_prev = (ny_prev, nx_prev)
    S_prev_2d = S_padded_prev.reshape(shape_prev, order="F")

    dx_prev = (prev_bounds["x"][1] - prev_bounds["x"][0]) / nx_prev
    dy_prev = (prev_bounds["y"][1] - prev_bounds["y"][0]) / ny_prev

    x0_idx_prev = int(round((center_box_bounds["x"][0] - prev_bounds["x"][0]) / dx_prev))
    x1_idx_prev = int(round((center_box_bounds["x"][1] - prev_bounds["x"][0]) / dx_prev))

    y0_idx_prev = int(round((center_box_bounds["y"][0] - prev_bounds["y"][0]) / dy_prev))
    y1_idx_prev = int(round((center_box_bounds["y"][1] - prev_bounds["y"][0]) / dy_prev))

    S_inner = S_prev_2d[y0_idx_prev:y1_idx_prev, x0_idx_prev:x1_idx_prev]

    S_resized = embed_density_in_extended_box(
        S_inner,
        grid_size_full=new_n,
        bounds_full=new_bounds
    )
    return S_resized


def embed_density_in_extended_box(S_inner: np.ndarray, grid_size_full: tuple[int, int], bounds_full: dict):
    """
    Embed a [0,1]^2 density into a larger zero-padded field matching `bounds_full`.

    Parameters
    ----------
    S_inner : np.ndarray, shape (ny_inner, nx_inner)
    grid_size_full : tuple[int,int]  # (nx_full, ny_full)
    bounds_full : dict

    Returns
    -------
    S_padded : np.ndarray, shape (ny_full*nx_full,)
        Flattened (C-order) padded field.
    """
    nx_full, ny_full = grid_size_full
    nx_inner, ny_inner = S_inner.shape

    dx = 1.0 / nx_inner

    total_x = nx_full
    total_y = ny_full

    pad_x_total = total_x - nx_inner
    pad_y_total = total_y - ny_inner

    pad_x_left = pad_x_total // 2
    pad_x_right = pad_x_total - pad_x_left

    pad_y_bottom = pad_y_total // 2
    pad_y_top = pad_y_total - pad_y_bottom
    if any(v < 0 for v in [pad_x_left, pad_x_right, pad_y_bottom, pad_y_top]):
        raise ValueError(
            f"Padding would be negative! Check bounds/grid. "
            f"Computed: left={pad_x_left}, right={pad_x_right}, "
            f"bottom={pad_y_bottom}, top={pad_y_top} – "
            f"with dx={dx}, bounds={bounds_full}, full grid={grid_size_full}"
        )
    S_padded = np.pad(
        S_inner,
        pad_width=((pad_x_left, pad_x_right), (pad_y_bottom, pad_y_top)),
        mode='constant',
        constant_values=0.0
    )
    
    return S_padded.flatten(order='C')


def visualize_1d_array(density_array, save_path=None, shape=None):
    """
    Quick heatmap of a 1D (reshaped) or 2D density array.

    Parameters
    ----------
    density_array : array_like
    save_path : str or None
    shape : tuple[int,int] or None
        Required if `density_array` is 1D.
    """
    density_array = np.array(density_array)

    if density_array.ndim == 1:
        if shape is None:
            raise ValueError("🔢 Bitte shape=(rows, cols) angeben für 1D-Array.")
        density_map = density_array.reshape(shape)
    else:
        density_map = density_array

    plt.figure(figsize=(8, 6))
    plt.imshow(density_map, cmap="viridis", origin="lower", aspect="auto")
    plt.colorbar(label="Pseudo-Density")
    plt.title(f"Pseudo-Density Map ({density_map.shape[0]}x{density_map.shape[1]})")
    plt.xlabel("X-Koordinate")
    plt.ylabel("Y-Koordinate")
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def generate_optimization_stages(
    extension_schedule=[1.0, 0.0],
    param=2,
    base_transition=0.1,
    aggregation={"method": "p-norm", "params": {"p": 7}},
    n_iter_grob=200,
    n_iter_fine=200,
    tol_grob=1e-6,
    tol_fine=1e-8,
    make_gif = True,
    fixed_r_schedule=None
):
    """
    Build a default sequence of coarse→fine stages with optional extension ramp-down.

    Returns
    -------
    stages : list[dict]
        Stage dicts consumable by `run_optimization_stage`.
    """
    stages = []
    reward_only = True
    for i, ext in enumerate(extension_schedule):
        is_last = (i == len(extension_schedule) - 1)
        stage = {
            "name": f"stage_{i}_{'final' if is_last else 'ext'}",
            "n_iterations": n_iter_fine if is_last else n_iter_grob,
            "trans_type": "smoothstep" if is_last else "smoothstep_extended",
            "param": param,
            "aggregation": aggregation,
            "reward_only": False if is_last else True,
            "reward_only2": False,
            "optimizer_type": "ipopt",
            "use_constraints": True,
            "make_gif": make_gif,
            "bounds": {
                "x": [0,1],
                "y": [0,1]
            },
            "extension": ext,
            "transition": base_transition,
            "tolerance": tol_fine if is_last else tol_grob,
            "fixed_r": fixed_r_schedule[i] if fixed_r_schedule else None

        }
        stages.append(stage)
        if reward_only:
            reward_only = False

    return stages


def visualize_density(density_array, shape, v_num, v_num_sum,v_ana, rel_error, save_path):
    """
    Save a density heatmap annotated with integral metrics.

    Parameters
    ----------
    density_array : array_like
    shape : tuple[int,int]  # (ny, nx)
    v_num, v_num_sum, v_ana : float
        Reported metrics.
    rel_error : float
    save_path : str
    """
    density_map = density_array.reshape(shape)
    plt.figure(figsize=(6, 5))
    plt.imshow(density_map, cmap="viridis", origin="lower", aspect="auto")
    plt.colorbar(label="Pseudo-Density")

    textstr = f"V_num = {v_num:.5f}V_num_sum = {v_num_sum:.5f}\nV_ana = {v_ana:.5f}\nrel. Fehler = {rel_error:.2%}"
    plt.gca().text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    plt.title(f"Density (shape={shape[0]}x{shape[1]})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()



def resize_S_star(S_star, new_shape):
    """
    Resample a target density to a new resolution using bilinear anti-aliased resize.

    Parameters
    ----------
    S_star : np.ndarray, shape (ny, nx)
    new_shape : tuple[int,int]  # (ny_new, nx_new)

    Returns
    -------
    S_star_resized : np.ndarray, shape new_shape
    """
    S_star_resized = resize(S_star, new_shape, order=1, preserve_range=True, anti_aliasing=True)
    return S_star_resized

def save_resized_target(S_star_resized, name="resampled", base_dir="precomputed_targets"):
    """
    Persist resized target to disk as .npy and .png for reuse/inspection.

    Parameters
    ----------
    S_star_resized : np.ndarray
    name : str
    base_dir : str
    """
    os.makedirs(base_dir, exist_ok=True)
    np.save(os.path.join(base_dir, f"cfs_{name}.npy"), S_star_resized)

    plt.imsave(os.path.join(base_dir, f"cfs_{name}.png"), S_star_resized, cmap="viridis")

    print(f"✅ Neues Ziel gespeichert unter: {os.path.join(base_dir, f'cfs_{name}.npy')}")

LOG_LEVELS = {
    "none": 0,
    "warning": 1,
    "info": 2,
    "debug": 3
}

def log(msg, level="info"):
    """
    Conditional logger controlled via `pill.glob.log_level`.

    Parameters
    ----------
    msg : str
    level : {"none","warning","info","debug"}
    """
    current_level = LOG_LEVELS.get(glob.log_level, 2)
    msg_level = LOG_LEVELS.get(level, 2)
    if msg_level <= current_level:
        print(msg)


def load_config(config_path: str) -> dict:
    """
    Load JSON config and propagate `log_level` into `pill.glob`.

    Parameters
    ----------
    config_path : str

    Returns
    -------
    dict
        Parsed configuration.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    glob.log_level = config.get("log_level", "info")

    #should we push other stuff to glob here as well? 

    return config

def parse_args():
    """
    CLI parser for density + config paths (short helper; see also the main() parser below).

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Starte Feature-basierte Optimierung.")
    parser.add_argument("density_path", help="Pfad zur Dichteausgabe (z.B. *.density.xml)")
    parser.add_argument("config_path", help="Pfad zur JSON-Konfigurationsdatei")
    return parser.parse_args()




def find_best_grid(num_crosses: int) -> tuple[int, int]:
    """
    Find near-square (rows, cols) tiling for a requested number of cross blocks.

    Parameters
    ----------
    num_crosses : int

    Returns
    -------
    rows, cols : tuple[int,int]
    """
    best_rows, best_cols = None, None
    min_diff = float('inf')

    for rows in range(1, num_crosses + 1):
        cols = math.ceil(num_crosses / rows)
        diff = abs(rows - cols)

        if rows * cols >= num_crosses and diff < min_diff:
            best_rows, best_cols = rows, cols
            min_diff = diff

    return best_rows, best_cols


def build_bounds(s0: np.ndarray, fixed_r: Optional[float], smaller_box: bool, inactive_idx=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct lower/upper bounds for IPOPT runs with optional fixed radii and inactive indices.

    Parameters
    ----------
    s0 : np.ndarray, shape (n,)
    fixed_r : float or None
    smaller_box : bool
        If True, shrink [0,1] box to [0.02,0.98] for P/Q components.
    inactive_idx : iterable[int] or None
        Variables to be fixed to s0.

    Returns
    -------
    lb, ub : np.ndarray, np.ndarray
    """
    lb = np.zeros_like(s0)
    ub = np.ones_like(s0)
    if smaller_box:
        lb = lb + 0.02
        ub = ub - 0.02
    for i in range(4, len(s0), 5):
        lb[i] = fixed_r if fixed_r is not None else 0.005
        ub[i] = fixed_r if fixed_r is not None else 2.0
    if inactive_idx is not None:
        for k in np.atleast_1d(inactive_idx):
            lb[k] = s0[k]
            ub[k] = s0[k]
    return lb, ub



def solve_trust_constr(problem, s0, lb, ub, constraint_obj=None,
                       maxiter=500, verbose=3):
    """
    Solve with SciPy 'trust-constr' using exact gradient/Hessian (and constraint Hessians).

    Parameters
    ----------
    problem : FeatureOptimizationProblemConstraints
    s0 : np.ndarray
    lb, ub : np.ndarray
    constraint_obj : object or None
        Exposes .constraint/.jacobian/.hessian.
    maxiter : int
    verbose : int

    Returns
    -------
    res : OptimizeResult
    """

    def fun(x):  return float(problem.objective(x))
    def jac(x):  return np.asarray(problem.gradient(x), dtype=float)
    def hess(x): return np.asarray(problem.hessian_matrix(x), dtype=float)

    constraints = []
    if constraint_obj is not None:
        def c_vals(x):
            return np.atleast_1d(constraint_obj.constraint(x).astype(float)).ravel()
        def J_all(x):
            J = np.atleast_2d(constraint_obj.jacobian(x).astype(float))
            if J.shape[0] != c_vals(x).size:
                J = J.reshape(c_vals(x).size, -1)
            return J
        m_probe = c_vals(s0).size
        for i in range(m_probe):
            def c_fun_i(x, i=i):   return float(c_vals(x)[i])
            def c_jac_i(x, i=i):   return J_all(x)[i, :].reshape(1, -1)
            def c_hess_i(x, v, i=i):
                lam = np.zeros(m_probe, dtype=float); lam[i] = float(v)
                return np.asarray(constraint_obj.hessian(x, lam), dtype=float)
            constraints.append(NonlinearConstraint(c_fun_i, 0.0, np.inf, jac=c_jac_i, hess=c_hess_i))

    res = minimize(fun, s0, method="trust-constr",
                   jac=jac, hess=hess,
                   bounds=Bounds(lb, ub),
                   constraints=constraints,
                   options=dict(maxiter=maxiter, verbose=verbose,
                                gtol=1e-8, xtol=1e-10, barrier_tol=1e-8,
                                initial_tr_radius=0.25))
    return res


def penalty_wrap(problem, constraint_obj, rho: float):
    """
    Quadratic-penalty wrapper φ = 0.5*rho*Σ max(0, -c_i)^2 for ≥-type constraints.

    Returns
    -------
    fun, jac, hess, hessp : callables
        Penalized objective, gradient, Hessian, Hessian-vector product.
    """

    def _all_constraints(x):
        if constraint_obj is None:
            return np.zeros(0), np.zeros((0, len(x))), [np.zeros((len(x), len(x)))]
        c = np.atleast_1d(constraint_obj.constraint(x).astype(float)).ravel()
        m = c.size
        J = np.atleast_2d(constraint_obj.jacobian(x).astype(float))
        if J.shape[0] != m:
            J = J.reshape(m, -1)
        H_list = []
        for i in range(m):
            lam = np.zeros(m, dtype=float)
            lam[i] = 1.0
            Hi = np.asarray(constraint_obj.hessian(x, lam), dtype=float)
            H_list.append(Hi)
        return c, J, H_list

    def fun(x):
        f = float(problem.objective(x))
        if rho <= 0.0 or constraint_obj is None:
            return f
        c, J, Hs = _all_constraints(x)
        neg = c < 0.0
        if not np.any(neg):
            return f
        return f + 0.5 * rho * np.sum((-c[neg])**2)

    def jac(x):
        g = np.asarray(problem.gradient(x), dtype=float)
        if rho <= 0.0 or constraint_obj is None:
            return g
        c, J, Hs = _all_constraints(x)
        neg = c < 0.0
        if not np.any(neg):
            return g
        return g + rho * (c[neg] @ J[neg, :])
    def hess(x):
        H = np.asarray(problem.hessian_matrix(x), dtype=float)
        if rho <= 0.0 or constraint_obj is None:
            return H
        c, J, Hs = _all_constraints(x)
        neg_idx = np.where(c < 0.0)[0]
        for i in neg_idx:
            Hi = Hs[i]
            Ji = J[i, :].reshape(-1, 1)
            H += rho * (Ji @ Ji.T + c[i] * Hi)
        return H
    def hessp(x, p):
        return hess(x) @ p
    return fun, jac, hess, hessp


def run_optimization_stage(s0, S_star, stage_config,inactive_idx = None, stage_id=0, out_dir="out"):
    """
    Execute a single optimization stage with selected optimizer and options.

    Parameters
    ----------
    s0 : array_like, shape (n,)
        Initial design vector (5 per feature).
    S_star : np.ndarray, shape (ny, nx) or (ny*nx,)
        Target density; reshaped internally as needed.
    stage_config : dict
        Keys like 'optimizer', 'max_iter', 'tolerance', 'reward_only', 'constraints', etc.
    inactive_idx : array_like[int] or None
        Indices to freeze.
    stage_id : int | str
    out_dir : str

    Returns
    -------
    s_opt : np.ndarray, shape (n_opt,)
    stage_history : list[dict]
        One-element list with run metadata and history vectors.
    """
    start_time = time.time()
    if inactive_idx is None:
        inactive_idx = np.array([], dtype=int)
    else:
        inactive_idx = np.array(inactive_idx, dtype=int)
        inactive_idx = inactive_idx[(inactive_idx >= 0) & (inactive_idx < len(s0))]
        inactive_idx = np.unique(inactive_idx)
    for key in ["transition", "extension","extension_kind", "boundary", "N", "k", "combine", "p", "n", "bounds", "beta", "bezier_order", "gamma"]:
        if key in stage_config:
            setattr(sp.glob, key, stage_config[key])
    if "n" in stage_config:
        nx, ny = stage_config["n"]
        sp.glob._n = [nx, ny]
        sp.glob.dx = 1.0 / nx
        sp.glob.dy = 1.0 / ny
    if "bounds" in stage_config:
        sp.glob._bounds = stage_config["bounds"]


    log(f"[STAGE {stage_id}] combine={sp.glob.combine}, p={getattr(sp.glob,'p',None)}, "
        f"extension={sp.glob.extension}, n={sp.glob.n}, bounds={sp.glob.bounds}", level="info")

    os.makedirs(out_dir, exist_ok=True)
    frame_dir = os.path.join(out_dir, f"frames_stage_{stage_id}")
    os.makedirs(frame_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"log_stage_{stage_id}.txt")
    s0 = np.array(s0)

    num_vars = len(s0)
    use_constraints = stage_config.get("use_constraints", False)
    fixed_r = stage_config.get("fixed_r", None)
    make_gif = stage_config.get("make_gif", False)
    tolerance = stage_config.get("tolerance", 1e-6)
    optimizer_type = stage_config.get("optimizer", "ipopt")
    n_iterations = stage_config.get("max_iter", 100)
    reward_only = stage_config.get("reward_only", False)
    check_derivatives = stage_config.get("check_derivatives", False)
    use_first_derivative = stage_config.get("use_first_derivative",False)
    smaller_box = stage_config.get("smaller_box",False)
    log(f" Starte Optimierungsstufe {stage_id} ({optimizer_type}), Iterationen: {n_iterations}", level="info")

    if use_constraints:
        constraints_list = build_constraints_from_config(stage_config)
        combined_constraint = CombinedConstraints(constraints_list)
    else:
        combined_constraint = None


    problem = FeatureOptimizationProblemConstraints(
        num_vars=num_vars,
        S_Star=S_star,
        constraint_obj=combined_constraint,
        plot=True,
        frame_dir=frame_dir,
        reward_only=reward_only
    )

    if optimizer_type == "ipopt":
        lb = np.zeros_like(s0)
        ub = np.ones_like(s0)
        if smaller_box:
            lb = lb+0.02
            ub = ub-0.02
        for i in range(4, len(s0), 5):
            lb[i] = fixed_r if fixed_r is not None else 0.005
            ub[i] = fixed_r if fixed_r is not None else 0.5
        if inactive_idx is not None:
            for k in np.atleast_1d(inactive_idx):
                lb[k] = s0[k]
                ub[k] = s0[k]
        if use_constraints:
            constraint_vals = problem.constraints(s0)
            m = len(constraint_vals)
            cl = np.zeros(m, dtype=float)
            cu = np.full(m, 1e20, dtype=float)
            log(f"[STAGE {stage_id}] Constraints aktiv: m={m}", level="info")
        else:
            m = 0
            cl, cu = [], []

        nlp = cyipopt.Problem(
            n=num_vars, m=m, problem_obj=problem,
            lb=lb, ub=ub, cl=cl, cu=cu
        )
        nlp.add_option("tol", tolerance)
        nlp.add_option("max_iter", n_iterations)
        if use_first_derivative:
            nlp.add_option("hessian_approximation", "limited-memory")
            nlp.add_option("limited_memory_max_history", 3) 
            nlp.add_option("nlp_scaling_method", "gradient-based")
            nlp.add_option("warm_start_init_point", "yes") 

        if check_derivatives:
            nlp.add_option("derivative_test", "second-order")
            nlp.add_option("derivative_test_tol", 1e-6)
            nlp.add_option("derivative_test_perturbation", 1e-8) 
            nlp.add_option("derivative_test_print_all", "yes")

        s_opt, info = run_ipopt_and_capture(nlp, s0, log_path)
        final_value = info["obj_val"]
        status = info["status"]
        objective_values = problem.objective_history

    elif optimizer_type == "newton":
        #optimizer = NewtonOptimizer(problem, max_iter=n_iterations)
        #s_opt = optimizer.optimize(s0)
        #final_value = problem.objective(s_opt)
        #status = 0
        #objective_values = problem.objective_history
        raise NotImplementedError("Newton optimizer not implemented yet.")

    elif optimizer_type == "trust-constr":
        lb, ub = build_bounds(s0, fixed_r, smaller_box, inactive_idx)
        res = solve_trust_constr(problem, s0, lb, ub,
                                constraint_obj=problem.constraint_obj if use_constraints else None,
                                maxiter=n_iterations, verbose=3)
        s_opt = res.x
        final_value = res.fun
        status = res.status
        objective_values = problem.objective_history

    elif optimizer_type == "trust-exact":
        fun = lambda x: float(problem.objective(x))
        jac = lambda x: np.asarray(problem.gradient(x), dtype=float)
        hess = lambda x: np.asarray(problem.hessian_matrix(x), dtype=float)
        res = minimize(fun, s0, method="trust-exact", jac=jac, hess=hess,
                      options=dict(maxiter=n_iterations, verbose=3, gtol=1e-8))
        s_opt = res.x
        final_value = res.fun
        status = res.status
        objective_values = problem.objective_history

    elif optimizer_type in ("trust-ncg", "newton-cg-penalty"):
        rho_pen = 1e2 if use_constraints else 0.0
        fun, jac, hess, hessp = penalty_wrap(problem,
                                            problem.constraint_obj if use_constraints else None,
                                            rho=rho_pen)
        method = "trust-ncg" if optimizer_type == "trust-ncg" else "Newton-CG"
        if method == "trust-ncg":
            res = minimize(fun, s0, method="trust-ncg", jac=jac, hessp=hessp,
               options=dict(maxiter=n_iterations, gtol=1e-8, initial_tr_radius=0.25, disp=True))
        else:
            res = minimize(fun, s0, method="Newton-CG", jac=jac, hessp=hessp,
                          options=dict(maxiter=n_iterations, xtol=1e-8, disp=True))
        s_opt = res.x
        final_value = res.fun
        status = res.status
        objective_values = problem.objective_history
    else:
        raise ValueError(f"Unbekannter Optimierer: {optimizer_type}")
    end_time = time.time()
    runtime = end_time - start_time

    if make_gif:
        gif_path = os.path.join(out_dir, f"run_stage_{stage_id}.gif")
        generate_gif_from_frames(gif_path, frame_dir, duration=0.5)

    obj_path = os.path.join(out_dir, f"objective_values_stage_{stage_id}.txt")
    np.savetxt(obj_path, objective_values)

    deriv_plot_dir = os.path.join(out_dir, f"derivatives_stage_{stage_id}")
    plot_derivative_history(problem, deriv_plot_dir)
    s0 = np.array(s0)
    result = {
        "index": stage_id,
        "s0": s0.tolist(),
        "s_opt": s_opt.tolist(),
        "objective_values": objective_values,
        "final_value": final_value,
        "status": status,
        "success": status == 0,
        "config": stage_config.copy(),
        "runtime": runtime
    }

    s_log_path = os.path.join(out_dir, f"s_log_stage_{stage_id}.txt")
    np.savetxt(s_log_path, np.vstack(problem.s_history), fmt="%.6f")

    log(f" Stufe {stage_id} abgeschlossen: Zielwert = {final_value:.6f}, Status = {status}, Laufzeit = {runtime:.2f}s", level="info")
    return s_opt, [result]


def parse_density_file(output_file):
    """
    Load a density field from .npy or from a *.density.xml-like format.

    Parameters
    ----------
    output_file : str

    Returns
    -------
    density_grid : np.ndarray, shape (ny, nx)
    """
    ext = os.path.splitext(output_file)[1].lower()
    if ext == ".npy":
        return np.load(output_file)
    density_file = f"{output_file}"
    tree = ET.parse(density_file)
    root = tree.getroot()
    mesh_info = root.find(".//mesh")
    res_y = int(mesh_info.get("y"))
    res_x = int(mesh_info.get("x"))
    density_grid = np.zeros((res_y, res_x))
    elements = root.findall(".//element")
    for i, elem in enumerate(elements):
        density = float(elem.get("physical"))
        y_idx = i // res_x
        x_idx = i % res_x
        density_grid[y_idx, x_idx] = density
    return density_grid



def initialize_features(
    strategy: str,
    num_features: int,
    start_radius: float,
    random_seed: Optional[int] = None,
    max_seg_len: Optional[float] = None,  
    bounds: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0)),
) -> list[float]:
    """
    Initialize a 5n design vector according to a placement strategy.

    Parameters
    ----------
    strategy : {"random","cross","cross_with_horizontal"}
    num_features : int
    start_radius : float
    random_seed : int or None
    max_seg_len : float or None
        Cap on |Q-P| in world coordinates.
    bounds : ((xmin,xmax),(ymin,ymax))

    Returns
    -------
    s0 : list[float], length 5*num_features
        Clipped to [0,1] where appropriate.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    log(f"📌 Strategie: {strategy} | num_features={num_features} | radius={start_radius} | l_max={max_seg_len}", level="debug")

    (xmin, xmax), (ymin, ymax) = bounds
    Lx, Ly = xmax - xmin, ymax - ymin

    def clamp_len(P, Q):
        P = np.asarray(P, float)
        Q = np.asarray(Q, float)
        v = Q - P
        ell = float(np.hypot(v[0], v[1]))
        if max_seg_len is not None and ell > max_seg_len and ell > 0:
            Q = P + v * (max_seg_len / ell)
        Q[0] = np.clip(Q[0], xmin, xmax)
        Q[1] = np.clip(Q[1], ymin, ymax)
        return P, Q

    features = []

    if strategy == "random":
        base_len = 0.1 if max_seg_len is None else min(0.1, max_seg_len)
        for _ in range(num_features):
            px = xmin + Lx * np.random.rand()
            py = ymin + Ly * np.random.rand()
            angle = 2 * np.pi * np.random.rand()
            dx = base_len * np.cos(angle)
            dy = base_len * np.sin(angle)
            P = np.array([px, py])
            Q = np.array([px + dx, py + dy])
            P, Q = clamp_len(P, Q)
            features.append([P[0], P[1], Q[0], Q[1], start_radius])

    elif strategy == "cross":
        num_crosses = math.ceil(num_features / 2)
        rows, cols = find_best_grid(num_crosses)
        dx = Lx / cols
        dy = Ly / rows
        diag = np.hypot(dx, dy)
        seg_len = diag * 0.95
        if max_seg_len is not None:
            seg_len = min(seg_len, max_seg_len)

        feature_count = 0
        for row in range(rows):
            for col in range(cols):
                if feature_count >= num_features:
                    break
                x0, x1 = xmin + col * dx, xmin + (col + 1) * dx
                y0, y1 = ymin + row * dy, ymin + (row + 1) * dy
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                u = np.array([dx, dy], float)
                u /= np.linalg.norm(u)
                half = 0.5 * seg_len * u

                if feature_count < num_features:
                    P = np.array([cx, cy]) - half
                    Q = np.array([cx, cy]) + half
                    P, Q = clamp_len(P, Q)
                    features.append([P[0], P[1], Q[0], Q[1], start_radius])
                    feature_count += 1

                if feature_count < num_features:
                    u2 = np.array([dx, -dy], float)
                    u2 /= np.linalg.norm(u2)
                    half2 = 0.5 * seg_len * u2
                    P = np.array([cx, cy]) - half2
                    Q = np.array([cx, cy]) + half2
                    P, Q = clamp_len(P, Q)
                    features.append([P[0], P[1], Q[0], Q[1], start_radius])
                    feature_count += 1

    elif strategy == "cross_with_horizontal":
        num_blocks = math.ceil(num_features / 3)
        rows, cols = find_best_grid(num_blocks)
        dx = Lx / cols
        dy = Ly / rows
        diag = np.hypot(dx, dy)
        seg_len = diag * 0.95
        if max_seg_len is not None:
            seg_len = min(seg_len, max_seg_len)

        feature_count = 0
        for row in range(rows):
            for col in range(cols):
                if feature_count >= num_features:
                    break
                x0, x1 = xmin + col * dx, xmin + (col + 1) * dx
                y0, y1 = ymin + row * dy, ymin + (row + 1) * dy
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                u = np.array([dx, dy], float); u /= np.linalg.norm(u)
                half = 0.5 * seg_len * u
                if feature_count < num_features:
                    P = np.array([cx, cy]) - half
                    Q = np.array([cx, cy]) + half
                    P, Q = clamp_len(P, Q)
                    features.append([P[0], P[1], Q[0], Q[1], start_radius])
                    feature_count += 1
                u = np.array([dx, -dy], float); u /= np.linalg.norm(u)
                half = 0.5 * seg_len * u
                if feature_count < num_features:
                    P = np.array([cx, cy]) - half
                    Q = np.array([cx, cy]) + half
                    P, Q = clamp_len(P, Q)
                    features.append([P[0], P[1], Q[0], Q[1], start_radius])
                    feature_count += 1
                if feature_count < num_features:
                    u = np.array([1.0, 0.0])
                    half = 0.5 * min(seg_len, dx * 0.95) * u
                    P = np.array([cx, y1]) - half
                    Q = np.array([cx, y1]) + half
                    P, Q = clamp_len(P, Q)
                    features.append([P[0], P[1], Q[0], Q[1], start_radius])
                    feature_count += 1

    else:
        raise ValueError(f"Unbekannte Startstrategie: '{strategy}'")

    return np.clip(np.array(features).flatten(), 0.0, 1.0).tolist()



def build_constraints_from_config(config: dict):
    """
    Build a list of constraint objects from config['constraints'].

    Notes
    -----
    - For orientation stages (reward_only=True), optionally prioritize max-length.
    - 'enforce_both' forces both min and max constraints.
    """
    constraints_cfg = config.get("constraints", {})
    constraints = []

    is_orientation = bool(config.get("reward_only", False)) 
    force_both = bool(constraints_cfg.get("segment_length", {}).get("enforce_both", False))
    seg_cfg = constraints_cfg.get("segment_length", {})
    if seg_cfg.get("enabled", False):
        l_min = seg_cfg.get("l_min", None)
        l_max = seg_cfg.get("l_max", None)

        if (not is_orientation) or force_both:
            if l_min is not None:
                constraints.append(SegmentLengthConstraint(l_min=float(l_min)))

        if is_orientation or force_both:
            if l_max is not None:
                constraints.append(SegmentMaxLengthConstraint(l_max=float(l_max)))

    return constraints



def compute_feature_contributions(s, S_star):
    """
    Per-feature error contribution by applying feature j to residual without j.

    Parameters
    ----------
    s : array_like, shape (5n,)
    S_star : np.ndarray, shape (ny, nx)

    Returns
    -------
    contributions : list[float]
        Sum of squared residuals after placing feature j on the remaining residual.
    """
    s = np.array(s)
    num_features = len(s) // 5
    contributions = []

    full_shape = S_star.shape

    for j in range(num_features):

        s_reduced = np.concatenate([s[i*5:i*5+5] for i in range(num_features) if i != j])
        rho_wo_j = sp.dichte(s_reduced).reshape(full_shape, order='F')
        error_rest = np.maximum(S_star - rho_wo_j, 0.0)


        s_j = s[j*5:j*5+5]
        rho_j = sp.dichte(s_j).reshape(full_shape, order='F')

        contribution_error = np.sum((error_rest - rho_j) ** 2)

        contributions.append(contribution_error)

    return contributions




def _make_residual(S_tgt, S_cur, thr):
    """
    Thresholded nonnegative residual mask R = 1[(S_tgt - S_cur) >= thr].

    Returns
    -------
    R : np.ndarray, same shape as inputs, values in {0,1}.
    """
    R_lin = np.clip(S_tgt - S_cur, 0.0, 1.0)
    return (R_lin >= thr).astype(float)

def _objective_mse(S_tgt, S):
    """
    Mean-squared error between two fields (flattened).

    Returns
    -------
    mse : float
    """
    D = (np.asarray(S_tgt, float) - np.asarray(S, float)).ravel()
    return float(np.dot(D, D) / max(D.size, 1))


def greedy_additive_refinement(s_init, S_tgt, config, output_dir):
    """
    Greedy additive loop: add one feature per cycle to reduce residual,
    with (orientation → single-feature convergence → combined convergence).

    Parameters
    ----------
    s_init : np.ndarray or None
    S_tgt : np.ndarray, shape (ny, nx)
    config : dict
        Uses keys under 'additive', 'global', and 'constraints'.
    output_dir : str

    Returns
    -------
    s_best : np.ndarray
        Best design found (may equal s_init).
    """
    add = dict(config.get("additive", {}))
    if not add.get("enable", False):
        return np.array(s_init, float)

    residual_thr    = float(add.get("residual_threshold", 0.5))
    max_additions   = int(add.get("max_additions", 10))
    fixed_r         = add.get("fixed_r", None)
    improve_min_rel = float(add.get("improve_min_rel", 1e-3))
    improve_min_abs = float(add.get("improve_min_abs", 0.0))
    l_max_add       = add.get("l_max", 2)

    glob_cfg        = dict(config.get("global", {}))
    constraints_cfg = dict(config.get("constraints", {}))

    if "orientation_stage" not in add:
        raise ValueError("additive.orientation_stage fehlt in der Config.")
    if "convergence_relevant" not in add:
        raise ValueError("additive.convergence_relevant fehlt in der Config.")
    if "convergence_combined" not in add:
        raise ValueError("additive.convergence_combined fehlt in der Config.")

    orient_base   = dict(add["orientation_stage"])
    conv_relevant = dict(add["convergence_relevant"])
    conv_combined = dict(add["convergence_combined"])

    orient_constraints = copy.deepcopy(constraints_cfg)
    if l_max_add is not None:
        seg = dict(orient_constraints.get("segment_length", {}))
        seg["enabled"] = True
        seg["l_max"] = float(l_max_add)
        orient_constraints["segment_length"] = seg

    orient_stage = {
        **glob_cfg, **orient_base,
        "reward_only": True,
        "use_constraints": True,
        "fixed_r":0.05,
        "constraints": orient_constraints
    }
    if fixed_r is not None:
        orient_stage["fixed_r"] = float(fixed_r)

    single_conv_stage = {
        **glob_cfg, **conv_relevant,
        "constraints": constraints_cfg,
        "reward_only": False
    }
    combined_conv_stage = {
        **glob_cfg, **conv_combined,
        "constraints": constraints_cfg,
        "reward_only": False
    }

    if s_init is None:
        s_best = None
        best_obj = np.inf
        S_cur = 0
    else:
        s_best  = np.array(s_init, float).ravel()
        S_best  = sp.dichte(s_best)
        S_cur = S_best
        best_obj = _objective_mse(S_tgt, S_best)

    additions = 0
    while additions < max_additions:
        R = _make_residual(S_tgt, S_cur, thr=residual_thr)
        if not np.any(R > 0):
            log(np.sum(R), level="info")
            log("[additive] Kein Residuum über Schwellwert mehr – Ende.", level="info")
            break

        if additions%2==0:
          s_new = np.array([0.23,0.28,0.76,0.71,0.05])
        else: 
          s_new = np.array([0.28,0.75,0.74,0.29,0.05]) 
        out_o = os.path.join(output_dir, f"add_{additions}_orient")
        s_new, _ = run_optimization_stage(s_new, R, orient_stage,
                                          stage_id=f"add_{additions}_orient",
                                          out_dir=out_o)
        log(s_new,level="info")
        out_c1 = os.path.join(output_dir, f"add_{additions}_residconv")
        s_new, _ = run_optimization_stage(s_new, R, single_conv_stage,
                                          stage_id=f"add_{additions}_residconv",
                                          out_dir=out_c1)
        log(s_new,level="info")
        if s_best is None:
            s_best = s_new
            S_best = sp.dichte(s_new)
            S_cur = S_best
            best_obj = _objective_mse(S_tgt, S_best) 
            additions +=1
            continue
        s_candidate = np.concatenate([s_best, np.array(s_new, float).ravel()])
        out_c2 = os.path.join(output_dir, f"add_{additions}_fullconv")
        s_full, _ = run_optimization_stage(s_candidate, S_tgt, combined_conv_stage,
                                           stage_id=f"add_{additions}_fullconv",
                                           out_dir=out_c2)
        S_new   = sp.dichte(s_full)
        obj_new = _objective_mse(S_tgt, S_new)
        rel_gain = (best_obj - obj_new) / max(best_obj, 1e-12)
        abs_gain = best_obj - obj_new
        log(f"[additive {additions}] OBJ {best_obj:.6e} → {obj_new:.6e}  "
            f"(Δrel={rel_gain:.3e}, Δabs={abs_gain:.3e})", level="info")

        if (abs_gain > improve_min_abs) or (rel_gain > improve_min_rel):
            s_best, S_best, best_obj = np.array(s_full, float), S_new, obj_new
            S_cur = sp.dichte(s_best)
            additions += 1
            continue
        else:
            log("[additive] Keine Verbesserung – breche ab und gebe bestes s zurück.", level="info")
            break
    return s_best


def config_glob(s):
    """
    Sync `pill.glob.shapes` from a 5n vector s without recomputing fields.

    Parameters
    ----------
    s : array_like, shape (5n,)
    """
    sp.glob.shapes = []  
    num_shapes = len(s)//5
    for shape_id in range(num_shapes):
        P = s[shape_id * 5: shape_id * 5 + 2]
        Q = s[shape_id * 5 + 2: shape_id * 5 + 4]
        p = s[shape_id * 5 + 4]

        if shape_id >= len(sp.glob.shapes):
            sp.glob.shapes.append(sp.Pill(shape_id, P, Q, p))
        else:
            sp.glob.shapes[shape_id].set(P, Q, p)

def _get_pts(s, i):
    """
    Extract P, Q, r for feature i from a 5n vector.

    Returns
    -------
    P, Q : np.ndarray, shape (2,)
    r : float
    """
    px, py, qx, qy, r = s[i*5:(i+1)*5]
    P = np.array([px, py], dtype=float)
    Q = np.array([qx, qy], dtype=float)
    return P, Q, float(r)

def _min_endpoint_distance(P1, Q1, P2, Q2):
    """
    Minimum pairwise endpoint distance between segments P1Q1 and P2Q2.

    Returns
    -------
    dmin : float
    """
    dPP = np.linalg.norm(P1 - P2)
    dPQ = np.linalg.norm(P1 - Q2)
    dQP = np.linalg.norm(Q1 - P2)
    dQQ = np.linalg.norm(Q1 - Q2)
    return min(dPP, dPQ, dQP, dQQ)

def _segment_segment_distance(P, Q, R, S, eps=1e-12):
    """
    Shortest distance between segments PQ and RS (Ericson formulas; robust to parallelism).

    Returns
    -------
    d : float
    """
    u = Q - P
    v = S - R
    w = P - R
    a = np.dot(u, u) 
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)
    D = a*c - b*b

    sc, sN, sD = 0.0, D, D
    tc, tN, tD = 0.0, D, D

    if D < eps:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = (b*e - c*d)
        tN = (a*e - b*d)
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = (-d + b)
            sD = a

    sc = 0.0 if abs(sN) < eps else sN / sD
    tc = 0.0 if abs(tN) < eps else tN / tD

    dP = w + (sc * u) - (tc * v)
    return np.linalg.norm(dP)


def compute_angle_adjacent_groups(
    s,
    theta_lim_deg=10.0,
    min_dist=0.15,
    use_segment_distance=False
):
    """
    Group features that are (a) similarly oriented and (b) adjacent.

    Parameters
    ----------
    s : np.ndarray, shape (5n,)
    theta_lim_deg : float
        Max angular deviation (degrees) to consider collinear (folded to [0,90]).
    min_dist : float
        Adjacency threshold (endpoint or segment distance).
    use_segment_distance : bool
        If True, use segment-segment distance; else endpoint-based.

    Returns
    -------
    merge_groups : list[list[int]]
    angle_dict : dict[int, list[float]]
        Debug: per-anchor angle deviations.
    """
    num_features = len(s) // 5
    config_glob(s)
    dirs = [sp.glob.shapes[i].U0 for i in range(num_features)]

    visited = set()
    merge_groups = []
    angle_dict = {}

    def angle_between_ui_uj(i, j):
        dot = float(np.clip(np.dot(dirs[i], dirs[j]), -1.0, 1.0))
        theta = np.degrees(np.arccos(dot))
        if theta > 90.0:
            theta = 180.0 - theta
        return theta

    for i in range(num_features):
        if i in visited:
            continue
        group = [i]
        visited.add(i)
        queue = [i]
        angles_here = []

        while queue:
            k = queue.pop()
            Pk, Qk, _ = _get_pts(s, k)

            for j in range(num_features):
                if j == k or j in visited:
                    continue

                theta = angle_between_ui_uj(k, j)
                angles_here.append(theta)
                if theta >= theta_lim_deg:
                    continue

                Pj, Qj, _ = _get_pts(s, j)
                if use_segment_distance:
                    d = _segment_segment_distance(Pk, Qk, Pj, Qj)
                    adjacent = (d < min_dist)
                else:
                    d = _min_endpoint_distance(Pk, Qk, Pj, Qj)
                    adjacent = (d < min_dist)

                if adjacent:
                    visited.add(j)
                    group.append(j)
                    queue.append(j)

        angle_dict[i] = angles_here
        if len(group) > 1:
            merge_groups.append(sorted(group))

    return merge_groups, angle_dict


def compute_collinearity_angles(s, theta_lim_deg=10.0):
    """
    Identify groups of similarly oriented features (θ < theta_lim_deg).

    Returns
    -------
    merge_groups : list[list[int]]
    angle_dict : dict
    """
    num_features = len(s) // 5
    config_glob(s)
    dirs = [sp.glob.shapes[i].U0 for i in range(num_features)]

    merged = set()
    merge_groups = []
    angle_dict = {}

    for i in range(num_features):
        if i in merged:
            continue
        group = [i]
        angles = []
        for j in range(num_features):
            if j == i or j in merged:
                continue

            dot = np.clip(np.dot(dirs[i], dirs[j]), -1.0, 1.0)
            theta = np.degrees(np.arccos(dot))
            if theta > 90:
                theta = 180 - theta

            angles.append(theta)

            if theta < theta_lim_deg:
                group.append(j)
                merged.add(j)

        angle_dict[i] = angles
        if len(group) > 1:
            merged.update(group)
            merge_groups.append(group)

    return merge_groups, angle_dict



def merge_group_along_direction(s, group, dirs):
    """
    Merge features along their average direction (endpoints spanned across projected extrema).

    Parameters
    ----------
    s : np.ndarray, shape (5n,)
    group : list[int]
    dirs : list[np.ndarray]
        Direction vectors U0 for all features.

    Returns
    -------
    merged : list[float]
        [px, py, qx, qy, r] of merged feature (r = max radii).
    """
    points = []
    u_avg = np.mean([dirs[i] for i in group], axis=0)
    u = u_avg / np.linalg.norm(u_avg)

    for i in group:
        px, py, qx, qy, _ = s[i * 5:(i + 1) * 5]
        points.extend([np.array([px, py]), np.array([qx, qy])])
    origin = min(points, key=lambda p: p[0])
    t_values = [(np.dot(p - origin, u), p) for p in points]
    before = [p for t, p in t_values if t < 0]
    after = [p for t, p in t_values if t > 0]
    P_merge = min(before, key=lambda p: np.dot(p - origin, u)) if before else origin
    Q_merge = max(after, key=lambda p: np.dot(p - origin, u)) if after else origin
    r_merge = max([s[i * 5 + 4] for i in group])

    return [*P_merge, *Q_merge, r_merge]


def _merge_extent_fallback(s, group):
    """
    Fallback merge via PCA along principal direction; r = max(r_i).

    Returns
    -------
    merged : list[float]
    """
    pts = []
    radii = []
    for i in group:
        P, Q, r = _get_pts(s, i)
        pts.extend([P, Q])
        radii.append(r)
    pts = np.asarray(pts, float)
    P0 = pts.mean(axis=0)
    X = pts - P0
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    u = Vt[0]
    ts = X @ u
    P_merge = P0 + u * ts.min()
    Q_merge = P0 + u * ts.max()
    r_merge = max(radii) if len(radii) else 0.0
    return [float(P_merge[0]), float(P_merge[1]),
            float(Q_merge[0]), float(Q_merge[1]),
            float(r_merge)]

def merge_group_touch_chain(
    s,
    group
):
    """
    Robust merge heuristic:
      - take the feature with the largest segment length |P-Q| as geometry,
      - set radius to min(r_i) across the group.

    Returns
    -------
    merged : list[float] = [px, py, qx, qy, r_min]
    """
    s = np.asarray(s, dtype=float).ravel()
    assert s.size % 5 == 0, "s muss 5*k Werte haben."
    n = s.size // 5
    gg, seen = [], set()
    for idx in group:
        j = int(idx)
        if 0 <= j < n and j not in seen:
            gg.append(j)
            seen.add(j)

    if len(gg) == 0:
        return _merge_extent_fallback(s, gg)

    r_min = float("inf")
    best_len = -1.0
    bestP = None
    bestQ = None

    for i in gg:
        P, Q, r = _get_pts(s, i)
        L = float(np.linalg.norm(Q - P))
        if L > best_len:
            best_len = L
            bestP = P
            bestQ = Q
        if r < r_min:
            r_min = float(r)

    if bestP is None or bestQ is None:
        return _merge_extent_fallback(s, gg)

    return [float(bestP[0]), float(bestP[1]),
            float(bestQ[0]), float(bestQ[1]),
            float(max(r_min, 0.0))]


def identify_removal_candidates(s, ARlim=0.1, URlim=0.3):
    """
    Propose removable features by area ratio (AR) and unique-area ratio (UR).

    Parameters
    ----------
    s : np.ndarray, shape (5n,)
    ARlim : float
    URlim : float

    Returns
    -------
    removal_candidates : list[int]
    metrics : dict
        Includes 'AR', 'UR', 'removal_candidates'.
    """
    AR_list, UR_list = sp.compute_AR_UR(s)
    removal_candidates = []

    for i, (ARn, URn) in enumerate(zip(AR_list, UR_list)):
        if ARn < ARlim or URn < URlim :
            removal_candidates.append(i)

    metrics = {
        "AR": AR_list,
        "UR": UR_list,
        "removal_candidates": removal_candidates
    }

    return removal_candidates, metrics


def _compute_segment_lengths(s: np.ndarray) -> np.ndarray:
    """
    Vectorized segment lengths for all features from a 5n vector.

    Returns
    -------
    seglen : np.ndarray, shape (n,)
    """
    feats = np.asarray(s, dtype=float).ravel().reshape(-1, 5)
    px, py, qx, qy = feats[:,0], feats[:,1], feats[:,2], feats[:,3]
    return np.hypot(px - qx, py - qy)

def identify_simple_removals(s: np.ndarray, min_r: float, min_seglen: float):
    """
    Remove-by-threshold candidates: r < min_r OR |P-Q| < min_seglen.

    Returns
    -------
    remove_ids : list[int]
    metrics : dict with arrays 'r' and 'seglen'
    """
    feats = np.asarray(s, dtype=float).ravel().reshape(-1, 5)
    r = feats[:, 4]
    seglen = _compute_segment_lengths(s)
    mask = (r < float(min_r)) | (seglen < float(min_seglen))
    remove_ids = np.nonzero(mask)[0].tolist()
    metrics = {"r": r, "seglen": seglen}
    return remove_ids, metrics

def apply_heuristics(
    s,
    enable_merge=False,
    theta_lim_deg=10.0,
    min_dist=0.15,
    use_segment_distance=False,
    ARlim=0.05,
    URlim=0.10,
    simple_cfg=None,
):
    """
    Heuristic pipeline:
      (0) optional SIMPLE thresholds (r/segment length),
      (1) AR/UR-based pruning,
      (2) optional angle+adjacency grouping,
      (3) optional merging.

    Returns
    -------
    s_new : np.ndarray or None
        New 5n vector if changed, else None.
    """
    s = np.asarray(s, dtype=float).ravel()
    assert s.size % 5 == 0, "s muss 5*k Werte haben."
    n = s.size // 5
    if n == 0:
        return None

    feats = s.reshape(-1, 5)
    keep_mask = np.ones(n, dtype=bool)
    changed = False

    if simple_cfg and simple_cfg.get("enabled", False):
        min_r = float(simple_cfg.get("min_r", 0.0))
        min_seglen = float(simple_cfg.get("min_seglen", 0.0))
        simple_ids, simple_metrics = identify_simple_removals(s, min_r=min_r, min_seglen=min_seglen)
        simple_set = set(int(i) for i in simple_ids)
        if simple_set:
            log(f" SIMPLE-Removal: entferne Indizes {sorted(simple_set)} "
                f"(min_r={min_r}, min_seglen={min_seglen})", level="info")
            keep_mask[list(simple_set)] = False
            changed = True
        else:
            log(" SIMPLE-Removal: keine Kandidaten", level="info")

    if keep_mask.any():
        feats_after_simple = feats[keep_mask]
        s_after_simple = feats_after_simple.ravel()
        remove_ids_ARUR, _metrics = identify_removal_candidates(
            s_after_simple, ARlim=ARlim, URlim=URlim
        )
        if remove_ids_ARUR:
            kept_idx = np.nonzero(keep_mask)[0]
            remove_orig = [int(kept_idx[i]) for i in remove_ids_ARUR]
            log(f"🔍 AR/UR-Removal: entferne Indizes {sorted(remove_orig)}", level="info")
            keep_mask[remove_orig] = False
            changed = True
        else:
            log("🔍 AR/UR-Removal: keine Kandidaten", level="info")
    else:
        return np.array([], dtype=float)

    feats_eligible = feats[keep_mask]
    if feats_eligible.shape[0] == 0:
        return np.array([], dtype=float)

    if not enable_merge:
        return feats_eligible.ravel() if changed else None

    s_eligible_flat = feats_eligible.ravel()
    groups, _ = compute_angle_adjacent_groups(
        s_eligible_flat,
        theta_lim_deg=theta_lim_deg,
        min_dist=min_dist,
        use_segment_distance=use_segment_distance
    )
    log(f" Gruppen (θ<{theta_lim_deg}°, min_dist={min_dist}, "
        f"use_segment_distance={use_segment_distance}): {groups}", level="info")

    if not groups:
        return s_eligible_flat if changed else None
    grouped_idx = set(j for g in groups for j in g if len(g) >= 2)
    keep_idx = [i for i in range(feats_eligible.shape[0]) if i not in grouped_idx]

    config_glob(s_eligible_flat)
    dirs = [sp.glob.shapes[i].U0 for i in range(feats_eligible.shape[0])]

    out = []
    for i in keep_idx:
        out.extend(feats_eligible[i].tolist())

    for g in groups:
        g2 = [i for i in g if i < feats_eligible.shape[0]]
        if len(g2) < 2:
            continue
        merged = merge_group_touch_chain(s_eligible_flat, g2)
        out.extend(merged)

    out = np.asarray(out, dtype=float)
    return out



def plot_objective_curve(values, title, filename, output_dir):
    """
    Plot and save a scalar objective trajectory.

    Parameters
    ----------
    values : list[float]
    title : str
    filename : str
    output_dir : str
    """
    if not values:
        log(f"Kein Verlauf für {title} gefunden.", level="warning")
        return

    plt.figure(figsize=(6, 4))
    plt.plot(values, marker='o', linewidth=1.5, markersize=3)
    plt.xlabel("Iteration")
    plt.ylabel("Zielfunktion")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    log(f" Zielfunktionsverlauf gespeichert unter: {path}", level="info")


def merge_s_logs(output_dir, subdir_prefix, target_filename):
    """
    Concatenate per-stage 's_log_*.txt' files into a single text file.

    Parameters
    ----------
    output_dir : str
    subdir_prefix : str
        e.g. 'orientation_stage_' or 'convergence_stage_'.
    target_filename : str
    """
    merged_lines = []
    stage_dirs = sorted([d for d in os.listdir(output_dir) if d.startswith(subdir_prefix)])
    for stage_dir in stage_dirs:
        s_log_path = os.path.join(output_dir, stage_dir, f"s_log_stage_{stage_dir.split('_')[-1]}.txt")
        if os.path.exists(s_log_path):
            with open(s_log_path, 'r') as f:
                merged_lines.append(f"# === {stage_dir} ===\n")
                merged_lines.extend(f.readlines())
                merged_lines.append("\n")
    with open(os.path.join(output_dir, f"{target_filename}"), 'w') as f_out:
        f_out.writelines(merged_lines)

def load_density_from_image(image_path, target_shape=None,
                            invert=True, binarize=False, thresh=0.5):
    """
    Load grayscale image as a [0,1] density field.

    Parameters
    ----------
    image_path : str
    target_shape : tuple[int,int] or None
        (ny, nx) to resample with bicubic.
    invert : bool
        If True, black→1, white→0.
    binarize : bool
        If True, apply thresholding at `thresh`.
    thresh : float

    Returns
    -------
    density : np.ndarray, shape (ny, nx)
    """


    img = Image.open(image_path).convert("L")
    if target_shape is not None:
        ny, nx = target_shape
        img = img.resize((nx, ny), Image.BICUBIC)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    density = 1.0 - arr if invert else arr
    if binarize:
        density = (density >= thresh).astype(np.float32)
    return density

def run_configured_optimization(density_path: str, config: dict, output_dir: str):
    """
    End-to-end pipeline:
      1) Orientation run (reward_only=True)
      2) Convergence run (reward_only=False)
      3) Heuristics (optional prune/merge)
      4) Optional greedy additive refinement and final convergence

    Parameters
    ----------
    density_path : str
        Path to target density (.npy or *.density.xml).
    config : dict
        JSON configuration with 'global', 'orientation_run', 'convergence_run',
        'constraints', 'heuristics', 'additive', etc.
    output_dir : str
        Root directory for artifacts.

    Side Effects
    ------------
    - Writes frames, GIFs, logs, objective histories, and merged s-logs.
    """
    log(f"\n Lade Dichtefeld von {density_path}", level="info")
    S_star = parse_density_file(density_path)
    target_n = config.get("global", {}).get("n", [100, 100])
    sp.glob._n = target_n
    sp.glob.dx = 1 / target_n[0]
    sp.glob.dy = 1 / target_n[1]
    sp.glob.order = config.get("global", {}).get("order", 3)

    if list(S_star.shape) != list(reversed(target_n)):
        log(f" Skaliere S_star von {S_star.shape} auf {tuple(reversed(target_n))}", level="info")
        S_star = resize_S_star(S_star, new_shape=tuple(reversed(target_n)))


    s_rand = None
    if config.get("testcase") == "random_feature":
            while True:
                px, py = np.random.uniform(0.1, 0.9, size=2)
                qx, qy = np.random.uniform(0.1, 0.9, size=2)
                if np.linalg.norm([px - qx, py - qy]) >= 0.2:
                    break
            r = np.random.uniform(0.05, 0.25)
            s_rand = np.array([px, py, qx, qy, r])

  
    init_s0 = config.get("initial_s0", None)
    if config.get("testcase") == "O":
        S_star *= 0.0
        H, W = S_star.shape
        S_star[H//4:H//4+5, W//2-2:W//2+3] = 1.0
        S_star[3*H//4-5:3*H//4, W//2-2:W//2+3] = 1.0
        S_star[H//2-2:H//2+3, W//4:W//4+5] = 1.0
        S_star[H//2-2:H//2+3, 3*W//4-5:3*W//4] = 1.0

    elif config.get("testcase") == "P":
        S_star *= 0.0
        H, W = S_star.shape
        S_star[H//2-5:H//2+5, W//4-5:W//4+5] = 1.0
        S_star[H//2-5:H//2+5, 3*W//4-5:3*W//4+5] = 1.0
    elif config.get("testcase") == "N":
        S_star *= 0.0
    elif config.get("testcase") == "extreme":
        S_star *= 0.0
        H, W = S_star.shape
        S_star[H-5:H, 0:5] = 1.0
    elif config.get("testcase") == "extreme_boxextension":
        S_star *= 0.0
        H, W = S_star.shape
        S_star[230:250, 150:170] = 1.0
    elif config.get("testcase") == "random_feature":
        while True:
            px, py = np.random.uniform(0.1, 0.9, size=2)
            qx, qy = np.random.uniform(0.1, 0.9, size=2)
            if np.linalg.norm([px - qx, py - qy]) >= 0.2:
                break
        r = np.random.uniform(0.05, 0.25)
        z_star = np.array([px, py, qx, qy, r])
        bevor = sp.glob.extension
        sp.glob.extension = 0
        S_star = sp.dichte(z_star)
        sp.glob.extension = bevor
    elif config.get("testcase") == "random_feature2":
        while True:
            px, py = np.random.uniform(0.1, 0.9, size=2)
            qx, qy = np.random.uniform(0.1, 0.9, size=2)
            if np.linalg.norm([px - qx, py - qy]) >= 0.2:
                break
        r = np.random.uniform(0.05, 0.25)
        z_star = np.array([px, py, qx, qy, r])
        bevor = sp.glob.extension
        sp.glob.extension = 0
        S_star = sp.dichte(z_star)
        sp.glob.extension = bevor
    elif config.get("testcase") == "smaller_box":
        S_star *= 0.0
        H, W = S_star.shape
        S_star[H//2-5:H//2+5, 0:W//2] = 1.0
    elif config.get("testcase") == "featurecrossing":
        S_star *= 0.0
        H, W = S_star.shape
        S_star[H//2-2:H//2+2, 0:W] = 1.0
        S_star[H//2-12:H//2-8, 0:W] = 1.0
    elif config.get("testcase") == "featurecrossing2":
        S_star *= 0.0
        z_star = np.array([0.01,0.99,0.5,0.01,0.05,0.5,0.2,0.5,0.8,0.1])
        bevor = sp.glob.extension
        sp.glob.extension = 0
        S_star = sp.dichte(z_star)
        sp.glob.extension = bevor
    elif config.get("testcase") == "from_image":
        S_star *= 0
        IMG_PATH = "plots_masterarbeit/zieldichte_from_image.png"
        S_star = load_density_from_image(
            IMG_PATH,
            target_shape=tuple(reversed(target_n)),
            invert=True,                   
            binarize=False                  
        )



    constraints = config.get("constraints", {})
    seg_cfg = dict(constraints.get("segment_length", {}))
    l_max_init = None
    if seg_cfg.get("enabled", False):
        l_max_init = seg_cfg.get("l_max", None)
        if l_max_init is not None:
            l_max_init = float(l_max_init)

    bounds_cfg = config.get("global", {}).get("bounds", {"x":[0,1],"y":[0,1]})
    bounds = ((bounds_cfg["x"][0], bounds_cfg["x"][1]),
              (bounds_cfg["y"][0], bounds_cfg["y"][1]))

    if init_s0 is not None:
        s0 = np.array(init_s0, dtype=float)
        if s0.ndim != 1 or (s0.size % 5) != 0:
            raise ValueError("initial_s0 muss 1D sein und eine Länge haben, die durch 5 teilbar ist.")
    else:
        if config.get("num_features", 0) > 0:
            log(f"\n Initialisiere Features mittels Strategie '{config['start_strategy']}'", level="info")
            s0 = initialize_features(
                config["start_strategy"],
                config["num_features"],
                config.get("start_radius", 0.1),
                config.get("random_seed", None),
                max_seg_len=l_max_init,   
                bounds=bounds              
            )
            s0 = np.array(s0, dtype=float)
        else:
            s0 = np.array([], dtype=float)

    if s_rand is not None:
        s0=s_rand
    history = []

    log("\n Starte Orientation Run (reward_only=True)", level="info")
    for stage_id, stage_config in enumerate(config.get("orientation_run", [])):
        if stage_config.get("max_iter", 0) <= 0:
            log(f" Überspringe Orientation-Stage {stage_id} (max_iter=0)", level="info")
            continue
        merged_config = {
            **config.get("global", {}),
            **stage_config,
            "constraints": config.get("constraints", {}),
            "reward_only": True
        }

        stage_output_dir = os.path.join(output_dir, f"orientation_stage_{stage_id}")
        s0, stage_history = run_optimization_stage(s0, S_star, merged_config, stage_id=stage_id, out_dir=stage_output_dir)
        s0 = np.array(s0)
        history.extend(stage_history)
    combine = "Combine nach orientation run "+str(sp.glob.combine)
    log(combine, level="info")
    log("\nStarte Convergence Run (reward_only=False)", level="info")
    for stage_id, stage_config in enumerate(config.get("convergence_run", [])):
        if stage_config.get("max_iter", 0) <= 0:
            log(f"Überspringe Convergence-Stage {stage_id} (max_iter=0)", level="info")
            continue
        merged_config = {
            **config.get("global", {}),
            **stage_config,
            "constraints": config.get("constraints", {}),
            "reward_only": False
        }

        stage_output_dir = os.path.join(output_dir, f"convergence_stage_{stage_id}")
        s0, stage_history = run_optimization_stage(s0, S_star, merged_config, stage_id=stage_id, out_dir=stage_output_dir)
        s0 = np.array(s0)
        history.extend(stage_history)
    hcfg = config.get("heuristics", {})
    if s0 is not None and np.size(s0) > 0 and hcfg.get("enable", False):
        heu_dir = os.path.join(output_dir, "heuristics")
        os.makedirs(heu_dir, exist_ok=True)

        heuristics_enabled = (hcfg != {}) and (hcfg.get("enable_merge", True) or True)

        if heuristics_enabled:
            log("\n🛠️ Wende Heuristiken an", level="info")
            s0_new = apply_heuristics(
                s0,
                enable_merge=hcfg.get("enable_merge", True),
                theta_lim_deg=hcfg.get("theta_lim_deg", 10.0),
                min_dist=hcfg.get("min_dist", 0.15),
                use_segment_distance=hcfg.get("use_segment_distance", False),
                ARlim=hcfg.get("ARmin", 0.05),
                URlim=hcfg.get("URmin", 0.10),
                simple_cfg=hcfg.get("simple", None),
            )
        else:
            s0_new = None


        if s0_new is not None:
            np.savetxt(os.path.join(heu_dir, "s_after_heuristics.txt"), np.array(s0_new).ravel())
            log("\nConvergence Run nach Heuristik", level="info")
            final_convergence_stage = config.get("convergence_run", [])[-1]
            merged_config = {
                **config.get("global", {}),
                **final_convergence_stage,
                "constraints": config.get("constraints", {}),
                "reward_only": False
            }
            stage_output_dir = os.path.join(output_dir, f"convergence_after_heuristik_stage")
            s0, stage_history = run_optimization_stage(np.array(s0_new), S_star, merged_config,
                                                        stage_id="after_heuristics", out_dir=stage_output_dir)
            s0 = np.array(s0)
            history.extend(stage_history)
        else:
            np.savetxt(os.path.join(heu_dir, "s_nochange_kept.txt"), np.array(s0).ravel())
            with open(os.path.join(heu_dir, "README.txt"), "w") as f:
                f.write("Heuristics liefen, aber es gab keine Änderungen.\n")
            log("Keine Änderungen durch Heuristiken", level="info")

    add_cfg = config.get("additive", {})
    if add_cfg.get("enable", False):
        log("\nStarte greedy Additive-Refinement (ein Feature pro Zyklus)", level="info")
        add_dir = os.path.join(output_dir, "additive_refinement")
        os.makedirs(add_dir, exist_ok=True)
        s_after_add = greedy_additive_refinement(s0, S_star, config, output_dir=add_dir)

        if s_after_add is not None and len(s_after_add) != len(s0):
            log(f"Additive Refinement hat Features erhöht: {len(s0)//5} → {len(s_after_add)//5}", level="info")
            s0 = np.array(s_after_add, float)
            if config.get("convergence_run", []):
                final_conv = config["convergence_run"][-1]
                merged_config = {
                    **config.get("global", {}),
                    **final_conv,
                    "constraints": config.get("constraints", {}),
                    "reward_only": False
                }
                add_final_dir = os.path.join(add_dir, "final_convergence_after_additive")
                s0, stage_history = run_optimization_stage(s0, S_star, merged_config,
                                                          stage_id="final_after_additive", out_dir=add_final_dir)
                history.extend(stage_history)

        log("Optimierung abgeschlossen.", level="info")

    history_mse = []
    history_reward = []
    history_reward2 = []

    for stage in history:
        obj_values = stage.get("objective_values", [])
        config = stage.get("config", {})
        if config.get("reward_only2", False):
            history_reward2.extend(obj_values)
        elif config.get("reward_only", False):
            history_reward.extend(obj_values)
        else:
            history_mse.extend(obj_values)
    
    merge_s_logs(output_dir, "orientation_stage_", "s_log_orientation_all.txt")
    merge_s_logs(output_dir, "convergence_stage_", "s_log_convergence_all.txt")

    plot_objective_curve(history_mse, "Zielfunktion (MSE)", "objective_mse.png", output_dir)
    plot_objective_curve(history_reward, "Zielfunktion (Reward Only)", "objective_reward.png", output_dir)
    #plot_objective_curve(history_reward2, "Zielfunktion (Reward Only 2)", "objective_reward2.png", output_dir)






def parse_args():
    """
    CLI parser for the main entrypoint.

    Returns
    -------
    argparse.Namespace
        {density_path, config_path, output_dir}
    """
    parser = argparse.ArgumentParser(description="Starte Feature-basierte Optimierung.")
    parser.add_argument("density_path", type=str, help="Pfad zur Zieldichte-Datei (z. B. out_2bar.cfs)")
    parser.add_argument("config_path", type=str, help="Pfad zur Konfigurationsdatei (z. B. config_5bar_cross.json)")
    parser.add_argument("--output_dir", type=str, default="results", help="Verzeichnis für Logs, Plots und Ergebnisse")
    return parser.parse_args()

def main():
    """
    CLI entrypoint: load JSON config and run the configured optimization.
    """
    args = parse_args()
    config = load_config(args.config_path)
    run_configured_optimization(args.density_path, config, output_dir=args.output_dir)

def run_cfs(mesh_file, config_file, output_file):
    """
    Convenience wrapper to run an external 'cfs' executable with mesh/config.

    Parameters
    ----------
    mesh_file : str
    config_file : str
    output_file : str
    """
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mesh_path = os.path.join(project_dir, "mesh", mesh_file)
    config_path = os.path.join(project_dir, "config", config_file)
    command = ["cfs", "-m", mesh_path, "-p", config_path, output_file]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    print("Starte Optimierung mit CLI-Argumenten...")
    main()