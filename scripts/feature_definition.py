# Imports / NUMBA flag
"""
External dependencies and JIT availability.

Notes
-----
- `NUMBA_AVAILABLE` gates the use of the parallel kernel `_numba_boundary_grad`.
- If Numba is not available, code paths fall back to pure-NumPy pointwise evaluation.
"""


import numpy as np
from numpy.linalg import norm
from math import comb
from scipy.special import betainc, beta
from bezier_lut import lut_manager
from bezier_math import find_gamma
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

"""
Feature field infrastructure for topology optimization.

This module defines:
  - Global: global discretization, grid metrics, and per-feature parameters.
  - NUMBA-accelerated kernel for boundary-gradient accumulation on transition cells.
  - Utility routines for smoothstep coefficients and Beta-based transition shaping.

All quantities are defined over a structured cell grid (nx, ny) on bounds
`glob.bounds`. Features (e.g., Pill) rasterize density ρ and its sensitivities
onto this grid and query per-feature parameters via `glob`.
"""


if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True)
    def _numba_boundary_grad(
        Xb, Yb,                    # (M,)
        Px, Py, Qx, Qy,           # floats
        U0x, U0y, V0x, V0y,       # floats
        p,                        # float
        grad_u0vert,              # (5,2)
        grad_Q,                   # (5,2)
        grad_P,                   # (5,2)
        h, ext,                   # floats
        method_id,                # 0=smoothstep, 1=sigmoid, 2=smoothstep_shift, 3=beta_mid, 4=Bezier_lut
        k,                        # float (sigmoid)
        N,                        # int   (smoothstep / shift)
        cs,                       # (N+1,) smoothstep coeffs
        alpha,                    # float (beta_mid, hier 3.0)
        beta_param,                # float (beta_mid)
        lut_x,                    # ndarray, shape (num_pts,) 
        lut_dy                    # ndarray, shape (num_pts,)
    ):
        """
        Vectorized per-point boundary gradient G = (dB/dφ) * (∂φ/∂s) on transition cells.

        Parameters
        ----------
        Xb, Yb : ndarray, shape (M,)
            Coordinates of transition-cell centers to evaluate.
        Px, Py, Qx, Qy : float
            Capsule endpoints.
        U0x, U0y, V0x, V0y : float
            Unit tangent U0 and unit normal V0 components of the segment.
        p : float
            Half-width (radius) of the capsule.
        grad_u0vert, grad_Q, grad_P : ndarray
            Precomputed geometric sensitivity stencils, shapes (5,2), (5,2), (5,2).
        h, ext : float
            Transition half-width and extension defining the smoothing band.
        method_id : int
            Transition mapping: 0=smoothstep, 1=sigmoid, 2=smoothstep_shift, 3=beta_mid.
        k : float
            Sigmoid sharpness (only for method_id==1).
        N : int
            Smoothstep order (only for method_id in {0,2}).
        cs : ndarray, shape (N+1,)
            Cached smoothstep coefficients.
        alpha : float
            Beta CDF alpha parameter (beta_mid).
        beta_param : float
            Beta CDF beta parameter (beta_mid).
        lut_x, lut_dy : ndarray, shape (num_pts,) (only for method_id==4)

        Returns
        -------
        G : ndarray, shape (M, 5)
            ∂ρ/∂s at each queried cell for the active branch (segment/P-cap/Q-cap).

        Notes
        -----
        - Chooses the active distance branch per point, computes ∂φ/∂s, and applies
          the method-specific derivative dB/dφ scaled to [rhomin, rhomax] by callers.
        - Parallel over points (Numba prange), assumes double precision inputs.
        """
        M = Xb.shape[0]
        G = np.zeros((M, 5), dtype=np.float64)
        for m in prange(M):
            x = Xb[m]; y = Yb[m]
            dot_Q = (x - Qx)*U0x + (y - Qy)*U0y
            dot_P = (x - Px)*U0x + (y - Py)*U0y
            on_seg = (dot_Q <= 0.0) and (dot_P >= 0.0)
            d_seg_signed = (x - Qx)*V0x + (y - Qy)*V0y
            phi_seg = abs(d_seg_signed) - p
            dxP = x - Px; dyP = y - Py
            rP  = np.sqrt(dxP*dxP + dyP*dyP)
            phi_P = rP - p
            dxQ = x - Qx; dyQ = y - Qy
            rQ  = np.sqrt(dxQ*dxQ + dyQ*dyQ)
            phi_Q = rQ - p
            phi = phi_P
            case = 0 
            if phi_Q < phi:
                phi = phi_Q; case = 2
            if on_seg and (phi_seg < phi):
                phi = phi_seg; case = 1
            dphi = np.zeros(5, dtype=np.float64)
            dphi[4] = -1.0

            if case == 1:
                sgn = 1.0
                proj = (x - Qx)*V0x + (y - Qy)*V0y
                if proj < 0: sgn = -1.0
                for var in range(4):
                    t1 = (x - Qx)*grad_u0vert[var,0] + (y - Qy)*grad_u0vert[var,1]
                    t2 = grad_Q[var,0]*V0x + grad_Q[var,1]*V0y
                    dphi[var] = sgn * (t1 - t2)
            elif case == 0:
                if rP > 1e-15:
                    dphi[0] = (Px - x) / rP
                    dphi[1] = (Py - y) / rP
                else:
                    dphi[0] = 1.0; dphi[1] = 0.0

            else:
                if rQ > 1e-15:
                    dphi[2] = (Qx - x) / rQ
                    dphi[3] = (Qy - y) / rQ
                else:
                    dphi[2] = 1.0; dphi[3] = 0.0

            dB_dphi_scaled = 0.0
            if (-h < phi) and (phi < h + ext):
                if method_id == 0:
                    if phi <= 0.0:
                        t = (-phi + h) / (2.0*h)
                        scaling = 2.0*h
                    else:
                        t = (-phi + h + ext) / (2.0*(h + ext))
                        scaling = 2.0*(h + ext)
                    dB_dt = 0.0
                    for n in range(N+1):
                        power = N + n
                        dB_dt += cs[n] * (power + 1) * (t ** power)
                    dB_dphi_scaled = - dB_dt / scaling

                elif method_id == 1:
                    if phi <= 0.0:
                        sc = h
                    else:
                        sc = (h + ext)
                    t_sig = (k * phi) / sc
                    tanhk = np.tanh(k)
                    sech2 = 1.0 / (np.cosh(t_sig)**2)
                    dB_dphi_scaled = -0.5 * (k / (sc * tanhk)) * sech2

                elif method_id == 2:
                    W = 2.0*h + ext
                    t = (h + ext - phi) / W
                    dB_dt = 0.0
                    for n in range(N+1):
                        power = N + n
                        dB_dt += cs[n] * (power + 1) * (t ** power)
                    dB_dphi_scaled = - dB_dt / W
                elif method_id == 3:
                    H_all = 2.0*h + ext
                    u = (phi + h) / H_all
                    if u < 1e-12:
                        u = 1e-12
                    elif u > 1.0 - 1e-12:
                        u = 1.0 - 1e-12
                    invB = (beta_param * (beta_param + 1.0) * (beta_param + 2.0)) * 0.5
                    pdf_u = invB * (u*u) * ( (1.0 - u) ** (beta_param - 1.0) )
                    dB_dphi_scaled = - pdf_u / H_all
                elif method_id == 4: 
                    dB_dphi_scaled = np.interp(phi, lut_x, lut_dy)           
            for v in range(5):
                G[m, v] = dB_dphi_scaled * dphi[v]

        return G



class Global:
  """
  Process-wide configuration, grid geometry, and feature parameters.

  Purpose
  -------
  Central store for:
    - domain bounds and grid (nx, ny),
    - global per-feature parameters (boundary method, transition width, etc.),
    - cached grid coordinates (nodes, cell origins, centers),
    - book-keeping of registered feature shapes.

  Notes
  -----
  Feature instances read parameters via `glob.get_feature_param(name, feature_id)`.
  Grid-dependent metrics are lazily (re)computed and cached via `_ensure_grids()`.
  """
  
  def __init__(self):
    """
    Initialize defaults and compute initial grid metrics.

    Sets
    ----
    bounds : dict
        {"x":[0,1], "y":[0,1]} by default.
    n : list[int]
        [nx, ny] = [100, 100] by default.
    boundary : str
        Transition law ('smoothstep' | 'sigmoid' | 'beta_mid' | 'smoothstep_shift').
    transition, extension : float
        Half-width and extra extension of the smoothing band.
    rhomin, rhomax : float
        Density range.
    Other caches/fields are set to None and filled on demand.
    """
    self.shapes = []  
    self._rhomin = 1e-30  
    self._rhomax = 1 
    self._boundary = 'smoothstep'  
    self._transition = 0.1 
    self._n = [100, 100] 
    self.opts = {}  
    self.dx = 1 / self.n[0]
    self.dy = 1/self.n[1]
    self.p = 3
    self.combine = 'p-norm'
    self.idx_field = None  
    self.dist_field = None  
    self.grad_field = None
    self.info_field = {} 
    self._order = 2
    self._k = 7
    self._N = 3
    self.beta = 7
    self._extension = 0.3
    self._bounds = {"x": [0,1], "y": [0,1]}
    self.log_level = None
    self._nodes_x = None
    self._nodes_y = None
    self._cell_x0 = None   
    self._cell_y0 = None
    self._cell_cx = None 
    self._cell_cy = None
    self._grid_sig = None   
    self.sum_cap = 1.1 
    self.extension_kind = "scaled_flank"
    self.update_grid_metrics()

    # Bezier Parameters
    self._bezier_order = 5 
    self._gamma = find_gamma(self.bezier_order, self.transition / 2.0, (self.transition / 2.0) + self.extension)

  @property
  def bezier_order(self):
    return self._bezier_order
  @bezier_order.setter
  def bezier_order(self, v):
    self._bezier_order = int(v); self.reset_global_fields()

  @property
  def gamma(self):
    return self._gamma
  @gamma.setter
  def gamma(self, v):
    self._gamma = float(v); self.reset_global_fields()

  @property
  def bounds(self):
      """
      Domain bounds.

      Returns
      -------
      dict
          {"x":[xmin, xmax], "y":[ymin, ymax]} as floats.
      """
      return self._bounds

  @bounds.setter
  def bounds(self, val):
      """
      Set domain bounds and refresh grid metrics/caches.

      Parameters
      ----------
      val : dict
          Keys 'x' and 'y' each mapping to [min, max] (floats).

      Effect
      ------
      Recomputes dx, dy and invalidates grid-dependent caches/fields.
      """

      self._bounds = {
          "x": [float(val["x"][0]), float(val["x"][1])],
          "y": [float(val["y"][0]), float(val["y"][1])]
      }
      self.update_grid_metrics()
      self.reset_global_fields()

  def reset_all_shape_caches(self):
      """
      Invalidate all cached global fields and per-shape caches.

      Effect
      ------
      Clears `idx_field`, `dist_field`, `grad_field` and calls each shape's
      `_invalidate_fields()` / `invalidate_cache()` if present.
      """
      self.idx_field = None
      self.dist_field = None
      self.grad_field = None
      for s in self.shapes:
        if hasattr(s, "_invalidate_fields"):
            s._invalidate_fields()
        elif hasattr(s, "invalidate_cache"):
            s.invalidate_cache()

  @property
  def order(self):
      """Polynomial order for downstream FE or filters (int)."""
      return self._order

  @order.setter
  def order(self, val):
      """
      Set polynomial order and clear global fields.

      Parameters
      ----------
      val : int
          Desired order (will be cast to int).
      """
      self._order = int(val)
      self.reset_global_fields()

  @property
  def transition(self):
      """Smoothing band half-width h (float)."""
      return self._transition

  @transition.setter
  def transition(self, val):
      """
      Set smoothing half-width and clear global fields.

      Parameters
      ----------
      val : float
      """
      self._transition = float(val)
      self._gamma = find_gamma(self.bezier_order, self._transition / 2.0, (self._transition / 2.0) + self._extension)
      self.reset_global_fields()

  @property
  def n(self):
      """Grid resolution [nx, ny] (list[int])."""
      return self._n

  @n.setter
  def n(self, value):
      """
      Set grid resolution and refresh metrics/caches.

      Parameters
      ----------
      value : iterable[int]
          (nx, ny). Recomputes dx, dy and invalidates global fields.
      """
      nx, ny = int(value[0]), int(value[1])
      self._n = [nx, ny]
      self.update_grid_metrics()
      self.reset_global_fields() 

  def update_grid_metrics(self):
      """
      Update dx, dy from bounds and resolution.

      Notes
      -----
      Called on changes of `bounds` or `n`; resets `_grid_sig` to force
      re-materialization of coordinate arrays.
      """
      x_min, x_max = self.bounds["x"]
      y_min, y_max = self.bounds["y"]

      self.dx = (x_max - x_min) / self._n[0]
      self.dy = (y_max - y_min) / self._n[1]
      self._grid_sig = None

  def get_feature_param(self, name, feature_id):
        """
        Fetch per-feature parameter, supporting scalar-or-array semantics.

        Parameters
        ----------
        name : str
            Attribute name on `Global` (e.g., 'transition', 'boundary').
        feature_id : int
            Index of the feature.

        Returns
        -------
        Any
            If the attribute is array-like, returns val[feature_id]; else returns val.
        """
        val = getattr(self, name)
        if isinstance(val, (list, np.ndarray)):
            return val[feature_id]
        return val

  def _grid_signature(self):
      """
      Immutable signature of the current grid geometry.

      Returns
      -------
      tuple
          (bounds.x, bounds.y, nx, ny) as hashable tuple for cache checks.
      """
      return (
          tuple(self.bounds["x"]), tuple(self.bounds["y"]),
          int(self._n[0]), int(self._n[1])
      )

  def _ensure_grids(self):
      """
      Lazily (re)build coordinate arrays for nodes and cell centers.

      Side Effects
      ------------
      _nodes_x, _nodes_y : ndarray, shape (nx+1,), (ny+1,)
      _cell_x0, _cell_y0 : ndarray, shape (nx,), (ny,)   (cell origins)
      _cell_cx, _cell_cy : ndarray, shape (nx,), (ny,)   (cell centers)
      _grid_sig : tuple
          Updated to the current `_grid_signature()`.
      """
      sig = self._grid_signature()
      if self._grid_sig == sig:
          return
      x_min, x_max = self.bounds["x"]
      y_min, y_max = self.bounds["y"]
      nx, ny = self._n
      self.dx = (x_max - x_min) / nx
      self.dy = (y_max - y_min) / ny

      self._nodes_x = np.linspace(x_min, x_max, nx + 1)
      self._nodes_y = np.linspace(y_min, y_max, ny + 1)

      self._cell_x0 = x_min + np.arange(nx) * self.dx
      self._cell_y0 = y_min + np.arange(ny) * self.dy

      self._cell_cx = self._cell_x0 + 0.5 * self.dx
      self._cell_cy = self._cell_y0 + 0.5 * self.dy

      self._grid_sig = sig

  def cell_origins(self):
      """
      Cell origin coordinates (lower-left corner per cell).

      Returns
      -------
      (x0, y0) : tuple[ndarray, ndarray]
          1D arrays, shapes (nx,), (ny,).
      """
      self._ensure_grids()
      return self._cell_x0, self._cell_y0  

  def cell_centers(self):
      """
      Cell center coordinates.

      Returns
      -------
      (cx, cy) : tuple[ndarray, ndarray]
          1D arrays, shapes (nx,), (ny,).
      """
      self._ensure_grids()
      return self._cell_cx, self._cell_cy

  def nodes(self):
      """
      Grid node coordinates on the structured mesh.

      Returns
      -------
      (xn, yn) : tuple[ndarray, ndarray]
          1D arrays, shapes (nx+1,), (ny+1,).
      """
      self._ensure_grids()
      return self._nodes_x, self._nodes_y
      
  def total(self):
    """
    Total number of optimization variables across all shapes.

    Returns
    -------
    int
        Sum over len(s.optvar()) for each registered shape.
    """
    return sum(len(s.optvar()) for s in self.shapes)  

  def var_all(self):
    """
    Flattened list of all optimization variables across shapes.

    Returns
    -------
    list[float]
        Concatenated [Px, Py, Qx, Qy, p, ...] for all shapes.
    """
    vars = []
    for i in range(len(self.shapes)):
        vars.extend(self.shapes[i].optvar())
    return vars

  @property
  def boundary(self):
    """Boundary mapping method as string."""
    return self._boundary
  @boundary.setter
  def boundary(self, v):
      """
      Set boundary mapping method and clear global fields.

      Parameters
      ----------
      v : str
          'smoothstep' | 'sigmoid' | 'beta_mid' | 'smoothstep_shift' | 'bezier'.
      """
      self._boundary = str(v)
      self.reset_global_fields()

  @property
  def k(self):
     """Sigmoid sharpness parameter k (float)."""
     return self._k
  @k.setter
  def k(self, v):
      """Set k and clear global fields."""
      self._k = float(v); self.reset_global_fields()

  @property
  def N(self):
    """Smoothstep order N (int)."""  
    return self._N
  @N.setter
  def N(self, v):
      """Set N and clear global fields."""
      self._N = int(v); self.reset_global_fields()

  @property
  def extension(self):
    """Transition band extension (float)."""
    return self._extension
  @extension.setter
  def extension(self, v):
      """Set extension and clear global fields."""
      self._extension = float(v);
      self._gamma = find_gamma(self.bezier_order, self._transition / 2.0, (self._transition / 2.0) + self._extension)
      self.reset_global_fields()

  @property
  def rhomin(self):
    """Lower density bound (float)."""
    return self._rhomin
  @rhomin.setter
  def rhomin(self, v):
      """Set rhomin and clear global fields."""
      self._rhomin = float(v); self.reset_global_fields()

  @property
  def rhomax(self):
    """Upper density bound (float)."""
    return self._rhomax
  @rhomax.setter
  def rhomax(self, v):
      """Set rhomax and clear global fields."""
      self._rhomax = float(v); self.reset_global_fields()

  def reset_global_fields(self):
    """
    Invalidate global raster fields and grid signature.

    Effect
    ------
    Sets `idx_field`, `dist_field`, `grad_field` to None and clears `_grid_sig`
    to force `_ensure_grids()` on next access.
    """
    self.idx_field = None
    self.dist_field = None
    self.grad_field = None
    self._grid_sig = None 

  def add_shape(self, shape):
    """
    Register a new feature shape.

    Parameters
    ----------
    shape : object
        Must implement `optvar()` and (optionally) `_invalidate_fields()`.
    """
    self.shapes.append(shape)

  def remove_shape(self, shape_or_id):
      """
      Remove a shape by instance or by id.

      Parameters
      ----------
      shape_or_id : object | int
          Feature instance with `.id` or integer id.
      """
      if hasattr(shape_or_id, "id"):
          self.shapes = [s for s in self.shapes if s is not shape_or_id]
      else:
          self.shapes = [s for s in self.shapes if s.id != shape_or_id]

  def clear_shapes(self):
      """
      Remove all registered shapes.
      """
      self.shapes = []

glob = []
glob = Global()

def smoothstep_coeffs(N: int) -> np.ndarray:
    """
    Closed-form coefficients for (2N+1)-order smoothstep polynomial.

    Parameters
    ----------
    N : int
        Smoothstep order.

    Returns
    -------
    cs : ndarray, shape (N+1,)
        Coefficients such that B(t) = sum_{n=0..N} cs[n] * t^{N+n+1}.
    """
    cs = np.empty(N+1, dtype=np.float64)
    for n in range(N+1):
        cs[n] = ((-1.0)**n) * comb(N + n, n) * comb(2*N + 1, N - n)
    return cs

def _find_beta_for_mid(alpha, u0, tol=1e-10):
    """
    Solve betainc(alpha, beta, u0) = 0.5 for beta via bisection with interval growth.

    Parameters
    ----------
    alpha : float
        Fixed alpha parameter (> 2 for our usage).
    u0 : float
        Target normalized position in [0,1].
    tol : float, default 1e-10
        Absolute function tolerance for bisection termination.

    Returns
    -------
    beta_param : float
        Beta parameter such that I_{u0}(alpha, beta_param) ≈ 0.5.

    Notes
    -----
    Expands the search bracket adaptively to ensure inclusion before bisecting.
    """
    def F(b):
        return betainc(alpha, b, u0) - 0.5

    b_lo, b_hi = 2.1, 2.1
    f_lo = F(b_lo)
    f_hi = F(b_hi)
    grow = 1.6
    it = 0
    while f_lo * f_hi > 0 and it < 50:
        b_hi *= grow
        f_hi = F(b_hi)
        it += 1
    it2 = 0
    while f_lo * f_hi > 0 and it2 < 50 and b_lo > 2.0001:
        b_lo = 2.0001 + (b_lo - 2.0001)/grow
        f_lo = F(b_lo)
        it2 += 1
    for _ in range(100):
        b_mid = 0.5*(b_lo + b_hi)
        f_mid = F(b_mid)
        if abs(f_mid) < tol: return b_mid
        if f_lo * f_mid <= 0:
            b_hi, f_hi = b_mid, f_mid
        else:
            b_lo, f_lo = b_mid, f_mid
        if abs(b_hi - b_lo) < 1e-10:
            break
    return 0.5*(b_lo + b_hi)

def _beta_pdf(u, a, b):
    """
    Beta(a,b) PDF at u with clipping for numerical stability.

    Parameters
    ----------
    u : float or ndarray
        Evaluation point(s), clipped to [1e-14, 1-1e-14].
    a, b : float
        Beta distribution parameters.

    Returns
    -------
    pdf : float or ndarray
        (u^{a-1} (1-u)^{b-1}) / Beta(a,b).
    """
    u = np.clip(u, 1e-14, 1-1e-14)
    return (u**(a-1) * (1-u)**(b-1)) / beta(a, b)

def _beta_pdf_prime(u, a, b):
    """
    Derivative of the Beta(a,b) PDF w.r.t. u.

    Parameters
    ----------
    u : float or ndarray
        Evaluation point(s), clipped to [1e-12, 1-1e-12].
    a, b : float
        Beta distribution parameters.

    Returns
    -------
    d_pdf_du : float or ndarray
        d/du [ BetaPDF(u; a, b) ] = PDF(u)*( (a-1)/u - (b-1)/(1-u) ).
    """
    u = np.clip(u, 1e-12, 1-1e-12)
    pdf = _beta_pdf(u, a, b)
    return pdf * ((a-1)/u - (b-1)/(1-u))



"""
Capsule-shaped feature primitive used in topology optimization.

A Pill represents a 2D capsule (line segment from P to Q with circular end caps and
half-width p). It provides:
  - a signed distance/offset function φ(x,y) to the capsule boundary,
  - a smooth boundary-to-density mapping ρ(φ) with configurable transition laws,
  - first/second derivatives of ρ w.r.t. the feature optimization variables s = [px, py, qx, qy, p].

The class rasterizes these quantities on the global cell grid defined in `glob`
and caches per-cell fields (density, gradient, Hessian) for efficient sampling
and FE assembly.
"""


class Pill: 
  def __init__(self, id, P, Q, p):
    """
    Initialize a capsule feature.

    Parameters
    ----------
    id : int
        Unique feature identifier (used to query per-feature parameters in `glob`).
    P : array_like, shape (2,)
        First endpoint (Px, Py) of the centerline segment.
    Q : array_like, shape (2,)
        Second endpoint (Qx, Qy) of the centerline segment.
    p : float
        Half-width (radius) of the capsule.

    Notes
    -----
    Initializes sensitivity stencils w.r.t. s = [px, py, qx, qy, p] and
    computes unit tangential/normal vectors (U0, V0). Any previously
    cached per-cell fields are invalidated.
    """
    self.id = id
    self.namedidx = {'px': 0, 'py': 1, 'qx': 2, 'qy': 3, 'p':4}
    self.idx_field = None 
    self.rho_field = None  
    self.grad_field = None
    self.hessian_field = None   
    self._cache_signature = None
    self._has_grad = False
    self._has_hess = False
    self.set(P, Q, p)
  

  def _invalidate_fields(self):
    """
    Drop all cached raster fields and derivative flags.

    Effect
    ------
    Sets `idx_field`, `rho_field`, `grad_field`, `hessian_field` to None and
    resets `_has_grad`, `_has_hess`, `_cache_signature`. Forces re-computation
    on next `ensure_fields(...)` or `sample(...)`.
    """
    self.idx_field = None
    self.rho_field = None
    self.grad_field = None
    self.hessian_field = None
    self._cache_signature = None
    self._has_grad = False
    self._has_hess = False

  def set(self, P, Q, p, reset_fields=True):
    """
    Update capsule geometry and rebuild local sensitivity stencils.

    Parameters
    ----------
    P, Q : array_like, shape (2,)
        New endpoints for the centerline segment.
    p : float
        New half-width (radius).
    reset_fields : bool, default True
        If True, also clears global caches in `glob` (idx/dist/grad fields)
        and invalidates this feature's cached fields.

    Notes
    -----
    Recomputes U, U0, V0 and analytic first/second-order sensitivity tensors:
      - sens['grad_u'], sens['grad_norm_u'], sens['grad_u0vert'],
        sens['grad_P'], sens['grad_Q'], sens['hess_u0'].
    These stencils are used by distance and boundary mapping derivatives.
    """
    if reset_fields:
      glob.idx_field = None
      glob.dist_field = None
      glob.grad_field = None
  
    self.num_optvar = 5
    
    self.sens = {
      'grad_u': np.zeros((self.num_optvar, 2)),
      'grad_norm_u': np.zeros((self.num_optvar)),
      'grad_u0vert': np.zeros((self.num_optvar, 2)),
      'grad_P': np.zeros((self.num_optvar, 2)),
      'grad_Q': np.zeros((self.num_optvar, 2)),
      'hess_proj': np.zeros((self.num_optvar, self.num_optvar))
    }
    assert len(P) == 2 and len(Q) == 2  
    self.P = np.array(P) 
    self.Q = np.array(Q) 
    self.p = p 
    self.transition = glob.transition
    self.U = self.Q - self.P
    norm_U = norm(self.U)
    if norm_U < 1e-20:
      self.U[:] = [1e-20, 0]
      norm_U = norm(self.U)
    self.U0 = self.U / norm_U
    self.sens['grad_u'][0:5, :] = [
      [-1, 0],
      [0, -1],
      [1, 0],
      [0, 1],
      [0,0]
    ]
    
    self.sens['grad_norm_u'][0:5] = [
      -self.U[0] / norm_U, 
      -self.U[1] / norm_U, 
      self.U[0] / norm_U, 
      self.U[1] / norm_U,
      0
    ]
    self.V = np.array([-self.U[1], self.U[0]])
    self.V0 = self.V / norm(self.V)
    nU3 = norm_U ** 3
    self.sens['grad_u0vert'][0:5, :] = [
      [-self.U[0] * self.U[1] / nU3, -self.U[1] ** 2 / nU3],
      [self.U[0] ** 2 / nU3, self.U[0] * self.U[1] / nU3],
      [self.U[0] * self.U[1] / nU3, self.U[1] ** 2 / nU3],
      [-self.U[0] ** 2 / nU3, -self.U[0] * self.U[1] / nU3],
      [0,0]
    ]
    self.sens['grad_Q'][0:5, :] = [
      [0, 0],
      [0, 0],
      [1, 0],
      [0, 1],
      [0, 0]
    ]
    self.sens['grad_P'][0:5, :] = [
      [1, 0],
      [0, 1],
      [0, 0],
      [0, 0],
      [0, 0]
    ]
    self.sens['hess_u0'] = np.zeros((self.num_optvar, self.num_optvar, 2))

    for i in range(self.num_optvar):
        for j in range(self.num_optvar):
            dU_i = self.sens['grad_Q'][i] - self.sens['grad_P'][i]
            dU_j = self.sens['grad_Q'][j] - self.sens['grad_P'][j]
            U_dot_dU_i = np.dot(self.U, dU_i)
            U_dot_dU_j = np.dot(self.U, dU_j)
            dU_i_dot_dU_j = np.dot(dU_i, dU_j)
            term1 = -(U_dot_dU_j * dU_i + U_dot_dU_i * dU_j + dU_i_dot_dU_j * self.U) / (norm_U ** 3)
            self.sens['hess_u0'][i, j] = term1
    self.segs = [P, Q]
    if reset_fields:
      self._invalidate_fields() 

  def optvar(self):
    """
    Return current optimization variables.

    Returns
    -------
    list[float]
        [Px, Py, Qx, Qy, p] in that order.
    """
    return [self.P[0], self.P[1], self.Q[0], self.Q[1], self.p]

  def optvar_names(self):
    """
    Names/order of optimization variables.

    Returns
    -------
    list[str]
        ['px', 'py', 'qx', 'qy', 'p'].
    """
    return ['px', 'py', 'qx', 'qy', 'p']

  def compute_hessian(self, x ,y, sign):
      """
      Local second derivatives of the projected-distance branch.

      Parameters
      ----------
      x, y : float
          Query point coordinates.
      sign : float
          Sign of the signed distance along the normal branch (+1 or −1).
          Used to orient the Hessian contribution.

      Returns
      -------
      H : ndarray, shape (5, 5)
          Hessian ∂²φ/∂s_m∂s_n of the *segment-projection* distance branch
          w.r.t. s = [px, py, qx, qy, p]. The p-row/column is zero here.
          The matrix is scaled by `sign`.

      Notes
      -----
      This routine is only valid if the closest point to (x,y) lies on the
      interior of the segment (not on the end caps).
      """
      px = self.P[0]
      py = self.P[1]
      qx = self.Q[0]
      qy = self.Q[1]

      partials = {
                    "px_px": (py - qy) * (
                        ((px - qx)**2 + (py - qy)**2) * (qx - x)
                        + 3 * (px - qx) * (qx**2 - qx * x + px * (-qx + x) - (py - qy) * (qy - y))
                    ) / ((px - qx)**2 + (py - qy)**2)**2.5,

                    "px_py": (
                        -(((px - qx)**2 + (py - qy)**2) * (qx**2 - qx * x + px * (-qx + x) - (py - qy) * (qy - y)))
                        + 3 * (py - qy)**2 * (qx**2 - qx * x + px * (-qx + x) - (py - qy) * (qy - y))
                        + ((px - qx)**2 + (py - qy)**2) * (py - qy) * (qy - y)
                    ) / ((px - qx)**2 + (py - qy)**2)**2.5,

                    "px_qx": (py - qy) * (
                        ((px - qx)**2 + (py - qy)**2) * (px - 2 * qx + x)
                        + 3 * (px - qx) * (-qx**2 + px * (qx - x) + qx * x + (py - qy) * (qy - y))
                    ) / ((px - qx)**2 + (py - qy)**2)**2.5,

                    "px_qy": (
                        ((px - qx)**2 + (py - qy)**2) * (qx**2 - qx * x + px * (-qx + x) - (py - qy) * (qy - y))
                        + 3 * (py - qy)**2 * (-qx**2 + px * (qx - x) + qx * x + (py - qy) * (qy - y))
                        + ((px - qx)**2 + (py - qy)**2) * (py - qy) * (py - 2 * qy + y)
                    ) / ((px - qx)**2 + (py - qy)**2)**2.5,

                    "py_py": -(
                        (px - qx) * (
                            -3 * (py - qy) * (-qx**2 + px * (qx - x) + qx * x + (py - qy) * (qy - y))
                            + ((px - qx)**2 + (py - qy)**2) * (qy - y)
                        )
                    ) / ((px - qx)**2 + (py - qy)**2)**2.5,

                    "py_qx": (
                        - (px - qx) * ((px - qx)**2 + (py - qy)**2) * (px - 2 * qx + x)
                        - 3 * (px - qx)**2 * (-qx**2 + px * (qx - x) + qx * x + (py - qy) * (qy - y))
                        + ((px - qx)**2 + (py - qy)**2) * (-qx**2 + px * (qx - x) + qx * x + (py - qy) * (qy - y))
                    ) / ((px - qx)**2 + (py - qy)**2)**2.5,

                    "py_qy": -(
                        (px - qx) * (
                            3 * (py - qy) * (-qx**2 + px * (qx - x) + qx * x + (py - qy) * (qy - y))
                            + ((px - qx)**2 + (py - qy)**2) * (py - 2 * qy + y)
                        )
                    ) / ((px - qx)**2 + (py - qy)**2)**2.5,

                    "qx_qx": -(
                        (py - qy) * (
                            ((px - qx)**2 + (py - qy)**2) * (-px + x)
                            + 3 * (px - qx) * (px**2 + py**2 + qx * x - px * (qx + x) + qy * y - py * (qy + y))
                        )
                    ) / ((px - qx)**2 + (py - qy)**2)**2.5,

                    "qx_qy": (
                        - ((px - qx)**2 + (py - qy)**2) * (py - qy) * (-py + y)
                        + ((px - qx)**2 + (py - qy)**2) * (px**2 + py**2 + qx * x - px * (qx + x) + qy * y - py * (qy + y))
                        - 3 * (py - qy)**2 * (px**2 + py**2 + qx * x - px * (qx + x) + qy * y - py * (qy + y))
                    ) / ((px - qx)**2 + (py - qy)**2)**2.5,

                    "qy_qy": -(
                        (px - qx) * (
                            ((px - qx)**2 + (py - qy)**2) * (py - y)
                            - 3 * (py - qy) * (px**2 + py**2 + qx * x - px * (qx + x) + qy * y - py * (qy + y))
                        )
                    ) / ((px - qx)**2 + (py - qy)**2)**2.5,
                }

      H = np.array([
          [partials["px_px"], partials["px_py"], partials["px_qx"], partials["px_qy"], 0],
          [partials["px_py"], partials["py_py"], partials["py_qx"], partials["py_qy"], 0],
          [partials["px_qx"], partials["py_qx"], partials["qx_qx"], partials["qx_qy"], 0],
          [partials["px_qy"], partials["py_qy"], partials["qx_qy"], partials["qy_qy"], 0],
          [0,              0,               0,               0,              0]
      ])
      return sign *H

  def compute_distance(self, X, derivative=False, second_derivative=False):
      """
      Closest-offset distance to the capsule and its derivatives.

      Parameters
      ----------
      X : array_like, shape (2,)
          Query point [x, y].
      derivative : bool, default False
          If True, also return ∂φ/∂s.
      second_derivative : bool, default False
          If True, also return ∂²φ/∂s_m∂s_n.

      Returns
      -------
      If derivative == False:
          (dist, branch) : tuple[float, int]
              `dist` = min distance to {projected segment, end cap P, end cap Q} minus p.
              `branch` ∈ {0: P-cap, 1: segment, 2: Q-cap}.
      If derivative == True and second_derivative == False:
          (dist, grad) : tuple[float, ndarray (5,)]
              grad = ∂φ/∂s evaluated on the active branch.
      If derivative == True and second_derivative == True:
          (dist, grad, H) : tuple[float, ndarray (5,), ndarray (5,5)]
              H = ∂²φ/∂s_m∂s_n for the active branch.

      Notes
      -----
      Chooses among three branches: segment projection, P-end, Q-end.
      The distance is signed along V0 on the segment branch; end-cap
      distances are Euclidean minus p. Derivatives are branch-consistent.
      """    
      width = self.p
      closest_distance = (np.inf, -1)
      H = np.zeros((5, 5))
      h_p = np.zeros((5, 5))
      h_q = np.zeros((5, 5))

      gradients = []
      gradient_proj = []
      gradient_P = []
      gradient_Q = []

      if derivative:
          for var_name in self.optvar_names():
              if var_name == 'p':
                  gradients.append(-1)
              else:
                  var_index = self.namedidx[var_name]
                  sign_factor = np.sign(np.dot(X - self.Q, self.V0))
                  if sign_factor == 0:
                      sign_factor = 1
                  grad_val = sign_factor * (
                      np.dot(X - self.Q, self.sens['grad_u0vert'][var_index, :]) -
                      np.dot(self.sens['grad_Q'][var_index, :], self.V0)
                  )
                  gradients.append(grad_val)
          gradient_proj = np.array(gradients)

      if np.dot((X - self.Q), self.U0) <= 0 and np.dot((X - self.P), self.U0) >= 0:
          projected_distance = abs(np.dot((X - self.Q), self.V0)) - width
          sign = np.sign(np.dot((X - self.Q), self.V0))
          if closest_distance[0] > projected_distance:
              closest_distance = (projected_distance, 1)
          if second_derivative:
              H = self.compute_hessian(X[0],X[1], sign)

      dist_P = norm(X - self.P) - width
      if closest_distance[0] > dist_P:
          closest_distance = (dist_P, 0)
      if derivative and closest_distance[0] == dist_P:
          grad = (self.P - X) / norm(X - self.P) if norm(X - self.P) > 1e-15 else np.ones(2)
          gradient_P = np.concatenate((grad, np.zeros(2), np.array([-1])))
      if second_derivative:
          dx = self.P[0] - X[0]
          dy = self.P[1] - X[1]
          denom = (dx**2 + dy**2) ** 1.5

          h_p[0, 0] = dy**2 / denom
          h_p[0, 1] = h_p[1, 0] = -dx * dy / denom
          h_p[1, 1] = dx**2 / denom


      dist_Q = norm(X - self.Q) - width
      if closest_distance[0] > dist_Q:
          closest_distance = (dist_Q, 2)
      if derivative and closest_distance[0] == dist_Q:
          grad = (self.Q - X) / norm(X - self.Q) if norm(X - self.Q) > 1e-15 else np.ones(2)
          gradient_Q = np.concatenate((np.zeros(2), grad, np.array([-1])))
      if second_derivative:
          denom = norm(X - self.Q) ** 3
          h_q[2:4, 2:4] = np.array([
              [(X[1] - self.Q[1])**2 / denom, -(self.Q[0] - X[0]) * (self.Q[1] - X[1]) / denom],
              [-(self.Q[0] - X[0]) * (self.Q[1] - X[1]) / denom, (self.Q[0] - X[0])**2 / denom]
          ])
          h_q[4, 2:4] = h_q[2:4, 4] = 0

      if not derivative:
          return closest_distance

      if second_derivative:
          if closest_distance[1] == 0:
              return (closest_distance[0], gradient_P, h_p)
          elif closest_distance[1] == 1:
              return (closest_distance[0], gradient_proj, H)
          elif closest_distance[1] == 2:
              return (closest_distance[0], gradient_Q, h_q)
          else:
              return (closest_distance[0], np.zeros(5), np.zeros((5, 5)))
      else:
          if closest_distance[1] == 0:
              return (closest_distance[0], gradient_P)
          elif closest_distance[1] == 1:
              return (closest_distance[0], gradient_proj)
          elif closest_distance[1] == 2:
              return (closest_distance[0], gradient_Q)
          else:
              return (closest_distance[0], np.zeros(5))

  def compute_boundary_value(self, distance, derivative=False, second_derivative=False):
      
      """
      Smooth boundary-to-density mapping ρ(φ) with sensitivities.

      Parameters
      ----------
      distance : float or tuple
          Either φ (float) or the tuple returned by `compute_distance`:
          (φ,) or (φ, ∂φ/∂s) or (φ, ∂φ/∂s, ∂²φ/∂s²).
      derivative : bool, default False
          If True, return ∂ρ/∂s using chain rule via ∂ρ/∂φ ⋅ ∂φ/∂s.
      second_derivative : bool, default False
          If True, return ∂²ρ/∂s² using ∂²ρ/∂φ² and ∂²φ/∂s².

      Returns
      -------
      If derivative == False and second_derivative == False:
          rho : float
              Density in [rhomin, rhomax].
      If derivative == True and second_derivative == False:
          (rho, grad) : tuple[float, ndarray (5,)]
              grad = ∂ρ/∂s.
      If derivative == True and second_derivative == True:
          (rho, grad, H) : tuple[float, ndarray (5,), ndarray (5,5)]
              H = ∂²ρ/∂s².

      Notes
      -----
      Uses per-feature parameters from `glob`:
        - transition half-width h, extension, method ∈ {'smoothstep','sigmoid','beta_mid','smoothstep_shift'},
          and method params (N, k).
      Outside the transition band, ρ is clamped to {rhomax, rhomin} and gradients/Hessians are zero.
      """
      phi = distance if isinstance(distance, (float, np.float64)) else distance[0]
      grad = np.zeros(self.num_optvar)
      hess = np.zeros((self.num_optvar, self.num_optvar))
      min_density = glob.rhomin
      max_density = glob.rhomax
      feature_id = self.id

      h = glob.get_feature_param("transition", feature_id) / 2.0
      method = glob.get_feature_param("boundary", feature_id)
      k = glob.get_feature_param("k", feature_id)
      N = glob.get_feature_param("N", feature_id)
      extension = glob.get_feature_param("extension", feature_id)
      L = h + extension
      H = 2.0*h + extension
      inside_transition = (-h < phi < L)

      if not inside_transition:
          rho = max_density if phi <= -h else min_density
          if not derivative and not second_derivative:
              return rho
          zero_grad = np.zeros_like(distance[1])
          if second_derivative:
              zero_hess = np.zeros_like(distance[2])
              return rho, zero_grad, zero_hess
          return rho, zero_grad
      if -h <= phi <= 0:
          t = (-phi + h) / (2.0 * h)
          scaling = 2.0 * h
      else:
          t = (-phi + h + extension) / (2.0 * (h + extension))
          scaling = 2.0 * (h + extension)

      rho = 0.0
      dB_dphi_scaled = 0.0
      d2B_dphi2_scaled = 0.0
      need_first = derivative or second_derivative 
      if method == "beta_mid":
              alpha = 3.0  
              L = h + extension  
              H = 2.0*h + extension   
              u0 = h / H                  
              beta_param = _find_beta_for_mid(alpha, u0)
              if phi <= -h:
                  rho = max_density
                  if not derivative:
                      return rho
                  zero_grad = np.zeros_like(distance[1])
                  if second_derivative:
                      zero_hess = np.zeros_like(distance[2])
                      return rho, zero_grad, zero_hess
                  return rho, zero_grad
              if phi >= L:
                  rho = min_density
                  if not derivative:
                      return rho
                  zero_grad = np.zeros_like(distance[1])
                  if second_derivative:
                      zero_hess = np.zeros_like(distance[2])
                      return rho, zero_grad, zero_hess
                  return rho, zero_grad
              u = (phi + h) / H
              u_clip = np.clip(u, 0.0, 1.0)
              cdf = betainc(alpha, beta_param, u_clip)
              f_val = 1.0 - cdf
              if derivative or second_derivative:
                  pdf  = _beta_pdf(u, alpha, beta_param)
              if second_derivative:
                  pdfp = _beta_pdf_prime(u, alpha, beta_param)
              df_dphi  = -(pdf)  / H if derivative         else 0.0
              d2f_dphi2= -(pdfp) / (H*H) if second_derivative else 0.0
              span = (max_density - min_density)
              rho = min_density + span * f_val
              dB_dphi_scaled   = span * df_dphi if derivative else 0.0
              d2B_dphi2_scaled = span * d2f_dphi2 if second_derivative else 0.0
      elif method == "smoothstep":
          dB_dphi = 0.0
          d2B_dphi2 = 0.0
          for n in range(N + 1):
              c = (-1)**n * comb(N + n, n) * comb(2 * N + 1, N - n)
              power = N + n + 1
              rho += c * t**power
              if need_first and power >= 1:
                  dB_dphi += c * power * t**(power - 1)
              if second_derivative and power >= 2:
                  d2B_dphi2 += c * power * (power - 1) * t**(power - 2)
          rho = rho * (max_density - min_density) + min_density
          if need_first:
              dB_dphi_scaled = -(max_density - min_density) * dB_dphi / scaling
          if second_derivative:
              d2B_dphi2_scaled = (max_density - min_density) * d2B_dphi2 / (scaling**2)

      elif method == "smoothstep_shift":
          W = 2.0 * h + extension       
          t = (h + extension - phi) / W    
          scaling = W     
          dB_dphi = 0.0
          d2B_dphi2 = 0.0
          for n in range(N + 1):
              c = (-1)**n * comb(N + n, n) * comb(2 * N + 1, N - n)
              power = N + n + 1
              rho += c * t**power
              if need_first and power >= 1:
                  dB_dphi += c * power * t**(power - 1)
              if second_derivative and power >= 2:
                  d2B_dphi2 += c * power * (power - 1) * t**(power - 2)
          rho = rho * (max_density - min_density) + min_density
          if need_first:
              dB_dphi_scaled = -(max_density - min_density) * dB_dphi / scaling
          if second_derivative:
              d2B_dphi2_scaled = (max_density - min_density) * d2B_dphi2 / (scaling**2)

      elif method == "sigmoid":
        if extension > 0:
            if -h <= phi <= 0:
                t_sigmoid = (k * phi) / h
                scaling = h
            else:
                t_sigmoid = (k * phi) / (h + extension)
                scaling = h + extension
        else:
            t_sigmoid = (k * phi) / h
            scaling = h
        tanhk = np.tanh(k)
        rho = 0.5 * (1 + np.tanh(-t_sigmoid) / tanhk)
        rho = min_density + (max_density - min_density) * rho
        if need_first:
            sech2t = 1.0 / np.cosh(t_sigmoid)**2
            dB_dphi = -0.5 * k * sech2t / (scaling * tanhk)
            dB_dphi_scaled = (max_density - min_density) * dB_dphi
        if second_derivative:
            tanht = np.tanh(t_sigmoid)
            d2B_dphi2 = (k**2 / (scaling**2 * tanhk)) * sech2t * tanht
            d2B_dphi2_scaled = (max_density - min_density) * d2B_dphi2
      elif method == "bezier":
          order = int(glob.get_feature_param("bezier_order", feature_id))                           
          gamma = float(glob.get_feature_param("gamma", feature_id))                                
          lut_x, lut_y, lut_dy, lut_d2y = lut_manager.get_lut(order, h, h + extension, gamma)       
                                                                                                    
          rho = min_density + (max_density - min_density) * np.interp(phi, lut_x, lut_y)            
          if need_first:                                                                            
              dB_dphi_scaled = (max_density - min_density) * np.interp(phi, lut_x, lut_dy)          
          if second_derivative:                                                                     
              d2B_dphi2_scaled = (max_density - min_density) * np.interp(phi, lut_x, lut_d2y)       
      else:
          raise ValueError(f"Unbekannte boundary-Methode: {method}")

      if not derivative and not second_derivative:
          return rho

      if derivative:
          grad[0:4] = dB_dphi_scaled * distance[1][0:4]
          grad[4]   = -dB_dphi_scaled

      if not second_derivative:
          return rho, grad

      for i in range(5):
          for j in range(5):
              hess[i, j] = (
                  d2B_dphi2_scaled * distance[1][i] * distance[1][j]
                  + dB_dphi_scaled * distance[2][i][j]
              )

      return rho, grad, hess

  def _signature(self):
      """
      Build a cache signature for raster fields.

      Returns
      -------
      tuple
          Immutable tuple of geometry, grid bounds, resolution, and per-feature
          parameters (transition, boundary method, k, N, extension, rhomin/rhomax).

      Notes
      -----
      Used by `ensure_fields` to detect when cached per-cell fields are stale.
      """
      g = glob
      return (
          float(self.P[0]), float(self.P[1]),
          float(self.Q[0]), float(self.Q[1]),
          float(self.p),
          float(g.bounds["x"][0]), float(g.bounds["x"][1]),
          float(g.bounds["y"][0]), float(g.bounds["y"][1]),
          int(g.n[0]), int(g.n[1]),
          float(glob.get_feature_param("transition", self.id)),
          str(glob.get_feature_param("boundary", self.id)),
          float(glob.get_feature_param("k", self.id)),
          int(glob.get_feature_param("N", self.id)),
          float(glob.get_feature_param("extension", self.id)),
          float(glob.get_feature_param("rhomin", self.id)),
          float(glob.get_feature_param("rhomax", self.id)),
          int(glob.get_feature_param("bezier_order", self.id)),     
          float(glob.get_feature_param("gamma", self.id)))


  def _grid_centers(self):
    """
    Cell-center coordinates of the global grid.

    Returns
    -------
    (xs, ys) : tuple[ndarray, ndarray]
        1D arrays of x- and y-centers as provided by `glob.cell_centers()`.
    """
    xs, ys = glob.cell_centers()
    return xs, ys

  def ensure_fields(self, derivative=False, second_derivative=False):
      """
      Ensure per-cell fields (ρ, ∂ρ/∂s, ∂²ρ/∂s²) are computed and cached.

      Parameters
      ----------
      derivative : bool, default False
          If True, compute and cache `grad_field` with shape (ny, nx, 5).
      second_derivative : bool, default False
          If True, also compute and cache `hessian_field` with shape (ny, nx, 5, 5).

      Side Effects
      ------------
      idx_field : ndarray, shape (ny, nx), uint8
          Region index {0: outside, 1: transition, 2: inside}.
      rho_field : ndarray, shape (ny, nx)
          Density per cell.
      grad_field : ndarray, shape (ny, nx, 5) or None
          ∂ρ/∂s per cell (only if `derivative` or `second_derivative`).
      hessian_field : ndarray, shape (ny, nx, 5, 5) or None
          ∂²ρ/∂s² per cell (only if `second_derivative`).

      Notes
      -----
      Performs branch-wise φ evaluation on the grid and applies the selected
      boundary method. May use numba-accelerated kernels when available; otherwise
      falls back to pointwise calls of `compute_boundary_value(...)`.
      """
      sig = self._signature()
      need_grad = derivative or second_derivative
      need_hess = second_derivative

      if (self._cache_signature == sig and
          (not need_grad or self._has_grad) and
          (not need_hess or self._has_hess)):
          return

      xs, ys = glob.cell_centers()
      nx, ny = glob.n
      X, Y = np.meshgrid(xs, ys) 

      Px, Py = float(self.P[0]), float(self.P[1])
      Qx, Qy = float(self.Q[0]), float(self.Q[1])
      U0x, U0y = float(self.U0[0]), float(self.U0[1])
      V0x, V0y = float(self.V0[0]), float(self.V0[1])
      p = float(self.p)

      dot_Q = (X - Qx)*U0x + (Y - Qy)*U0y
      dot_P = (X - Px)*U0x + (Y - Py)*U0y
      mask_seg = (dot_Q <= 0.0) & (dot_P >= 0.0)

      d_seg_signed = (X - Qx)*V0x + (Y - Qy)*V0y
      phi_seg = np.abs(d_seg_signed) - p


      dPx = X - Px; dPy = Y - Py
      phi_P = np.sqrt(dPx*dPx + dPy*dPy) - p

      dQx = X - Qx; dQy = Y - Qy
      phi_Q = np.sqrt(dQx*dQx + dQy*dQy) - p


      phi = phi_P.copy()
      case = np.zeros_like(phi, dtype=np.uint8)

      mask_Q_better = phi_Q < phi
      phi[mask_Q_better] = phi_Q[mask_Q_better]
      case[mask_Q_better] = 2

      mask_seg_better = mask_seg & (phi_seg < phi)
      phi[mask_seg_better] = phi_seg[mask_seg_better]
      case[mask_seg_better] = 1

      h = glob.get_feature_param("transition", self.id) / 2.0
      ext = glob.get_feature_param("extension", self.id)
      idx = np.zeros_like(case, dtype=np.uint8)
      idx[phi < -h] = 2
      idx[(phi >= -h) & (phi <= (h + ext))] = 1


      method = glob.get_feature_param("boundary", self.id)
      rhomin = glob.get_feature_param("rhomin", self.id)
      rhomax = glob.get_feature_param("rhomax", self.id)
      rho = np.empty_like(phi, dtype=float)
      rho[idx == 2] = rhomax
      rho[idx == 0] = rhomin

      mask_b = (idx == 1)
      if np.any(mask_b):
          t = np.empty_like(phi)
          scaling = np.empty_like(phi)
          left = mask_b & (phi <= 0.0)
          right = mask_b & (phi > 0.0)

          if np.any(left):
              t[left] = (-phi[left] + h) / (2.0*h)
              scaling[left] = 2.0*h
          if np.any(right):
              t[right] = (-phi[right] + h + ext) / (2.0*(h + ext))
              scaling[right] = 2.0*(h + ext)

          if method == "smoothstep":
              N = glob.get_feature_param("N", self.id)
              B = np.zeros_like(phi)
              for n in range(N + 1):
                  c = ((-1)**n) * comb(N + n, n) * comb(2*N + 1, N - n)
                  power = N + n + 1
                  B[mask_b] += c * (t[mask_b]**power)
              rho[mask_b] = rhomin + (rhomax - rhomin) * B[mask_b]
          elif method == "beta_mid":
              alpha = 3.0
              L = h + ext
              H = 2.0*h + ext
              u0 = h / H

              beta_param = _find_beta_for_mid(alpha, u0)

              u = np.empty_like(phi)
              u[mask_b] = (phi[mask_b] + h) / H
              u_clip = np.clip(u[mask_b], 0.0, 1.0)

              cdf   = betainc(alpha, beta_param, u_clip)
              f_val = 1.0 - cdf

              B = np.zeros_like(phi)
              B[mask_b] = f_val

              span = (rhomax - rhomin)
              rho[mask_b] = rhomin + span * B[mask_b]

          elif method == "smoothstep_shift":
              W = 2.0*h + ext
              t = (h + ext - phi) / W
              scaling = W 
              N = glob.get_feature_param("N", self.id)
              B = np.zeros_like(phi)
              for n in range(N + 1):
                  c = ((-1)**n) * comb(N + n, n) * comb(2*N + 1, N - n)
                  power = N + n + 1
                  B[mask_b] += c * (t[mask_b]**power)
              rho[mask_b] = rhomin + (rhomax - rhomin) * B[mask_b]
          elif method == "sigmoid":
              k = glob.get_feature_param("k", self.id)

              t_sig = np.empty_like(phi)
              if np.any(left):
                  t_sig[left] = (k * phi[left]) / h
              if np.any(right):
                  t_sig[right] = (k * phi[right]) / (h + ext)
              tanhk = np.tanh(k)
              base = 0.5 * (1.0 + np.tanh(-t_sig) / tanhk)
              rho[mask_b] = rhomin + (rhomax - rhomin) * base[mask_b]
          
          elif method == "bezier":
            order = int(glob.get_feature_param("bezier_order", self.id))                            
            gamma = float(glob.get_feature_param("gamma", self.id))                                 
            lut_x, lut_y, _, _ = lut_manager.get_lut(order, h, h + ext, gamma)                      
            rho[mask_b] = rhomin + (rhomax - rhomin) * np.interp(phi[mask_b], lut_x, lut_y)         
          else:
              raise ValueError(f"Unbekannte boundary-Methode: {method}")
      self.idx_field = idx
      self.rho_field = rho

      by, bx = np.nonzero(idx == 1)

      ######################## HESS

      if need_hess:
          if NUMBA_AVAILABLE and method in ("smoothstep","sigmoid","beta_mid", "smoothstep_shift", "bezier"):
              Xb = X[by, bx].astype(np.float64)
              Yb = Y[by, bx].astype(np.float64)
              method = glob.get_feature_param("boundary", self.id)

              method_id = -1
              N = 0
              cs = np.empty(1, dtype=np.float64)
              alpha = 3.0
              beta_param = 0.0 

              if method == "smoothstep":
                  method_id = 0
                  N = int(glob.get_feature_param("N", self.id))
                  cs = smoothstep_coeffs(N)

              elif method == "sigmoid":
                  method_id = 1

              elif method == "smoothstep_shift":
                  method_id = 2
                  N = int(glob.get_feature_param("N", self.id))
                  cs = smoothstep_coeffs(N)

              elif method == "beta_mid":
                  method_id = 3
                  H_all = 2.0 * h + ext
                  u0 = h / H_all
                  beta_param = _find_beta_for_mid(alpha, u0)
              
              elif method == "bezier":
                 method_id = 4
                 order = int(glob.get_feature_param("bezier_order", self.id))                        
                 gamma = float(glob.get_feature_param("gamma", self.id))                             
                 lut_x, _, lut_dy, _ = lut_manager.get_lut(order, h, h + ext, gamma)                 
              else:

                  method_id = -1

              h = glob.get_feature_param("transition", self.id) / 2.0
              ext = glob.get_feature_param("extension", self.id)
              k = float(glob.get_feature_param("k", self.id))

              grad_u0vert = self.sens['grad_u0vert'].astype(np.float64)
              grad_Q      = self.sens['grad_Q'].astype(np.float64)
              grad_P      = self.sens['grad_P'].astype(np.float64)

              if method != "bezier":
                  lut_x = np.array([0.0], dtype=np.float64)                                           
                  lut_dy = np.array([0.0], dtype=np.float64)

              Gb = _numba_boundary_grad(
                  Xb, Yb,
                  float(self.P[0]), float(self.P[1]), float(self.Q[0]), float(self.Q[1]),
                  float(self.U0[0]), float(self.U0[1]), float(self.V0[0]), float(self.V0[1]),
                  float(self.p),
                  grad_u0vert, grad_Q, grad_P,
                  float(h), float(ext),
                  int(method_id),
                  float(k), int(N), cs,
                  float(alpha), float(beta_param), lut_x, lut_dy
              )
              scale = float(glob.get_feature_param("rhomax", self.id) - glob.get_feature_param("rhomin", self.id))
              Gb *= scale

              grad = np.zeros((ny, nx, 5), dtype=np.float64)
              grad[by, bx, :] = Gb
              self.grad_field = grad
              self._has_grad = True
          else:
              grad = np.zeros((ny, nx, 5), dtype=float)
              for j, i in zip(by, bx):
                  r, g = self.compute_boundary_value(
                      self.compute_distance(np.array([X[j, i], Y[j, i]]), derivative=True),
                      derivative=True, second_derivative=False
                  )
                  grad[j, i, :] = g
              self.grad_field = grad
              self._has_grad = True
          hess = np.zeros((ny, nx, 5, 5), dtype=float)
          for j, i in zip(by, bx):
              r, g, H = self.compute_boundary_value(
                  self.compute_distance(np.array([X[j, i], Y[j, i]]), derivative=True, second_derivative=True),
                  derivative=True, second_derivative=True
              )
              hess[j, i, :, :] = H
          self.hessian_field = hess
          self._has_hess = True

    ################## GRAD 

      elif need_grad:
          if NUMBA_AVAILABLE and method in ("smoothstep","sigmoid","beta_mid", "smoothstep_shift", "bezier"):
              Xb = X[by, bx].astype(np.float64)
              Yb = Y[by, bx].astype(np.float64)
              method = glob.get_feature_param("boundary", self.id)
              method_id = -1
              N = 0
              cs = np.empty(1, dtype=np.float64)
              alpha = 3.0
              beta_param = 0.0 

              if method == "smoothstep":
                  method_id = 0
                  N = int(glob.get_feature_param("N", self.id))
                  cs = smoothstep_coeffs(N)

              elif method == "sigmoid":
                  method_id = 1

              elif method == "smoothstep_shift":
                  method_id = 2
                  N = int(glob.get_feature_param("N", self.id))
                  cs = smoothstep_coeffs(N)

              elif method == "beta_mid":
                  method_id = 3
                  H_all = 2.0 * h + ext
                  u0 = h / H_all
                  beta_param = _find_beta_for_mid(alpha, u0)
            
              elif method == "bezier": 
                  method_id = 4 
                  order = int(glob.get_feature_param("bezier_order", self.id))                        
                  gamma = float(glob.get_feature_param("gamma", self.id))                             
                  lut_x, _, lut_dy, _ = lut_manager.get_lut(order, h, h + ext, gamma)                 
         

              else:
                  method_id = -1

              h = glob.get_feature_param("transition", self.id) / 2.0
              ext = glob.get_feature_param("extension", self.id)
              k = float(glob.get_feature_param("k", self.id))

              grad_u0vert = self.sens['grad_u0vert'].astype(np.float64)
              grad_Q      = self.sens['grad_Q'].astype(np.float64)
              grad_P      = self.sens['grad_P'].astype(np.float64)

              if method != "bezier":                         
                  lut_x = np.array([0.0], dtype=np.float64)                                           
                  lut_dy = np.array([0.0], dtype=np.float64)
              Gb = _numba_boundary_grad(
                  Xb, Yb,
                  float(self.P[0]), float(self.P[1]), float(self.Q[0]), float(self.Q[1]),
                  float(self.U0[0]), float(self.U0[1]), float(self.V0[0]), float(self.V0[1]),
                  float(self.p),
                  grad_u0vert, grad_Q, grad_P,
                  float(h), float(ext),
                  int(method_id),
                  float(k), int(N), cs,
                  float(alpha), float(beta_param),lut_x, lut_dy
              )
              scale = float(glob.get_feature_param("rhomax", self.id) - glob.get_feature_param("rhomin", self.id))
              Gb *= scale

              grad = np.zeros((ny, nx, 5), dtype=np.float64)
              grad[by, bx, :] = Gb
              self.grad_field = grad
              self.hessian_field = None
              self._has_grad = True
              self._has_hess = False
          else:
              grad = np.zeros((ny, nx, 5), dtype=float)
              for j, i in zip(by, bx):
                  r, g = self.compute_boundary_value(
                      self.compute_distance(np.array([X[j, i], Y[j, i]]), derivative=True),
                      derivative=True, second_derivative=False
                  )
                  grad[j, i, :] = g
              self.grad_field = grad
              self.hessian_field = None
              self._has_grad = True
              self._has_hess = False
      else:
          self.grad_field = None
          self.hessian_field = None
          self._has_grad = False
          self._has_hess = False

      self._cache_signature = sig



  def sample(self, x, y, derivative=False, second_derivative=False, interpolate=True):
      """
      Sample ρ (and optionally ∂ρ/∂s, ∂²ρ/∂s²) at a continuous position.

      Parameters
      ----------
      x, y : float
          Query coordinates in the global domain.
      derivative : bool, default False
          If True, also return ∂ρ/∂s.
      second_derivative : bool, default False
          If True, also return ∂²ρ/∂s².
      interpolate : bool, default True
          If True, bilinear interpolation over cell centers; if False, nearest neighbor.

      Returns
      -------
      If derivative == False and second_derivative == False:
          rho : float
      If derivative == True and second_derivative == False:
          (rho, grad) : tuple[float, ndarray (5,)]
      If derivative == True and second_derivative == True:
          (rho, grad, H) : tuple[float, ndarray (5,), ndarray (5,5)]

      Notes
      -----
      Ensures caches are up to date via `ensure_fields(...)` before sampling.
      """
      self.ensure_fields(derivative=derivative, second_derivative=second_derivative)
      x0, x1 = glob.bounds["x"]; y0, y1 = glob.bounds["y"]
      nx, ny = glob.n
      dx = (x1 - x0)/nx; dy = (y1 - y0)/ny
      mode = "bilinear" if interpolate else "nn"
      fx = (x - (x0 + 0.5*dx)) / dx
      fy = (y - (y0 + 0.5*dy)) / dy

      if mode == "nn":
          i = int(np.clip(np.round(fx), 0, nx-1))
          j = int(np.clip(np.round(fy), 0, ny-1))
          r = self.rho_field[j, i]
          if not derivative: return r
          g = self.grad_field[j, i, :] if self.grad_field is not None else np.zeros(5)
          if not second_derivative: return r, g
          H = self.hessian_field[j, i, :, :] if self.hessian_field is not None else np.zeros((5,5))
          return r, g, H

      # bilinear
      i0 = int(np.clip(np.floor(fx), 0, nx-1)); i1 = min(i0+1, nx-1)
      j0 = int(np.clip(np.floor(fy), 0, ny-1)); j1 = min(j0+1, ny-1)
      tx = np.clip(fx - i0, 0.0, 1.0); ty = np.clip(fy - j0, 0.0, 1.0)

      def blin(A):
          return ( (1-tx)*(1-ty)*A[j0,i0] + tx*(1-ty)*A[j0,i1]
                + (1-tx)*ty*A[j1,i0]    + tx*ty*A[j1,i1] )

      r = blin(self.rho_field)
      if not derivative: return r
      g = blin(self.grad_field) if self.grad_field is not None else np.zeros(5)
      if not second_derivative: return r, g
      H = blin(self.hessian_field) if self.hessian_field is not None else np.zeros((5,5))
      return r, g, H



def build_hessian(grad_shapes, hess_shapes, order):
    """
    Assemble integration-point averaged Hessian across features.

    Parameters
    ----------
    grad_shapes : list[ndarray]
        Per-feature gradient at each integration point; each entry has shape
        (order*order, m) with m = 5 (Px, Py, Qx, Qy, p).
    hess_shapes : list[ndarray]
        Per-feature Hessian at each integration point; each entry has shape
        (order*order, m, m).
    order : int
        Number of 1D quadrature subdivisions per cell edge; total points = order^2.

    Returns
    -------
    H_total : ndarray, shape (n*m, n*m)
        Cell-local Hessian averaged over integration points (block-structured
        w.r.t. features).
    """
    num_shapes = len(grad_shapes)
    num_vars = grad_shapes[0].shape[1]
    H_total = np.zeros((num_shapes * num_vars, num_shapes * num_vars))

    for ip in range(order * order):
        for i in range(num_shapes):
            i0 = i * num_vars
            g_i = grad_shapes[i][ip]
            for j in range(num_shapes):
                j0 = j * num_vars
                g_j = grad_shapes[j][ip]
                term = np.outer(g_i, g_j)
                if i == j:
                    term += hess_shapes[i][ip]
                H_total[i0:i0 + num_vars, j0:j0 + num_vars] += term / (order * order)
    return H_total

def _collect_shape_samples(x, y, derivative=False, second_derivative=False, interpolate=True):
    """
    Sample all registered features at (x,y) and collect ρ_i, ∇ρ_i, ∇²ρ_i.

    Parameters
    ----------
    x, y : float
        Query coordinates.
    derivative : bool, default False
        If True, also collect ∇ρ_i.
    second_derivative : bool, default False
        If True, also collect ∇²ρ_i.
    interpolate : bool, default True
        Delegated to each feature's `sample`.

    Returns
    -------
    rhos : ndarray, shape (n,)
        Per-feature densities.
    grads : list[ndarray] | None
        List of ∇ρ_i (each shape (m,)) if requested.
    hesss : list[ndarray] | None
        List of ∇²ρ_i (each shape (m,m)) if requested.
    optvar_dim : int
        m (number of variables per feature, here 5).
    total_dim : int
        n*m (concatenated design dimension).
    """
    shapes = glob.shapes
    n = len(shapes)
    if n == 0:
        if not derivative:
            return np.array([]), None, None, 0, 0
        if not second_derivative:
            return np.array([]), [], None, 0, 0
        return np.array([]), [], [], 0, 0

    optvar_dim = len(shapes[0].optvar())   # = 5
    total_dim = n * optvar_dim

    rhos = []
    grads = [] if derivative else None
    hesss = [] if second_derivative else None

    for s in shapes:
        if second_derivative:
            rho_i, g_i, H_i = s.sample(x, y, derivative=True, second_derivative=True, interpolate=interpolate)
            rhos.append(rho_i); grads.append(g_i); hesss.append(H_i)
        elif derivative:
            rho_i, g_i = s.sample(x, y, derivative=True, second_derivative=False, interpolate=interpolate)
            rhos.append(rho_i); grads.append(g_i)
        else:
            rho_i = s.sample(x, y, derivative=False, second_derivative=False, interpolate=interpolate)
            rhos.append(rho_i)

    return np.array(rhos), grads, hesss, optvar_dim, total_dim



"""
Combination operators g({ρ_i}) that aggregate per-feature densities (and their
sensitivities) at a query point. Each aggregator supports value, gradient, and
Hessian with respect to the concatenated design vector s of all features.
"""

def combine_designs(x, y, derivative=False, second_derivative=False, interpolate=True):
    """
    Dispatch to the active aggregation rule (`glob.combine`).

    Parameters
    ----------
    x, y : float
    derivative, second_derivative : bool
        Sensitivity flags propagated to the chosen combiner.
    interpolate : bool
        Passed to feature sampling.

    Returns
    -------
    Depending on flags:
      - ρ : float
      - (ρ, ∇ρ) : (float, ndarray (n*m,))
      - (ρ, ∇ρ, ∇²ρ) : (float, ndarray (n*m,), ndarray (n*m, n*m))
    """
    if glob.combine == 'p-norm':
        return combine_p_norm(x, y, derivative, second_derivative, interpolate=interpolate)
    elif glob.combine == 'softmax':
        return combine_softmax(x, y, derivative, second_derivative, interpolate=interpolate)
    elif glob.combine == 'sum':
        return combine_sum(x, y, derivative, second_derivative, interpolate=interpolate)
    elif glob.combine == 'harmonic':
        return combine_harmonic_mean(x, y, derivative, second_derivative, interpolate=interpolate)
    elif glob.combine == 'sum-softcap':
        return combine_sum_softcap(x, y, derivative, second_derivative, T=glob.sum_cap, beta=glob.beta, interpolate=interpolate)
    elif glob.combine == 'cosine':
        return combine_cosine_weighted(x, y, derivative, second_derivative, interpolate=interpolate)
    else:
        raise ValueError(f"Unbekannte Kombination: {glob.combine}")


def combine_sum_softcap(x, y, derivative=False, second_derivative=False, T=None, beta=None, interpolate=True):
    """
    Soft-capped sum: g(S) = softmin(S, T) = - (1/β) log( exp(-β S) + exp(-β T) ),  S=∑ρ_i.

    Parameters
    ----------
    x, y : float
    derivative, second_derivative : bool
    T : float, optional
        Cap level; defaults to `glob.sum_cap`.
    beta : float, optional
        Softness parameter; defaults to `glob.beta`.
    interpolate : bool

    Returns
    -------
    ρ | (ρ, ∇ρ) | (ρ, ∇ρ, ∇²ρ)
        Aggregated value and sensitivities w.r.t. concatenated s.
    """
    if beta is None: beta = glob.beta
    if T is None: T = glob.sum_cap

    rhos, grads, hesss, optvar_dim, total_dim = _collect_shape_samples(
        x, y, derivative=derivative, second_derivative=second_derivative, interpolate=interpolate
    )
    n = len(rhos)
    if n == 0:
        if not derivative: return T
        if not second_derivative: return (T, np.zeros(0))
        return (T, np.zeros(0), np.zeros((0,0)))

    S = float(np.sum(rhos))

    m = max(-beta*S, -beta*T)
    g = -(m + np.log(np.exp(-beta*S - m) + np.exp(-beta*T - m))) / beta

    if not derivative:
        return g
    gS = np.zeros(total_dim)
    for i, gi in enumerate(grads):
        i0 = i * optvar_dim
        gS[i0:i0+optvar_dim] = gi
    tau = 1.0 / (1.0 + np.exp(beta*(S - T)))

    grad = tau * gS

    if not second_derivative:
        return g, grad
    H_S = np.zeros((total_dim, total_dim))
    for i, Hi in enumerate(hesss):
        i0 = i * optvar_dim
        H_S[i0:i0+optvar_dim, i0:i0+optvar_dim] = Hi
    H = tau * H_S + (beta * tau * (1.0 - tau)) * np.outer(gS, gS)

    return g, grad, H


def combine_harmonic_mean(x, y, derivative=False, second_derivative=False, epsilon=1e-6, interpolate=True):
    """
    Harmonic mean aggregation: ρ = ( (1/n) Σ_i 1/(ρ_i + ε) )^{-1}.

    Parameters
    ----------
    x, y : float
    derivative, second_derivative : bool
    epsilon : float, default 1e-6
        Positivity regularization for stability.
    interpolate : bool

    Returns
    -------
    ρ | (ρ, ∇ρ) | (ρ, ∇ρ, ∇²ρ)
        Value and sensitivities w.r.t. stacked design variables.
    """
    rhos, grads, hesss, optvar_dim, total_dim = _collect_shape_samples(
        x, y, derivative=derivative, second_derivative=second_derivative, interpolate=interpolate
    )
    n = len(rhos)
    if n == 0:
        if not derivative:
            return 0.0
        if not second_derivative:
            return 0.0, np.zeros(0)
        return 0.0, np.zeros(0), np.zeros((0,0))

    rho_eff = rhos + epsilon
    inv_rho = 1.0 / rho_eff
    avg_inv = np.mean(inv_rho)
    combined_rho = 1.0 / avg_inv

    if not derivative:
        return combined_rho

    combined_grad = np.zeros(total_dim)
    for i, (rho_i, grad_i) in enumerate(zip(rho_eff, grads)):
        i0 = i * optvar_dim
        coeff = 1.0 / (n * rho_i**2)
        combined_grad[i0:i0 + optvar_dim] = coeff * grad_i

    combined_grad *= combined_rho**2

    if not second_derivative:
        return combined_rho, combined_grad

    combined_hess = np.zeros((total_dim, total_dim))

    for i in range(n):
        rho_i = rho_eff[i]
        g_i = grads[i]
        H_i = hesss[i]
        i0 = i * optvar_dim

        term1 = (2.0 / (rho_i**3)) * np.outer(g_i, g_i)
        term2 = (1.0 / (rho_i**2)) * H_i
        block = term1 - term2

        combined_hess[i0:i0 + optvar_dim, i0:i0 + optvar_dim] = block

    combined_hess *= -combined_rho**2 / n
    combined_hess += 2 * np.outer(combined_grad, combined_grad) / combined_rho


    return combined_rho, combined_grad, combined_hess



def combine_softmax(x, y, derivative=False, second_derivative=False,interpolate=True):
    """
    Log-sum-exp (softmax) aggregation over ρ_i.

    Parameters
    ----------
    x, y : float
    derivative, second_derivative : bool
    interpolate : bool

    Returns
    -------
    ρ | (ρ, ∇ρ) | (ρ, ∇ρ, ∇²ρ)
        With β = glob.beta, uses weights w_i = softmax(β ρ_i).
    """
    beta = glob.beta
    rhos, grads, hesss, optvar_dim, total_dim = _collect_shape_samples(
        x, y, derivative=derivative, second_derivative=second_derivative, interpolate=interpolate
    )
    n = len(rhos)
    if n == 0:
        return (0.0 if not derivative else (0.0, np.zeros(0)) if not second_derivative else (0.0, np.zeros(0), np.zeros((0,0))))

    m = np.max(beta * rhos)
    exp_vals = np.exp(beta * rhos - m)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    combined_rho = (m + np.log(sum_exp)) / beta


    if not derivative:
        return combined_rho

    combined_grad = np.zeros(total_dim)

    for i, (w, g) in enumerate(zip(weights, grads)):
        i0 = i * optvar_dim
        combined_grad[i0:i0 + optvar_dim] = w * g

    if not second_derivative:
        return combined_rho, combined_grad

    combined_hess = np.zeros((total_dim, total_dim))
    for i in range(len(rhos)):
        i0 = i * optvar_dim
        gi = grads[i]
        Hi = hesss[i]
        wi = weights[i]
        combined_hess[i0:i0 + optvar_dim, i0:i0 + optvar_dim] += wi * Hi

        for j in range(len(rhos)):
            j0 = j * optvar_dim
            gj = grads[j]
            wj = weights[j]

            wij_outer = np.outer(gi, gj)
            if i == j:
                combined_hess[i0:i0 + optvar_dim, j0:j0 + optvar_dim] += beta * wi * (1 - wi) * wij_outer
            else:
                combined_hess[i0:i0 + optvar_dim, j0:j0 + optvar_dim] += -beta * wi * wj * wij_outer

    return combined_rho, combined_grad, combined_hess

def combine_p_norm(x, y, derivative=False, second_derivative=False, interpolate=True):
    """
    p-norm aggregation: g = (Σ_i ρ_i^p)^{1/p}.

    Parameters
    ----------
    x, y : float
    derivative, second_derivative : bool
    interpolate : bool

    Returns
    -------
    ρ | (ρ, ∇ρ) | (ρ, ∇ρ, ∇²ρ)

    Notes
    -----
    Uses `build_pnorm_hessian(...)` for the exact Hessian when requested.
    """
    p = glob.p
    rhos, grads, hesss, optvar_dim, total_dim = _collect_shape_samples(
        x, y, derivative=derivative, second_derivative=second_derivative, interpolate=interpolate
    )
    n = len(rhos)
    if n == 0:
        return (0.0 if not derivative else (0.0, np.zeros(0)) if not second_derivative else (0.0, np.zeros(0), np.zeros((0,0))))

    S = np.sum(rhos**p)
    gval = S**(1.0/p)

    if not derivative:
        return gval

    combined_grad = np.zeros(total_dim)
    for i, (rho_i, gi) in enumerate(zip(rhos, grads)):
        if rho_i > 0:
            i0 = i * optvar_dim
            combined_grad[i0:i0+optvar_dim] = (rho_i**(p - 1)) * gi
    combined_grad *= 1.0 / (gval**(p - 1))

    if not second_derivative:
        return gval, combined_grad

    combined_hess = build_pnorm_hessian(rhos, grads, hesss,p)

    return gval, combined_grad, combined_hess


def combine_sum(x, y, derivative=False, second_derivative=False, interpolate=True):
    """
    Clamped sum aggregation: ρ = Σ_i max(ρ_i, rhomin).

    Parameters
    ----------
    x, y : float
    derivative, second_derivative : bool
    interpolate : bool

    Returns
    -------
    ρ | (ρ, ∇ρ) | (ρ, ∇ρ, ∇²ρ)
        Zero blocks are used for features with ρ_i ≤ rhomin.
    """
    rhos, grads, hesss, optvar_dim, total_dim = _collect_shape_samples(
        x, y, derivative=derivative, second_derivative=second_derivative, interpolate=interpolate
    )
    n = len(rhos)
    if n == 0:
        return (0.0 if not derivative else (0.0, np.zeros(0)) if not second_derivative else (0.0, np.zeros(0), np.zeros((0,0))))

    rho_eff = np.maximum(rhos, glob.rhomin)
    combined_rho = np.sum(rho_eff)

    if not derivative:
        return combined_rho

    combined_grad = np.zeros(total_dim)
    for i, g in enumerate(grads):
        i0 = i * optvar_dim
        combined_grad[i0:i0+optvar_dim] = g if rhos[i] > glob.rhomin else 0.0

    if not second_derivative:
        return combined_rho, combined_grad

    combined_hess = np.zeros((total_dim, total_dim))
    for i, H in enumerate(hesss):
        if rhos[i] > glob.rhomin:
            i0 = i * optvar_dim
            combined_hess[i0:i0+optvar_dim, i0:i0+optvar_dim] = H
    return combined_rho, combined_grad, combined_hess


"""
Exact Hessian builders for the p-norm aggregator with nonnegativity handling.
"""

def _pow_pos(x, q):
    """
    Stable power on the nonnegative part.

    Parameters
    ----------
    x : float
    q : float

    Returns
    -------
    float
        x^q for x>0, else 0.0 (avoids NaNs for negative bases with fractional q).
    """
    if x <= 0.0:
        return 0.0
    return float(np.power(x, q))

def build_pnorm_hessian(rho_values, gradients, hessians, p, eps=1e-300):
    """
    Exact Hessian for g = (Σ_i max(ρ_i,0)^p)^{1/p} (vectorized, stable).

    Parameters
    ----------
    rho_values : array_like, shape (n,)
        Per-feature ρ_i at a point.
    gradients : list[ndarray]
        ∇ρ_i, each shape (m,).
    hessians : list[ndarray]
        ∇²ρ_i, each shape (m,m).
    p : float
        p ≥ 1.
    eps : float, default 1e-300
        Tiny guard for degenerate S.

    Returns
    -------
    H : ndarray, shape (n*m, n*m)
        Exact ∇²g w.r.t. concatenated design variables.
    """
    rho_values = np.asarray(rho_values, dtype=np.float64)
    n_features = len(rho_values)
    optvar_dim = len(gradients[0])
    total_dim  = n_features * optvar_dim

    H = np.zeros((total_dim, total_dim), dtype=np.float64)

    r_pos = np.maximum(rho_values, 0.0).astype(np.float64)

    S = float(np.sum(np.power(r_pos, p)))
    if S < eps:
        return H
    gval = float(np.power(S, 1.0 / p))  
    S_a  = gval / S                    
    S_b  = S_a / S                     


    if p == 1:
        for i_f in range(n_features):
            i0 = i_f * optvar_dim
            H[i0:i0+optvar_dim, i0:i0+optvar_dim] = np.asarray(hessians[i_f], dtype=np.float64)
        return H


    eps_den = 1e-300
    for i_f in range(n_features):
        ri = float(r_pos[i_f])
        if ri <= 0.0:
            continue
        gi = np.asarray(gradients[i_f], dtype=np.float64)
        ui = _pow_pos(ri, p) / S  

        for j_f in range(n_features):
            if i_f == j_f:
                continue
            rj = float(r_pos[j_f])
            if rj <= 0.0:
                continue
            gj = np.asarray(gradients[j_f], dtype=np.float64)
            uj = _pow_pos(rj, p) / S  

            denom = max(ri * rj, eps_den)
            coef = (1.0 - p) * gval * (ui * uj) / denom

            i0 = i_f * optvar_dim
            j0 = j_f * optvar_dim
            H[i0:i0+optvar_dim, j0:j0+optvar_dim] += coef * np.outer(gi, gj)

    for f_idx in range(n_features):
        r = float(r_pos[f_idx])
        if r == 0.0:
            continue

        g  = np.asarray(gradients[f_idx], dtype=np.float64)
        Hf = np.asarray(hessians[f_idx],  dtype=np.float64)
        i0 = f_idx * optvar_dim

        r_p   = _pow_pos(r, p)        
        r_p_2 = _pow_pos(r, p - 2.0)   
        r_p_1 = _pow_pos(r, p - 1.0)   
        if r_p == 0.0:
            continue


        u = r_p / S

        coef_gg = ((1.0 - p) * u + (p - 1.0)) * S_a * r_p_2
        coef_H  = S_a * r_p_1

        for i in range(optvar_dim):
            dgi  = g[i]
            d2gi = Hf[i, i]
            H[i0 + i, i0 + i] = coef_gg * (dgi * dgi) + coef_H * d2gi
        for i in range(optvar_dim):
            dgi = g[i]
            for j in range(optvar_dim):
                if i == j:
                    continue
                dgj   = g[j]
                d2gij = Hf[i, j]
                H[i0 + i, i0 + j] = coef_gg * (dgi * dgj) + coef_H * d2gij

    return H


"""
Piecewise-smooth cap function A(Σρ_i) with closed-form first/second derivatives.
"""

def compute_cosine_weighted(x, N):
    """
    Piecewise cosine-weighted cap A(x) with N-dependent plateau.

    Parameters
    ----------
    x : float
        Aggregated input (typically x = Σ_i ρ_i).
    N : int
        Number of terms (features) used for shaping.

    Returns
    -------
    float
        A(x) per the piecewise definition.
    """
    a = 1 - ((N - 1)**2) / 2

    if x <= 1:
        return np.sin(x * np.pi / 2)
    elif x < N:
        return a + (1 - a) * 0.5 * (1 + np.cos(np.pi * (x - 1) / (N - 1)))
    else:
        return a

def compute_cosine_weighted_jacobian(x, N):
    """
    First derivative A'(x) for the cosine-weighted cap.

    Parameters
    ----------
    x : float
    N : int

    Returns
    -------
    float
        dA/dx evaluated at x.
    """
    a = 1 - ((N - 1)**2) / 2

    if x <= 1:
        return np.cos(x * np.pi / 2) * (np.pi / 2)
    elif x < N:
        return ( (np.pi / 2) * (a - 1) * np.sin(np.pi * (x - 1) / (N - 1))) / (N - 1)
    else:
        return 0.0


def compute_cosine_weighted_hessian(x, N):
    """
    Second derivative A''(x) for the cosine-weighted cap.

    Parameters
    ----------
    x : float
    N : int

    Returns
    -------
    float
        d²A/dx² evaluated at x.
    """
    a = 1 - ((N - 1)**2) / 2

    if x <= 1:
        return -np.sin(x * np.pi / 2) * (np.pi / 2)**2
    elif x < N:
        return ((np.pi**2 / 2) * (a - 1) * np.cos(np.pi * (1 - x) / (N - 1))) / (N - 1)**2
    else:
        return 0.0


def combine_cosine_weighted(x, y, derivative=False, second_derivative=False, interpolate=True):
    """
    Cosine-weighted aggregation: A(S) with S = Σ_i ρ_i.

    Parameters
    ----------
    x, y : float
    derivative, second_derivative : bool
    interpolate : bool

    Returns
    -------
    ρ | (ρ, ∇ρ) | (ρ, ∇ρ, ∇²ρ)
        With chain rule using A'(S) and A''(S).
    """
    rhos, grads, hesss, optvar_dim, total_dim = _collect_shape_samples(
        x, y, derivative=derivative, second_derivative=second_derivative, interpolate=interpolate
    )
    n = len(rhos)
    if n == 0:
        return (0.0 if not derivative else (0.0, np.zeros(0)) if not second_derivative else (0.0, np.zeros(0), np.zeros((0,0))))

    S = np.sum(rhos)
    A = compute_cosine_weighted(S, n)

    if not derivative:
        return A

    dA = compute_cosine_weighted_jacobian(S, n)
    combined_grad = np.zeros(total_dim)
    for i, gi in enumerate(grads):
        i0 = i * optvar_dim
        combined_grad[i0:i0+optvar_dim] = dA * gi

    if not second_derivative:
        return A, combined_grad

    d2A = compute_cosine_weighted_hessian(S, n)
    combined_hess = np.zeros((total_dim, total_dim))
    for i in range(n):
        i0 = i * optvar_dim
        gi = grads[i]; Hi = hesss[i]
        for j in range(n):
            j0 = j * optvar_dim
            gj = grads[j]
            outer = d2A * np.outer(gi, gj)
            if i == j:
                block = dA * Hi + outer
            else:
                block = outer
            combined_hess[i0:i0+optvar_dim, j0:j0+optvar_dim] = block

    return A, combined_grad, combined_hess

"""
Compute per-cell averages from multiple integration points for robustness and
reduced quadrature error; supports value, gradient, and Hessian.
"""

def combine_integration_points(x, y, derivative=False, second_derivative=False):
    """
    Average aggregator outputs over order×order subcell integration points.

    Parameters
    ----------
    x, y : float
        Cell origin (lower-left) coordinates.
    derivative, second_derivative : bool
        Sensitivity flags forwarded to `combine_designs`.
    Returns
    -------
    ρ | (ρ, ∇ρ) | (ρ, ∇ρ, ∇²ρ)
        Averaged over the chosen quadrature stencil.
    """
    order = glob.order
    dx = glob.dx
    dy = glob.dy

    rho_vals = []
    grad_vals = []
    hess_vals = []
    if order == 1:
        rel_points = [(0.5, 0.5)]
    else:
        grid = [i / (order - 1) for i in range(order)]
        rel_points = [(rx, ry) for rx in grid for ry in grid]

    for rel_x, rel_y in rel_points:
        px = x + rel_x * dx
        py = y + rel_y * dy

        result = combine_designs(px, py, derivative=derivative, second_derivative=second_derivative)

        if second_derivative:
            rho, grad, hess = result
            rho_vals.append(rho)
            grad_vals.append(grad)
            hess_vals.append(hess)
        elif derivative:
            rho, grad = result
            rho_vals.append(rho)
            grad_vals.append(grad)
        else:
            rho_vals.append(result)

    avg_rho = np.mean(rho_vals)

    if derivative:
        avg_grad = np.mean(np.stack(grad_vals), axis=0)
    if second_derivative:
        avg_hess = np.mean(np.stack(hess_vals), axis=0)

    if second_derivative:
        return avg_rho, avg_grad, avg_hess
    elif derivative:
        return avg_rho, avg_grad
    else:
        return avg_rho


def _sync_shapes_from_s(s):
    """
    Rebuild `glob.shapes` from a flat design vector s (5 vars per feature).

    Parameters
    ----------
    s : array_like, shape (5*n,)
        Concatenated [Px, Py, Qx, Qy, p] for n features.

    Returns
    -------
    n : int
        Number of features after sync.

    Notes
    -----
    Clears existing shapes and instantiates `Pill` features with ids 0..n-1.
    """
    glob.clear_shapes()
    num = len(s) // 5
    for sid in range(num):
        P = s[sid*5     : sid*5+2]
        Q = s[sid*5+2   : sid*5+4]
        r = s[sid*5+4]
        glob.add_shape(Pill(sid, P, Q, r))
    return num

"""
Rasterize aggregated density and its sensitivities on the global cell grid:
  - dichte(s)    : ρ per cell,
  - ableitung(s) : ∂ρ/∂s per cell,
  - hessian(s)   : ∂²ρ/∂s² per cell.
"""

def dichte(s):
    """
    Cell-wise density S(x,y; s) on the cell-origin grid.

    Parameters
    ----------
    s : array_like, shape (5*n,)
        Design vector stacked over all features.

    Returns
    -------
    rho : ndarray, shape (ny, nx)
        Aggregated density at each cell (averaged over integration points).
    """
    x0, y0 = glob.cell_origins()
    nx, ny = x0.size, y0.size

    rho = np.zeros((ny, nx))
    for j, y in enumerate(y0):
        for i, x in enumerate(x0):
            rho[j, i] = combine_integration_points(x, y, derivative=False)
    return rho

def ableitung(s):
    """
    Cell-wise first derivative ∂S/∂s on the cell-origin grid.

    Parameters
    ----------
    s : array_like, shape (5*n,)

    Returns
    -------
    grad_vals : ndarray, shape (n*5, nx, ny)
        For indexing consistency: grad[:, i, j] corresponds to cell (i,j).
    """
    num_shapes = _sync_shapes_from_s(s)
    var_size = 5
    total_vars = num_shapes * var_size

    x0, y0 = glob.cell_origins()
    nx, ny = x0.size, y0.size

    grad_vals = np.zeros((total_vars, nx, ny))
    for j, y in enumerate(y0):
        for i, x in enumerate(x0):
            _, grad = combine_integration_points(x, y, derivative=True)
            grad_vals[:, i, j] = grad
    return grad_vals

def hessian(s):
    """
    Cell-wise second derivative ∂²S/∂s_m∂s_n on the cell-origin grid.

    Parameters
    ----------
    s : array_like, shape (5*n,)

    Returns
    -------
    H : ndarray, shape (n*5, n*5, ny, nx)
        Per-cell Hessian blocks (averaged over integration points).
    """
    num_shapes = _sync_shapes_from_s(s)
    var_size = 5
    total_vars = num_shapes * var_size

    x0, y0 = glob.cell_origins()
    nx, ny = x0.size, y0.size

    H = np.zeros((total_vars, total_vars, ny, nx))
    for j, y in enumerate(y0):
        for i, x in enumerate(x0):
            _, _, H_total = combine_integration_points(x, y, derivative=True, second_derivative=True)
            H[:, :, j, i] = H_total
    return H


"""
Diagnostics for per-feature coverage:
  - AR_n: area ratio (feature area normalized by max feature area),
  - UR_n: uniqueness ratio (fraction of feature area not overlapped by others).
"""

def compute_AR_UR(s):
    """
    Compute area ratio (AR_n) and uniqueness ratio (UR_n) for each feature.

    Parameters
    ----------
    s : array_like, shape (5*n,)
        Design vector; features are evaluated individually.

    Returns
    -------
    AR_list : list[float], len n
        A_n / max_k A_k  with A_n = ∑_cells ρ_n ΔA.
    UR_list : list[float], len n
        A_unq_n / A_n where A_unq_n counts cells where only feature n contributes.

    Notes
    -----
    Uses current grid spacing ΔA = dx*dy and `dichte` per single-feature scene.
    """
    num_features = len(s) // 5
    dx, dy = glob.dx, glob.dy
    area_scale = dx * dy

    A_n_list = []
    A_unq_list = []
    densities = []
    for i in range(num_features):
        s_i = s[i * 5: (i + 1) * 5]
        glob.shapes = [Pill(0, s_i[0:2], s_i[2:4], s_i[4])]
        rho = dichte(s_i)
        densities.append(rho)

        A_n = np.sum(rho) * area_scale
        A_n_list.append(A_n)

    max_area = max(A_n_list)
    for i in range(num_features):
        rho_i = densities[i]
        other_rhos = [densities[j] for j in range(num_features) if j != i]
        other_rhos_stack = np.stack(other_rhos)  # shape: (n-1, ny, nx)
        overlap_mask = np.all(other_rhos_stack <= 1e-25, axis=0)

        A_unq_n = np.sum(rho_i * overlap_mask) * area_scale
        A_unq_list.append(A_unq_n)

    AR_list = [A / max_area if max_area > 0 else 0.0 for A in A_n_list]
    UR_list = [A_unq / A if A > 0 else 0.0 for A_unq, A in zip(A_unq_list, A_n_list)]

    return AR_list, UR_list
