"""bezier_math.py - self-contained Bezier transition function library.

Provides:
  Bezier(order, a, b, gamma)   - transition curve class
  find_gamma(order, a, b)      - optimal gamma via minimax |d2y/dx2|
"""

import numpy as np
import math
import scipy.optimize


class Bezier:
    """Bezier transition function  y : [-a, b] -> [1, 0].

    The curve is parametrised by t in [0, 1]:
      x(t) = Bernstein(Px, t),   Px determined by (order, a, b, gamma)
      y(t) = Bernstein(Py, t),   Py fixed to enforce y(-a)=1, y(b)=0,
                                  y'=0 at both endpoints, y(0)=0.5

    Parameters
    ----------
    order : int
        Bezier degree - 3 (cubic) or 5 (quintic).
    a : float
        Left half-width; left endpoint is -a.  Requires b >= a > 0.
    b : float
        Right endpoint.
    gamma : float
        Shape parameter (delta in the paper).  gamma=0 gives the smoothstep.
    """

    def __init__(self, order: int, a: float, b: float, gamma: float):
        assert(order in (3, 5))
        self.order = order
        self.a     = float(a)
        self.b     = float(b)
        self.gamma = float(gamma)

        n = order

        # arrays of x an y components of the control points W_i
        self.Wx = self._build_Wx()
        self.Wy = self._build_Wy()

        # precomputed binomial coefficients for the original curve and the first and second derivatives: (5,2) -> 10
        self.binom  = np.array([math.comb(n,   i) for i in range(n + 1)], dtype=float)
        self.binom1 = np.array([math.comb(n-1, i) for i in range(n)],     dtype=float)
        self.binom2 = np.array([math.comb(n-2, i) for i in range(n-1)],   dtype=float)



    # -- public interface (t-parametric) --------------------------------------────────
 
    def y(self, t):
        """y(t) = H(t) or more practical H(t(d)) -  scalar or array t in [0, 1]."""
        t, s = self._wrap(t)
        return self._unwrap(self._eval(t, self.Wy), s)

    def dydx(self, t):
        """dy/dx(t) - first derivative w.r.t. x, scalar or array t."""
        t, s = self._wrap(t)
        return self._unwrap(
            self._eval1(t, self.Wy) / self._eval1(t, self.Wx), s)

    def d2ydx2(self, t):
        """d2y/dx2(t) - second derivative w.r.t. x, scalar or array t."""
        t, s = self._wrap(t)
        dxt  = self._eval1(t, self.Wx)
        dyt  = self._eval1(t, self.Wy)
        d2xt = self._eval2(t, self.Wx)
        d2yt = self._eval2(t, self.Wy)
        return self._unwrap((d2yt * dxt - dyt * d2xt) / dxt**3, s)

    # -- inverse: x -> t ------------------------------------------------------──────────

    def t(self, x):
        """Find t in [0,1] s.t. x(t) == x, via Brent's method. x is the distance -a <= d <= b

        Accepts scalar or 1-D array.  Each call to brentq solves one nonlinear equation.
        """
        x, scalar = self._wrap(x)
        result = np.array([scipy.optimize.brentq(lambda tt: float(self._eval(np.array([tt]), self.Wx)[0]) - xi, 0.0, 1.0) for xi in x])
        return self._unwrap(result, scalar)
    
    # -- control point construction -------------------------------------------────

    def _build_Wx(self) -> np.ndarray:
        a, b, g = self.a, self.b, self.gamma
        if self.order == 5:
            c = (a - b) / 30.0
            return np.array([-a, c - g, c - g, c + g, c + g, b])
        else:                              # order == 3
            c = (a - b) / 6.0
            return np.array([-a, c - g, c + g, b])

    def _build_Wy(self) -> np.ndarray:
        if self.order == 5:
            return np.array([1., 1., 1., 0., 0., 0.])
        else:
            return np.array([1., 1., 0., 0.])

    # -- Bernstein evaluators -------------------------------------------------────

    def _eval(self, t: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Bezier curve value at parameter array t."""
        n, C = self.order, self.binom
        r = np.zeros_like(t)
        for i in range(n + 1):
            r += C[i] * t**i * (1.0 - t)**(n - i) * P[i]
        return r

    def _eval1(self, t: np.ndarray, P: np.ndarray) -> np.ndarray:
        """First parametric derivative d/dt."""
        n, C1 = self.order, self.binom1
        dP = np.diff(P)
        r = np.zeros_like(t)
        for i in range(n):
            r += C1[i] * t**i * (1.0 - t)**(n - 1 - i) * dP[i]
        return n * r

    def _eval2(self, t: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Second parametric derivative d2/dt2."""
        n, C2 = self.order, self.binom2
        d2P = np.diff(np.diff(P))
        r = np.zeros_like(t)
        for i in range(n - 1):
            r += C2[i] * t**i * (1.0 - t)**(n - 2 - i) * d2P[i]
        return n * (n - 1) * r

    # -- scalar/array helpers -------------------------------------------------────

    @staticmethod
    def _wrap(t):
        t = np.asarray(t, dtype=float)
        scalar = t.ndim == 0
        return np.atleast_1d(t), scalar

    @staticmethod
    def _unwrap(r, scalar):
        return float(r[0]) if scalar else r


# -- optimal gamma ------------------------------------------------------------────────

def find_gamma(order: int, a: float, b: float,
               n_grid: int = 500, n_t: int = 1000) -> float:
    """Return gamma* = argmin_{gamma feasible} max_t |d2y/dx2(t; gamma)|.

    Feasibility: dx/dt > 0 at all n_t equidistant parameter values.
    Search: exhaustive grid of n_grid candidates in [-2.25*a, 2.25*a].

    Parameters
    ----------
    order : 3 or 5
    a     : left half-width (left endpoint is -a)
    b     : right endpoint, b >= a
    n_grid: number of gamma candidates
    n_t   : number of t-samples for the max evaluation
    """
    T = np.linspace(0.0, 1.0, n_t)
    best_gamma, best_val = 0.0, np.inf
    g_max = 2.25 * a
    for g in np.linspace(-g_max, g_max, n_grid):
        bz = Bezier(order, a, b, g)
        # skip when for current ga dx/dt <= 0
        if not np.all(bz._eval1(T, bz.Wx) > 1e-9):
            continue
        val = float(np.max(np.abs(bz.d2ydx2(T))))
        if val < best_val:
            best_val, best_gamma = val, g
    return best_gamma
