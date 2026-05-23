"""
bezier.py

Quintic Bézier transition function displayed as N equidistant piecewise-linear
segments, with exact derivatives (chain rule) shown in a lower panel.

Options / sliders:
  • b          – right endpoint
  • δ₁, δ₂    – individual x-control offsets       (mode δ₁,δ₂)
  • δ          – symmetric offset                  (mode δ)
  • auto       – minimises max|y''(x)| over valid δ (mode auto)
  • N          – number of equidistant segments

Workflow:
  1. Build the parametric Bézier from the control points at high resolution
     (2000 t-samples).
  2. Invert t→x: interpolate N+1 equidistant x-nodes in [X0, b].
  3. Draw the piecewise-linear polyline through those nodes.
  4. For each node, invert x→t and evaluate exact y', y'' via chain rule.

Color convention:  blue = x-monotone,  red = NOT x-monotone.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons
from matplotlib.lines import Line2D
from bezier_lut import lut_manager

# ── Constants ──────────────────────────────────────────────────────────────────
X0    = -1.0
YCTRL = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
B5    = np.array([1, 5, 10, 10, 5, 1])
B4    = np.array([1, 4,  6,  4, 1])
T     = np.linspace(0.0, 1.0, 2000)   # high-res parametric samples for inversion


# ── Bézier helpers ─────────────────────────────────────────────────────────────
def make_px(b, d1, d2):
    c = (1.0 - b) / 30.0
    return np.array([X0, c - d1, c - d2, c + d2, c + d1, b])


def bezier_xy(Px):
    x = np.zeros(len(T))
    y = np.zeros(len(T))
    for i in range(6):
        b = B5[i] * T**i * (1 - T)**(5 - i)
        x += b * Px[i]
        y += b * YCTRL[i]
    return x, y


B3 = np.array([1, 3, 3, 1])

# fixed y second-difference coefficients (YCTRL = [1,1,1,0,0,0])
_dY  = np.diff(YCTRL)           # [0, 0, -1, 0, 0]
_d2Y = np.diff(_dY)             # [0, -1,  1, 0]


def dxdt(Px):
    dP = np.diff(Px)
    dx = np.zeros(len(T))
    for i in range(5):
        dx += B4[i] * T**i * (1 - T)**(4 - i) * dP[i]
    return 5.0 * dx


def dydt():
    """dy/dt of the quintic Bézier (depends only on YCTRL, constant)."""
    dy = np.zeros(len(T))
    for i in range(5):
        dy += B4[i] * T**i * (1 - T)**(4 - i) * _dY[i]
    return 5.0 * dy


def d2xdt2(Px):
    d2P = np.diff(np.diff(Px))
    d2x = np.zeros(len(T))
    for i in range(4):
        d2x += B3[i] * T**i * (1 - T)**(3 - i) * d2P[i]
    return 20.0 * d2x


def d2ydt2():
    """d²y/dt² of the quintic Bézier (constant for fixed YCTRL)."""
    d2y = np.zeros(len(T))
    for i in range(4):
        d2y += B3[i] * T**i * (1 - T)**(3 - i) * _d2Y[i]
    return 20.0 * d2y


# pre-compute y-derivatives once (YCTRL never changes)
_DYDT  = dydt()
_D2YDT = d2ydt2()


def exact_derivs(Px):
    """Return (x, y', y'') arrays using the parametric chain rule."""
    xb, _  = bezier_xy(Px)
    xt     = dxdt(Px)
    x2t    = d2xdt2(Px)
    # chain rule
    dydx   = _DYDT / xt
    d2ydx2 = (_D2YDT * xt - _DYDT * x2t) / xt**3
    return xb, dydx, d2ydx2


def sample_equidistant(Px, n_seg):
    """Return (x_nodes, y_nodes) of n_seg+1 equidistant x-samples of the Bézier."""
    xb, yb = bezier_xy(Px)
    sort_idx = np.argsort(xb)
    xs, ys   = xb[sort_idx], yb[sort_idx]
    b        = Px[-1]
    x_nodes  = np.linspace(X0, b, n_seg + 1)
    y_nodes  = np.interp(x_nodes, xs, ys)
    return x_nodes, y_nodes


def cubic_hermite_y(x_vals, b):
    """Cubic Hermite smoothstep: y'=0 at both endpoints, no y''=0 constraint."""
    xi = (np.asarray(x_vals) - X0) / (b - X0)
    return 1.0 - 3.0 * xi**2 + 2.0 * xi**3


def auto_delta(b):
    """Find delta minimising max|y''(x)| (minimax curvature) — auto (y'')."""
    T_auto   = np.linspace(0.0, 1.0, 1000)
    # precompute dydt / d²ydt² on T_auto
    dY_auto  = np.diff(YCTRL)
    d2Y_auto = np.diff(dY_auto)
    B3_auto  = np.array([1, 3, 3, 1])

    def _dydt_at(T_):
        dy = np.zeros(len(T_))
        for i in range(5):
            dy += B4[i] * T_**i * (1 - T_)**(4 - i) * dY_auto[i]
        return 5.0 * dy

    def _d2ydt_at(T_):
        d2y = np.zeros(len(T_))
        for i in range(4):
            d2y += B3_auto[i] * T_**i * (1 - T_)**(3 - i) * d2Y_auto[i]
        return 20.0 * d2y

    dydt_a  = _dydt_at(T_auto)
    d2ydt_a = _d2ydt_at(T_auto)

    def _dxdt_at(Px, T_):
        dP = np.diff(Px)
        dx = np.zeros(len(T_))
        for i in range(5):
            dx += B4[i] * T_**i * (1 - T_)**(4 - i) * dP[i]
        return 5.0 * dx

    def _d2xdt_at(Px, T_):
        d2P = np.diff(np.diff(Px))
        d2x = np.zeros(len(T_))
        for i in range(4):
            d2x += B3_auto[i] * T_**i * (1 - T_)**(3 - i) * d2P[i]
        return 20.0 * d2x

    delta_grid = np.linspace(-2.25, 2.25, 500)
    best_d, best_val = 0.0, np.inf
    for d in delta_grid:
        Px = make_px(b, d, d)
        xt = _dxdt_at(Px, T_auto)
        if not np.all(xt > 1e-9):
            continue
        x2t   = _d2xdt_at(Px, T_auto)
        d2ydx = (d2ydt_a * xt - dydt_a * x2t) / xt**3
        val   = np.max(np.abs(d2ydx))
        if val < best_val:
            best_val, best_d = val, d
    return float(best_d)

def auto_delta_dy(b):
    """Find delta minimising max|y'(x)| — auto (y')."""
    T_auto  = np.linspace(0.0, 1.0, 1000)
    dY_auto = np.diff(YCTRL)
    B3_auto = np.array([1, 3, 3, 1])

    def _dydt_at(T_):
        dy = np.zeros(len(T_))
        for i in range(5):
            dy += B4[i] * T_**i * (1 - T_)**(4 - i) * dY_auto[i]
        return 5.0 * dy

    def _dxdt_at(Px, T_):
        dP = np.diff(Px);  dx = np.zeros(len(T_))
        for i in range(5):
            dx += B4[i] * T_**i * (1 - T_)**(4 - i) * dP[i]
        return 5.0 * dx

    dydt_a = _dydt_at(T_auto)
    delta_grid = np.linspace(-2.25, 2.25, 500)
    best_d, best_val = 0.0, np.inf
    for d in delta_grid:
        Px = make_px(b, d, d)
        xt = _dxdt_at(Px, T_auto)
        if not np.all(xt > 1e-9):
            continue
        dydx = dydt_a / xt
        val  = np.max(np.abs(dydx))   # minimise the worst-case slope magnitude
        if val < best_val:
            best_val, best_d = val, d
    return float(best_d)


B2     = np.array([1, 2, 1])
B1     = np.array([1, 1])
YCTRL3 = np.array([1.0, 1.0, 0.0, 0.0])
_dY3   = np.diff(YCTRL3)        # [ 0, -1,  0]
_d2Y3  = np.diff(_dY3)          # [-1,  1]


def make_px3(b, d):
    """Cubic x-control: symmetric δ, enforces x(0.5)=0 (factor 6)."""
    c3 = (1.0 - b) / 6.0
    return np.array([X0, c3 - d, c3 + d, b])


def bezier_xy3(Px):
    x = np.zeros(len(T));  y = np.zeros(len(T))
    for i in range(4):
        b = B3[i] * T**i * (1 - T)**(3 - i)
        x += b * Px[i];  y += b * YCTRL3[i]
    return x, y


def dxdt3(Px):
    dP = np.diff(Px);  dx = np.zeros(len(T))
    for i in range(3):
        dx += B2[i] * T**i * (1 - T)**(2 - i) * dP[i]
    return 3.0 * dx


def d2xdt23(Px):
    d2P = np.diff(np.diff(Px));  d2x = np.zeros(len(T))
    for i in range(2):
        d2x += B1[i] * T**i * (1 - T)**(1 - i) * d2P[i]
    return 6.0 * d2x


def _dydt3_fn():
    dy = np.zeros(len(T))
    for i in range(3):
        dy += B2[i] * T**i * (1 - T)**(2 - i) * _dY3[i]
    return 3.0 * dy


def _d2ydt23_fn():
    d2y = np.zeros(len(T))
    for i in range(2):
        d2y += B1[i] * T**i * (1 - T)**(1 - i) * _d2Y3[i]
    return 6.0 * d2y


_DYDT3  = _dydt3_fn()
_D2YDT3 = _d2ydt23_fn()


def exact_derivs3(Px):
    """Return (x, dy/dx, d²y/dx²) for the cubic Bézier via chain rule."""
    xb, _ = bezier_xy3(Px)
    xt    = dxdt3(Px);  x2t = d2xdt23(Px)
    dydx   = _DYDT3 / xt
    d2ydx2 = (_D2YDT3 * xt - _DYDT3 * x2t) / xt**3
    return xb, dydx, d2ydx2


def sample_equidistant3(Px, n_seg):
    xb, yb   = bezier_xy3(Px)
    sort_idx = np.argsort(xb)
    b        = Px[-1]
    x_nodes  = np.linspace(X0, b, n_seg + 1)
    y_nodes  = np.interp(x_nodes, xb[sort_idx], yb[sort_idx])
    return x_nodes, y_nodes


def auto_delta3(b):
    """Find cubic delta minimising max|d²y/dx²| — auto (y'')."""
    T_a = np.linspace(0.0, 1.0, 1000)
    yt_a = np.zeros(len(T_a))
    for i in range(3):
        yt_a += B2[i] * T_a**i * (1 - T_a)**(2 - i) * _dY3[i]
    yt_a *= 3.0
    y2t_a = np.zeros(len(T_a))
    for i in range(2):
        y2t_a += B1[i] * T_a**i * (1 - T_a)**(1 - i) * _d2Y3[i]
    y2t_a *= 6.0

    def _xt(Px):
        dP = np.diff(Px);  dx = np.zeros(len(T_a))
        for i in range(3):
            dx += B2[i] * T_a**i * (1 - T_a)**(2 - i) * dP[i]
        return 3.0 * dx

    def _x2t(Px):
        d2P = np.diff(np.diff(Px));  d2x = np.zeros(len(T_a))
        for i in range(2):
            d2x += B1[i] * T_a**i * (1 - T_a)**(1 - i) * d2P[i]
        return 6.0 * d2x

    best_d, best_val = 0.0, np.inf
    for d in np.linspace(-2.25, 2.25, 500):
        Px = make_px3(b, d)
        xt = _xt(Px)
        if not np.all(xt > 1e-9):
            continue
        x2t = _x2t(Px)
        val = np.max(np.abs((y2t_a * xt - yt_a * x2t) / xt**3))
        if val < best_val:
            best_val, best_d = val, d
    return float(best_d)

def auto_delta3_dy(b):
    """Find cubic delta minimising max|y'(x)| — auto (y')."""
    T_a = np.linspace(0.0, 1.0, 1000)
    yt_a = np.zeros(len(T_a))
    for i in range(3):
        yt_a += B2[i] * T_a**i * (1 - T_a)**(2 - i) * _dY3[i]
    yt_a *= 3.0

    def _xt(Px):
        dP = np.diff(Px);  dx = np.zeros(len(T_a))
        for i in range(3):
            dx += B2[i] * T_a**i * (1 - T_a)**(2 - i) * dP[i]
        return 3.0 * dx

    best_d, best_val = 0.0, np.inf
    for d in np.linspace(-2.25, 2.25, 500):
        Px = make_px3(b, d)
        xt = _xt(Px)
        if not np.all(xt > 1e-9):
            continue
        val = np.max(np.abs(yt_a / xt))   # minimise the worst-case slope magnitude
        if val < best_val:
            best_val, best_d = val, d
    return float(best_d)


def compute_node_derivs(Px, xb, xn, deg):
    """Evaluate exact dy/dx and d²y/dx² at equidistant nodes by t-inversion."""
    sort_idx = np.argsort(xb)
    t_nodes  = np.interp(xn, xb[sort_idx], T[sort_idx])
    dxt  = np.zeros(len(t_nodes));  dyt  = np.zeros(len(t_nodes))
    d2xt = np.zeros(len(t_nodes));  d2yt = np.zeros(len(t_nodes))
    dP  = np.diff(Px);  d2P = np.diff(dP)
    if deg == 5:
        for i in range(5):
            b = B4[i] * t_nodes**i * (1 - t_nodes)**(4 - i)
            dxt += b * dP[i];  dyt += b * _dY[i]
        dxt *= 5.0;  dyt *= 5.0
        for i in range(4):
            b2 = B3[i] * t_nodes**i * (1 - t_nodes)**(3 - i)
            d2xt += b2 * d2P[i];  d2yt += b2 * _d2Y[i]
        d2xt *= 20.0;  d2yt *= 20.0
    else:  # cubic
        for i in range(3):
            b = B2[i] * t_nodes**i * (1 - t_nodes)**(2 - i)
            dxt += b * dP[i];  dyt += b * _dY3[i]
        dxt *= 3.0;  dyt *= 3.0
        for i in range(2):
            b2 = B1[i] * t_nodes**i * (1 - t_nodes)**(1 - i)
            d2xt += b2 * d2P[i];  d2yt += b2 * _d2Y3[i]
        d2xt *= 6.0;  d2yt *= 6.0
    return dyt / dxt, (d2yt * dxt - dyt * d2xt) / dxt**3


# ── δ-bounds helper ───────────────────────────────────────────────────────────
def min_dydx_over_delta(b, deg, n_delta=300):
    """For each δ in [-0.45, 0.45], compute min(dy/dx) of the symmetric Bézier.
    Returns (deltas, mins) where mins[i]=nan if curve is not x-monotone."""
    deltas = np.linspace(-2.25, 2.25, n_delta)
    mins   = np.full(n_delta, np.nan)
    for i, d in enumerate(deltas):
        if deg == 5:
            Px = make_px(b, d, d)
            xt = dxdt(Px)
            if not np.all(xt > 1e-9):
                continue
            mins[i] = np.min(_DYDT / xt)
        else:
            Px = make_px3(b, d)
            xt = dxdt3(Px)
            if not np.all(xt > 1e-9):
                continue
            mins[i] = np.min(_DYDT3 / xt)
    return deltas, mins


def max_abs_dydx_over_delta(b, deg, n_delta=300):
    """For each δ in [-0.45, 0.45], compute max|dy/dx| of the symmetric Bézier.
    Returns (deltas, vals) where vals[i]=nan if curve is not x-monotone."""
    deltas = np.linspace(-2.25, 2.25, n_delta)
    vals   = np.full(n_delta, np.nan)
    for i, d in enumerate(deltas):
        if deg == 5:
            Px = make_px(b, d, d)
            xt = dxdt(Px)
            if not np.all(xt > 1e-9):
                continue
            vals[i] = np.max(np.abs(_DYDT / xt))
        else:
            Px = make_px3(b, d)
            xt = dxdt3(Px)
            if not np.all(xt > 1e-9):
                continue
            vals[i] = np.max(np.abs(_DYDT3 / xt))
    return deltas, vals


def max_abs_d2ydx2_over_delta(b, deg, n_delta=300):
    """For each δ in [-0.45, 0.45], compute max|d²y/dx²| of the symmetric Bézier.
    Returns (deltas, vals) where vals[i]=nan if curve is not x-monotone."""
    deltas = np.linspace(-2.25, 2.25, n_delta)
    vals   = np.full(n_delta, np.nan)
    for i, d in enumerate(deltas):
        if deg == 5:
            Px = make_px(b, d, d)
            xt = dxdt(Px)
            if not np.all(xt > 1e-9):
                continue
            x2t = d2xdt2(Px)
            vals[i] = np.max(np.abs((_D2YDT * xt - _DYDT * x2t) / xt**3))
        else:
            Px = make_px3(b, d)
            xt = dxdt3(Px)
            if not np.all(xt > 1e-9):
                continue
            x2t = d2xdt23(Px)
            vals[i] = np.max(np.abs((_D2YDT3 * xt - _DYDT3 * x2t) / xt**3))
    return deltas, vals


# ── RAMP helpers ───────────────────────────────────────────────────────────────
def ramp(y, q):
    return y / (1.0 + q * (1.0 - y))

def ramp_deriv(y, dydx, q):
    """d/dx[ramp(y)] = (1+q)/(1+q(1-y))^2 * dy/dx"""
    return (1.0 + q) * dydx / (1.0 + q * (1.0 - y))**2

def ramp_deriv2(y, dydx, d2ydx, q):
    """d²/dx²[ramp(y)] = (1+q)*[2q/(1+q(1-y))^3*(dy/dx)^2 + d²y/dx²/(1+q(1-y))^2]"""
    denom = 1.0 + q * (1.0 - y)
    return (1.0 + q) * (2.0 * q / denom**3 * dydx**2 + d2ydx / denom**2)


# ── Figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 9))

ax       = fig.add_axes([0.07, 0.64, 0.82, 0.30])   # main plot (top)
ax_deriv = fig.add_axes([0.07, 0.36, 0.82, 0.24])   # derivatives (bottom)
ax_d2y   = ax_deriv.twinx()                          # y'' on right axis

# ── Initial state ──────────────────────────────────────────────────────────────
b0    = 3.0
N0    = 128
Px0   = make_px(b0, 0.0, 0.0)
xn0, yn0 = sample_equidistant(Px0, N0)

# background Bézier reference (light)
xb0, yb0 = bezier_xy(Px0)
(ln_ref,)    = ax.plot(xb0, yb0, color='lightsteelblue', lw=1.5,
                       ls='--', label='Bézier (reference)', zorder=1)
# cubic Bézier reference (shown only when b ≈ 0.2)
(ln_cubic,)  = ax.plot([], [], color='tomato', lw=1.5, ls=':',
                       label='cubic poly (b=1)', zorder=2)
_proxy_cubic = Line2D([0], [0], color='tomato', lw=1.5, ls=':')
# piecewise-linear foreground
(ln_fd,)     = ax.plot(xn0, yn0, color='steelblue', lw=2.5,
                       marker='o', ms=5, label='piecewise linear', zorder=3)
(ln_lut,)    = ax.plot([], [], color='purple', lw=2.0, ls=':', label='Bézier LUT', zorder=4)
# derivative LUT plots
(ln_lut_dy,) = ax_deriv.plot([], [], color='darkmagenta', lw=1.8, ls=':', label="y' LUT", zorder=4)
(ln_lut_d2y,) = ax_d2y.plot([], [], color='magenta', lw=1.8, ls=':', label="y'' LUT", zorder=4)
# control polygon
# quintic: green=outer δ₁ chain P0-P1-P4-P5, blue=inner δ₂ chain P1-P2-P3-P4
# cubic:   green=upper (P0-P1),                 blue=lower (P1-P2-P3)
(ln_cpoly_outer,) = ax.plot([Px0[0], Px0[1], Px0[4], Px0[5]],
                             [YCTRL[0], YCTRL[1], YCTRL[4], YCTRL[5]],
                             color='green',     lw=1.0, ls='--', alpha=0.5, zorder=2)
(ln_cpoly_inner,) = ax.plot(Px0[1:5],  YCTRL[1:5],  color='steelblue', lw=1.0, ls='--', alpha=0.5, zorder=2)
(ln_cpts_outer,)  = ax.plot([Px0[1], Px0[4]], [YCTRL[1], YCTRL[4]],
                             'o', color='green',     ms=6, alpha=0.7, label='ctrl pts (δ₁)', zorder=4)
(ln_cpts_inner,)  = ax.plot([Px0[2], Px0[3]], [YCTRL[2], YCTRL[3]],
                             'o', color='steelblue', ms=6, alpha=0.7, label='ctrl pts (δ₂)', zorder=4)

(pt_left,)   = ax.plot([X0], [1.0], 'g^', ms=10, zorder=6, label='endpoints')
(pt_right,)  = ax.plot([b0], [0.0], 'g^', ms=10, zorder=6)
(pt_mid,)    = ax.plot([0.0], [0.5], 'r*', ms=13, zorder=6, label='(0, 0.5)')

ax.axhline(0.5, color='gray', lw=0.5, ls=':')
ax.axvline(0.0, color='gray', lw=0.5, ls=':')
ax.set_ylabel('y')
ax.set_ylim(-0.10, 1.15)
# legend rebuilt dynamically in update()
ax.grid(True, alpha=0.4)
ax.tick_params(labelbottom=False)   # x labels on derivative panel instead
title = ax.set_title('')

# ── Derivative panel setup ─────────────────────────────────────────────────────
Px_init = make_px(b0, 0.0, 0.0)
xd0, dy0, d2y0 = exact_derivs(Px_init)

(ln_dy,)      = ax_deriv.plot(xd0, dy0,   color='darkorange',  lw=1.2, ls='--', label="y' exact", zorder=1)
(ln_d2y,)     = ax_d2y.plot(  xd0, d2y0,  color='forestgreen', lw=1.2, ls='--', label="y'' exact", zorder=1)
(ln_fd_dy,)   = ax_deriv.plot([], [], color='darkorange', lw=2.0,
                              marker='o', ms=5, ls='-', label="y' piecewise", zorder=3)
(ln_fd_d2y,)  = ax_d2y.plot(  [], [], color='forestgreen', lw=2.0,
                              marker='o', ms=5, ls='-', label="y'' piecewise", zorder=3)
# δ-bounds lines (hidden by default)
(ln_dbounds,)     = ax_deriv.plot([], [], color='steelblue', lw=2.0, label="min y'(δ)", zorder=4)
(ln_dbounds_cur,) = ax_deriv.plot([], [], color='crimson', lw=0, marker='D', ms=9,
                                  label='current δ', zorder=5)
ln_dbounds_zero = ax_deriv.axhline(0, color='gray', lw=0.8, ls=':', zorder=1)

ax_deriv.axhline(0, color='gray', lw=0.5, ls=':')
ax_deriv.axvline(0, color='gray', lw=0.5, ls=':')
ax_deriv.set_xlabel('x')
ax_deriv.set_ylabel("y'",  color='darkorange',  fontsize=10)
ax_d2y.set_ylabel(  "y''", color='forestgreen', fontsize=10)
ax_deriv.tick_params(axis='y', colors='darkorange')
ax_d2y.tick_params(  axis='y', colors='forestgreen')
ax_deriv.grid(True, alpha=0.3)
# legend rebuilt dynamically in update()

# ── Widgets ────────────────────────────────────────────────────────────────────
# Three radio groups placed side-by-side below the plots
ax_mode  = fig.add_axes([0.02, 0.18, 0.09, 0.08])
ax_mode.set_facecolor('lightyellow')
radio_mode = RadioButtons(ax_mode, ('smooth', 'piecewise'), active=0)

ax_deg  = fig.add_axes([0.13, 0.18, 0.09, 0.08])
ax_deg.set_facecolor('lightyellow')
radio_deg = RadioButtons(ax_deg, ('cubic', 'quintic'), active=1)

ax_shape  = fig.add_axes([0.24, 0.16, 0.09, 0.14])
ax_shape.set_facecolor('lightyellow')
radio = RadioButtons(ax_shape, ('δ₁, δ₂', 'δ', 'auto (y\u2032\u2032)', 'auto (y\u2032)'), active=2)

ax_sb  = fig.add_axes([0.35, 0.28, 0.60, 0.03])
ax_sd1 = fig.add_axes([0.35, 0.23, 0.60, 0.03])
ax_sd2 = fig.add_axes([0.35, 0.18, 0.60, 0.03])
ax_sd  = fig.add_axes([0.35, 0.23, 0.60, 0.03])
ax_sN  = fig.add_axes([0.35, 0.13, 0.60, 0.03])
ax_sq  = fig.add_axes([0.35, 0.08, 0.60, 0.03])
ax_chk  = fig.add_axes([0.02, 0.03, 0.10, 0.05])
ax_chk.set_facecolor('#eeeeff')
ax_chkd = fig.add_axes([0.14, 0.03, 0.13, 0.05])
ax_chkd.set_facecolor('#eeffee')
ax_lut  = fig.add_axes([0.29, 0.03, 0.12, 0.05])
ax_lut.set_facecolor('#ffeedd')

sl_b   = Slider(ax_sb,  'b',   1.00, 10.00, valinit=b0,  color='steelblue')
sl_d1  = Slider(ax_sd1, 'δ₁', -2.50,  2.50, valinit=0.0, color='darkorange')
sl_d2  = Slider(ax_sd2, 'δ₂', -2.50,  2.50, valinit=0.0, color='darkorange')
sl_d   = Slider(ax_sd,  'δ',  -2.50,  2.50, valinit=0.0, color='darkorange')
sl_N   = Slider(ax_sN,  'N (segments)', 2, 128, valinit=N0, valstep=1,
                color='mediumseagreen')
sl_q   = Slider(ax_sq,  'q (RAMP)', 0.0, 10.0, valinit=0.0, color='mediumpurple')
check_ramp   = CheckButtons(ax_chk,  ['RAMP'],     [False])
check_dbounds = CheckButtons(ax_chkd, ['δ-bounds'], [False])
check_lut    = CheckButtons(ax_lut,  ['LUT'],      [False])
ramp_on    = [False]
dbounds_on = [False]
lut_on     = [False]

info = fig.text(0.02, 0.01, '', fontsize=8, color='gray')

# ── Initial slider visibility ──────────────────────────────────────────────────
ax_sd1.set_visible(False)
ax_sd2.set_visible(False)
ax_sd.set_visible(True)


# ── Helpers ────────────────────────────────────────────────────────────────────
_last_auto_delta = [0.0]   # remembers the most recent auto-computed delta

def get_d1d2(deg=5):
    mode = radio.value_selected
    if deg == 3:
        if mode == 'auto (y\u2032\u2032)':
            d = auto_delta3(sl_b.val)
            _last_auto_delta[0] = d
        elif mode == 'auto (y\u2032)':
            d = auto_delta3_dy(sl_b.val)
            _last_auto_delta[0] = d
        else:
            d = sl_d.val
        return d, d
    if mode == 'δ₁, δ₂':
        return sl_d1.val, sl_d2.val
    elif mode == 'δ':
        return sl_d.val, sl_d.val
    elif mode == 'auto (y\u2032\u2032)':
        d = auto_delta(sl_b.val)
        _last_auto_delta[0] = d
        return d, d
    else:   # auto (y')
        d = auto_delta_dy(sl_b.val)
        _last_auto_delta[0] = d
        return d, d


# ── Update ─────────────────────────────────────────────────────────────────────
def update(_):
    b    = sl_b.val
    N    = int(sl_N.val)
    deg  = 3 if radio_deg.value_selected == 'cubic' else 5
    d1, d2 = get_d1d2(deg)

    if deg == 5:
        Px    = make_px(b, d1, d2)
        yctrl = YCTRL
        mono  = bool(np.all(dxdt(Px) > 0))
        xb, yb          = bezier_xy(Px)
        xn, yn          = sample_equidistant(Px, N)
        xd, dydx, d2ydx = exact_derivs(Px)
    else:
        Px    = make_px3(b, d1)
        yctrl = YCTRL3
        mono  = bool(np.all(dxdt3(Px) > 0))
        xb, yb          = bezier_xy3(Px)
        xn, yn          = sample_equidistant3(Px, N)
        xd, dydx, d2ydx = exact_derivs3(Px)

    # ── apply RAMP if active ───────────────────────────────────────────────────
    q = sl_q.val
    use_ramp = ramp_on[0] and q > 0
    if use_ramp:
        yb_at_xd   = np.interp(xd, xb, yb)
        yb_show    = ramp(yb, q)
        yn_show    = ramp(yn, q)
        dydx_show  = ramp_deriv(yb_at_xd, dydx, q)
        d2ydx_show = ramp_deriv2(yb_at_xd, dydx, d2ydx, q)
    else:
        yb_show    = yb
        yn_show    = yn
        dydx_show  = dydx
        d2ydx_show = d2ydx

    lut_valid = False
    x_lut = y_lut = dydx_lut = d2ydx_lut = np.array([])
    if lut_on[0]:
        if deg == 5 and radio.value_selected == 'δ₁, δ₂' and abs(d1 - d2) > 1e-9:
            lut_valid = False
        else:
            gamma = d1
            try:
                x_lut, y_lut, dydx_lut, d2ydx_lut = lut_manager.get_lut(
                    deg, 1.0, b, gamma, num_pts=max(2000, N * 10))
            except Exception:
                x_lut = y_lut = dydx_lut = d2ydx_lut = np.array([])
            else:
                if use_ramp:
                    y_lut_show = ramp(y_lut, q)
                    dydx_lut_show = ramp_deriv(y_lut, dydx_lut, q)
                    d2ydx_lut_show = ramp_deriv2(y_lut, dydx_lut, d2ydx_lut, q)
                    y_lut, dydx_lut, d2ydx_lut = y_lut_show, dydx_lut_show, d2ydx_lut_show
                lut_valid = x_lut.size > 0

    ln_ref.set_data(xb, yb_show)
    ln_fd.set_data(xn, yn_show)
    ln_fd.set_color('steelblue' if mono else 'crimson')
    ln_lut.set_data(x_lut, y_lut)
    ln_lut.set_visible(lut_on[0] and lut_valid)
    # control polygon
    if deg == 5:
        # outer (δ₁): P0-P1-P4-P5 chain
        ln_cpoly_outer.set_data([Px[0], Px[1], Px[4], Px[5]],
                                [yctrl[0], yctrl[1], yctrl[4], yctrl[5]])
        ln_cpoly_inner.set_data(Px[1:5], yctrl[1:5])
        ln_cpts_outer.set_data([Px[1], Px[4]], [yctrl[1], yctrl[4]])
        ln_cpts_inner.set_data(Px[2:4],  yctrl[2:4])
        ln_cpts_inner.set_label('ctrl pts (δ₂)')
    else:  # cubic: single δ, upper=green P0-P1, lower=blue P1-P2-P3
        ln_cpoly_outer.set_data(Px[:2], yctrl[:2])
        ln_cpoly_inner.set_data(Px[1:], yctrl[1:])
        ln_cpts_outer.set_data([Px[1]], [yctrl[1]])
        ln_cpts_inner.set_data(Px[1:3], yctrl[1:3])
        ln_cpts_inner.set_label('ctrl pts (δ)')
    pt_right.set_data([b], [0.0])

    piecewise = (radio_mode.value_selected == 'piecewise')

    # top panel: piecewise shows reference dashed + polyline;
    #            smooth shows the Bézier as the solid main curve
    ln_ref.set_linestyle('--' if piecewise else '-')
    ln_ref.set_linewidth(1.5 if piecewise else 2.5)
    ln_ref.set_color('lightsteelblue' if piecewise else ('steelblue' if mono else 'crimson'))
    ln_fd.set_visible(piecewise)
    ax_sN.set_visible(piecewise)

    # rebuild top-panel legend — piecewise entries only in piecewise mode
    _common = [_proxy_cubic, ln_cpoly_outer, ln_cpts_outer, ln_cpts_inner, pt_left, pt_mid]
    _common_labels = ['cubic poly (b=1)', 'ctrl polygon',
                      ln_cpts_outer.get_label(), ln_cpts_inner.get_label(),
                      'endpoints', '(0, 0.5)']
    if piecewise:
        legend_items = [ln_ref, ln_fd] + ([ln_lut] if lut_on[0] and lut_valid else []) + _common
        legend_labels = ['Bézier (reference)', 'piecewise linear'] + (['Bézier LUT'] if lut_on[0] and lut_valid else []) + _common_labels
        ax.legend(legend_items, legend_labels, loc='upper right', fontsize=9)
    else:
        legend_items = [ln_ref] + ([ln_lut] if lut_on[0] and lut_valid else []) + _common
        legend_labels = ['Bézier'] + (['Bézier LUT'] if lut_on[0] and lut_valid else []) + _common_labels
        ax.legend(legend_items, legend_labels, loc='upper right', fontsize=9)

    # cubic Bézier reference: only shown when b is close to a=1
    if abs(b - 1.0) < 0.05:
        x_cub = np.linspace(X0, 1.0, 400)
        ln_cubic.set_data(x_cub, cubic_hermite_y(x_cub, 1.0))
        ln_cubic.set_visible(True)
    else:
        ln_cubic.set_visible(False)

    # ── δ-bounds mode: replace derivative panel ──────────────────────────────
    _dmode = radio.value_selected
    show_dbounds = dbounds_on[0] and _dmode in ('auto (y\u2032\u2032)', 'auto (y\u2032)', 'δ')

    # normal derivative lines visibility
    for ln in [ln_dy, ln_d2y, ln_fd_dy, ln_fd_d2y, ln_lut_dy, ln_lut_d2y]:
        ln.set_visible(not show_dbounds)
    ax_d2y.set_visible(not show_dbounds)
    ln_dbounds.set_visible(show_dbounds)
    ln_dbounds_cur.set_visible(show_dbounds)

    if show_dbounds:
        _dy_mode  = (_dmode == 'auto (y\u2032)')
        _d2y_mode = (_dmode == 'auto (y\u2032\u2032)')
        if _dy_mode:
            deltas, mins = max_abs_dydx_over_delta(b, deg)
            _ylabel = "min max |y'|"
        elif _d2y_mode:
            deltas, mins = max_abs_d2ydx2_over_delta(b, deg)
            _ylabel = "min max |y''|"
        else:
            deltas, mins = max_abs_d2ydx2_over_delta(b, deg)
            _ylabel = "min max |y''|"
        ln_dbounds.set_data(deltas, mins)
        cur_d = d1   # auto or δ mode: d1 == d2
        cur_min = np.interp(cur_d, deltas, np.where(np.isnan(mins), -999, mins))
        ln_dbounds_cur.set_data([cur_d], [cur_min] if not np.isnan(cur_min) else [[]])
        # axis labels / limits
        ax_deriv.set_xlabel('δ')
        ax_deriv.set_ylabel(_ylabel, color='steelblue')
        ax_deriv.tick_params(axis='y', colors='steelblue')
        valid = mins[~np.isnan(mins)]
        if len(valid):
            _cap = 100.0 if not _dy_mode else np.inf
            _cur_val = cur_min if (not np.isnan(cur_min) and cur_min > -999) else 0.0
            vmax = max(min(valid.max(), _cap), _cur_val)
            pad = (vmax - valid.min()) * 0.15 + 0.05
            ax_deriv.set_ylim(valid.min() - pad, vmax + pad)
        ax_deriv.set_xlim(-2.35, 2.35)
        ax.set_xlim(min(X0, min(Px)) - 0.15, max(b, max(Px)) + 0.15)
        cur_label = (f'auto(y\u2032\u2032) δ={cur_d:.4f}' if _dmode == 'auto (y\u2032\u2032)'
                     else f'auto(y\u2032) δ={cur_d:.4f}' if _dmode == 'auto (y\u2032)'
                     else f'δ={cur_d:.4f}')
        ax_deriv.legend([ln_dbounds, ln_dbounds_cur],
                        [_ylabel, cur_label],
                        loc='upper right', fontsize=9)
    else:
        # restore normal labels/colors
        ax_deriv.set_xlabel('x')
        ax_deriv.set_ylabel("y'",  color='darkorange',  fontsize=10)
        ax_deriv.tick_params(axis='y', colors='darkorange')

        # clip extreme endpoint spikes before scaling (dx/dt ≈ 0 at t=0,1)
        skip = len(T) // 40
        dydx_inner  = dydx_show[skip:-skip]
        d2ydx_inner = d2ydx_show[skip:-skip]

        margin = 0.12
        def sym_lims(v):
            m = np.max(np.abs(v)) * (1 + margin)
            return -m, m

        ln_dy.set_data(xd, dydx_show)
        ln_d2y.set_data(xd, d2ydx_show)

        ln_dy.set_linestyle('--' if piecewise else '-')
        ln_dy.set_linewidth(1.2 if piecewise else 2.0)
        ln_d2y.set_linestyle('--' if piecewise else '-')
        ln_d2y.set_linewidth(1.2 if piecewise else 2.0)

        node_dy, node_d2y = compute_node_derivs(Px, xb, xn, deg)
        if use_ramp:
            node_dy  = ramp_deriv(yn, node_dy, q)
            node_d2y = ramp_deriv2(yn, node_dy, node_d2y, q)
        ln_fd_dy.set_data(xn, node_dy)
        ln_fd_d2y.set_data(xn, node_d2y)
        ln_fd_dy.set_visible(piecewise)
        ln_fd_d2y.set_visible(piecewise)
        ln_lut_dy.set_data(x_lut, dydx_lut)
        ln_lut_d2y.set_data(x_lut, d2ydx_lut)
        ln_lut_dy.set_visible(lut_on[0] and not show_dbounds and lut_valid)
        ln_lut_d2y.set_visible(lut_on[0] and not show_dbounds and lut_valid)

        if piecewise:
            leg_lines  = [ln_dy, ln_d2y, ln_fd_dy, ln_fd_d2y] + ([ln_lut_dy, ln_lut_d2y] if lut_on[0] and lut_valid else [])
            leg_labels = ["y' exact", "y'' exact", "y' piecewise", "y'' piecewise"] + (["y' LUT", "y'' LUT"] if lut_on[0] and lut_valid else [])
        else:
            leg_lines  = [ln_dy, ln_d2y] + ([ln_lut_dy, ln_lut_d2y] if lut_on[0] and lut_valid else [])
            leg_labels = ["y'", "y''"] + (["y' LUT", "y'' LUT"] if lut_on[0] and lut_valid else [])
        ax_deriv.legend(leg_lines, leg_labels, loc='upper right', fontsize=9)

        ax_deriv.set_ylim(*sym_lims(dydx_inner))
        ax_d2y.set_ylim(*sym_lims(d2ydx_inner))
        xlim = (min(X0, min(Px)) - 0.15, max(b, max(Px)) + 0.15)
        ax.set_xlim(*xlim)
        ax_deriv.set_xlim(*xlim)

    deg_str = 'cubic' if deg == 3 else 'quintic'
    mode = radio.value_selected
    c_val = (0.2 - b) / 6.0 if deg == 3 else (0.2 - b) / 30.0
    if deg == 3:
        d_str = (f'δ_auto(y\u2032\u2032)={d1:.4f}' if mode == 'auto (y\u2032\u2032)'
                 else f'δ_auto(y\u2032)={d1:.4f}' if mode == 'auto (y\u2032)'
                 else f'δ={d1:.4f}')
    else:
        d_str = (f'δ₁={d1:.4f}, δ₂={d2:.4f}' if mode == 'δ₁, δ₂'
                 else f'δ={d1:.4f}' if mode == 'δ'
                 else f'δ_auto(y\u2032\u2032)={d1:.4f}' if mode == 'auto (y\u2032\u2032)'
                 else f'δ_auto(y\u2032)={d1:.4f}')
    ramp_str = f'  |  RAMP q={q:.2f}' if use_ramp else ''
    n_str = f'  |  N={N} segments' if piecewise else ''
    title.set_text(
        f'{deg_str}  |  b={b:.3f}{n_str}  |  '
        f'{"\u2713 monotone" if mono else "\u2717 NOT monotone"}  |  {d_str}  |  c={c_val:.4f}{ramp_str}'
    )

    info.set_text(
        'Dashed light-blue = Bézier reference  |  '
        'Blue markers = piecewise-linear nodes (x-monotone ✓)  |  '
        'Red = NOT monotone ✗'
    )

    fig.canvas.draw_idle()


def on_mode(label):
    ax_sd1.set_visible(label == 'δ₁, δ₂')
    ax_sd2.set_visible(label == 'δ₁, δ₂')
    ax_sd.set_visible(label == 'δ')
    if label == 'δ' and _last_auto_delta[0] != 0.0:
        sl_d.set_val(_last_auto_delta[0])
    update(None)
    fig.canvas.draw_idle()


def on_display_mode(label):
    update(None)
    fig.canvas.draw_idle()


_radio_btn_sizes = None   # cached original sizes

def on_deg(label):
    global _radio_btn_sizes
    is_cubic = (label == 'cubic')
    # cache original sizes on first call
    if _radio_btn_sizes is None:
        _radio_btn_sizes = list(radio._buttons.get_sizes())
    # hide/show the δ₁,δ₂ option (index 0) by zeroing its marker size
    sizes = list(_radio_btn_sizes)
    sizes[0] = 0 if is_cubic else _radio_btn_sizes[0]
    radio._buttons.set_sizes(sizes)
    radio.labels[0].set_visible(not is_cubic)
    if is_cubic and radio.value_selected == 'δ₁, δ₂':
        radio.set_active(1)   # fall back to δ
    # sync slider visibility
    mode = radio.value_selected
    ax_sd1.set_visible(not is_cubic and mode == 'δ₁, δ₂')
    ax_sd2.set_visible(not is_cubic and mode == 'δ₁, δ₂')
    ax_sd.set_visible(mode == 'δ')
    update(None)
    fig.canvas.draw_idle()


sl_b.on_changed(update)
sl_d1.on_changed(update)
sl_d2.on_changed(update)
sl_d.on_changed(update)
sl_N.on_changed(update)
sl_q.on_changed(update)

def on_ramp(_):
    ramp_on[0] = check_ramp.get_status()[0]
    update(None)

def on_dbounds(_):
    dbounds_on[0] = check_dbounds.get_status()[0]
    update(None)

def on_lut(_):
    lut_on[0] = check_lut.get_status()[0]
    update(None)

check_ramp.on_clicked(on_ramp)
check_dbounds.on_clicked(on_dbounds)
check_lut.on_clicked(on_lut)
radio.on_clicked(on_mode)
radio_deg.on_clicked(on_deg)
radio_mode.on_clicked(on_display_mode)
update(None)

plt.show()
