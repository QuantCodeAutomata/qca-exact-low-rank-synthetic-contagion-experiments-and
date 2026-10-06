"""
graphon.py — Graphon/kernel contagion dynamics and structure-preserving
kernel truncation experiments (exp_3).

Graphon state equation:
    X_t(u) = x0(u) + μt + r t R_W(u) - ∫_I W(u,v) H_t(v) dv
where H_t(v) = ∫_0^t ℓ(X_s(v)) ds, discretised on midpoint grid.
"""

from __future__ import annotations

import numpy as np
from scipy.special import iv as bessel_i    # Modified Bessel functions I_m


# ---------------------------------------------------------------------------
# Midpoint grid
# ---------------------------------------------------------------------------

def midpoint_grid(n: int) -> np.ndarray:
    """Return n midpoints u_j = (j - 0.5)/n on [0,1], j=1,...,n."""
    return (np.arange(1, n + 1) - 0.5) / n


# ---------------------------------------------------------------------------
# Kernel definitions
# ---------------------------------------------------------------------------

def kernel_block_exponential(
    u: np.ndarray,
    v: np.ndarray,
    B: np.ndarray,
    c: float = 0.16,
    kappa: float = 10.0,
) -> np.ndarray:
    """W(u,v) = B_{ij} + c exp(-κ sin²(π(u-v))) on block I_i × I_j.

    Piecewise-smooth kernel from exp_3:
        B = [[0.38, 0.24], [0.31, 0.36]], c=0.16, κ=10.

    Parameters
    ----------
    u, v   : 1D arrays of evaluation points (broadcast-compatible)
    B      : (2,2) block matrix
    c      : amplitude of smooth component
    kappa  : decay rate in smooth component

    Returns
    -------
    W : (len(u), len(v)) kernel matrix
    """
    # Block assignments: block 0 = [0, 0.5), block 1 = [0.5, 1]
    u_block = (u >= 0.5).astype(int)   # (len(u),)
    v_block = (v >= 0.5).astype(int)   # (len(v),)

    # Build block part
    W_block = B[np.ix_(u_block, v_block)]   # (len(u), len(v))

    # Build smooth part: c exp(-κ sin²(π(u-v)))
    diff = u[:, None] - v[None, :]          # (len(u), len(v))
    W_smooth = c * np.exp(-kappa * np.sin(np.pi * diff) ** 2)

    return W_block + W_smooth


def kernel_directed_modes(
    u: np.ndarray,
    v: np.ndarray,
    sigma_m: np.ndarray,
    a_modes: list[np.ndarray],
    b_modes: list[np.ndarray],
    K_modes: int,
) -> np.ndarray:
    """Directed low-rank kernel W(u,v) = Σ_{m=1}^{K_modes} σ_m a_m(u) b_m(v).

    Parameters
    ----------
    u, v     : 1D arrays of evaluation points
    sigma_m  : (K_modes,) singular values
    a_modes  : list of (len(u),) receiver mode arrays
    b_modes  : list of (len(v),) sender mode arrays
    K_modes  : number of modes to include

    Returns
    -------
    W : (len(u), len(v)) kernel matrix
    """
    W = np.zeros((len(u), len(v)))
    for m in range(K_modes):
        W += sigma_m[m] * np.outer(a_modes[m], b_modes[m])
    return W


def compute_directed_modes(u: np.ndarray, M: int) -> tuple[np.ndarray, list, list]:
    """Compute σ_m, a_m(u), b_m(u) for the 30-mode directed kernel (exp_3).

    σ_m = 0.35 m^{-1.8}
    a_m(u) = max{0.05, 1 + 0.45 sin(2π m u) + 0.15 cos(2π(m+1)u)}
    b_m(v) = max{0.05, 1 + 0.40 cos(2π m v + 0.6) + 0.12 sin(2π(m+2)v)}

    Parameters
    ----------
    u : (n,) midpoint nodes
    M : number of modes

    Returns
    -------
    sigma_m : (M,) singular values
    a_modes : list of M arrays of shape (n,)
    b_modes : list of M arrays of shape (n,)
    """
    sigma_m = 0.35 * np.arange(1, M + 1) ** (-1.8)
    a_modes = []
    b_modes = []
    for m in range(1, M + 1):
        a_m = np.maximum(
            0.05,
            1 + 0.45 * np.sin(2 * np.pi * m * u)
            + 0.15 * np.cos(2 * np.pi * (m + 1) * u),
        )
        b_m = np.maximum(
            0.05,
            1 + 0.40 * np.cos(2 * np.pi * m * u + 0.6)
            + 0.12 * np.sin(2 * np.pi * (m + 2) * u),
        )
        a_modes.append(a_m)
        b_modes.append(b_m)
    return sigma_m, a_modes, b_modes


def kernel_trigonometric(
    u: np.ndarray,
    v: np.ndarray,
    M: int,
    c: float = 0.16,
    kappa: float = 5.0,
) -> np.ndarray:
    """Trigonometric smooth-part truncation W^{sm,(M)}(u,v).

    W^{sm,(M)}(u,v) = c exp(-κ) [I_0(κ) + 2 Σ_{m=1}^M I_m(κ) cos(2π m (u-v))]

    Parameters
    ----------
    u, v  : evaluation grids
    M     : number of modes
    c     : amplitude coefficient (0.16)
    kappa : Bessel parameter (5.0)

    Returns
    -------
    W_sm : (len(u), len(v)) matrix
    """
    prefactor = c * np.exp(-kappa)
    diff = u[:, None] - v[None, :]    # (len(u), len(v))
    W_sm = prefactor * bessel_i(0, kappa) * np.ones_like(diff)
    for m in range(1, M + 1):
        W_sm += 2 * prefactor * bessel_i(m, kappa) * np.cos(2 * np.pi * m * diff)
    return W_sm


# ---------------------------------------------------------------------------
# Graphon Euler solver
# ---------------------------------------------------------------------------

def simulate_graphon(
    x0: np.ndarray,
    W: np.ndarray,
    mu: float,
    r: float,
    T: float,
    dt: float,
    loss_fn,
    R_W: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Explicit Euler solver for graphon contagion dynamics.

    Update (using cumulative distress H_j):
        H_j^{n+1} = H_j^n + Δt ℓ(X_j^n)
        X_j^{n+1} = x0_j + μ t + r t R_W(u_j) - (1/n) Σ_m W_{jm} H_m^{n+1}

    Parameters
    ----------
    x0      : (n,) initial profile on midpoint grid
    W       : (n, n) precomputed kernel matrix W_{jm} = W(u_j, u_m)
    mu      : drift μ
    r       : imbalance parameter
    T       : horizon
    dt      : Euler step Δt
    loss_fn : callable(x) → array
    R_W     : (n,) precomputed imbalance profile R_W(u_j) = (1/n) Σ_m (W_{jm} - W_{mj})
              (supply if r ≠ 0, otherwise ignored)

    Returns
    -------
    times  : (n_steps+1,) times
    states : (n_steps+1, n) state trajectories
    """
    n = len(x0)
    n_steps = int(round(T / dt))
    times = np.linspace(0.0, T, n_steps + 1)

    if R_W is None:
        R_W = np.zeros(n)

    H = np.zeros(n)
    states = np.zeros((n_steps + 1, n))
    states[0] = x0.copy()

    for step in range(n_steps):
        X_now = states[step]
        H = H + dt * loss_fn(X_now)
        t_new = times[step + 1]
        # Graphon effect: (1/n) Σ_m W_{jm} H_m
        loss_effect = (W @ H) / n
        states[step + 1] = x0 + mu * t_new + r * t_new * R_W - loss_effect

    return times, states


def compute_R_W(W: np.ndarray) -> np.ndarray:
    """R_W(u_j) = (1/n) Σ_m (W_{jm} - W_{mj}) on midpoint grid."""
    n = W.shape[0]
    return (W.sum(axis=1) - W.sum(axis=0)) / n


# ---------------------------------------------------------------------------
# Initial profiles
# ---------------------------------------------------------------------------

def initial_profile_piecewise(u: np.ndarray) -> np.ndarray:
    """Piecewise-linear initial profile from exp_3 block kernel experiment.

    x0(u) = -0.18 + 1.95u        for u < 0.5
    x0(u) = 0.82 + 1.75(u-0.5)  for u ≥ 0.5
    """
    x0 = np.where(u < 0.5, -0.18 + 1.95 * u, 0.82 + 1.75 * (u - 0.5))
    return x0


def initial_profile_smooth(u: np.ndarray) -> np.ndarray:
    """Smooth initial profile from exp_3 directed truncation experiment.

    x0(u) = 0.13 + 0.008 sin(2π u) - 0.004 cos(6π u)
    """
    return 0.13 + 0.008 * np.sin(2 * np.pi * u) - 0.004 * np.cos(6 * np.pi * u)
