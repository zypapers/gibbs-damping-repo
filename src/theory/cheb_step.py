"""Core Chebyshev-projection machinery for the Gibbs ringing theory.

Spectrum convention (fixed throughout): the normalized-Laplacian spectrum is
lambda in [0, 2]. A spectral filter response h(lambda) is approximated by a
degree-K polynomial. We study the *truncated Chebyshev projection* -- the
L2(w)-optimal partial sum with Chebyshev weight w(x) = 1/sqrt(1 - x^2). Its
oscillation at a jump discontinuity is the Gibbs phenomenon (truncation ringing),
which is the pathology studied here. (This is NOT the Runge phenomenon --
that is an *interpolation-node* artifact on smooth targets, a distinct pathology.)

Affine map between the spectral domain and the Chebyshev domain [-1, 1]:

        x = lambda - 1        (so lambda = x + 1),      lambda in [0,2] <-> x in [-1,1]

Under the further substitution x = cos(theta), theta in [0, pi], the degree-K
Chebyshev projection of a function equals the degree-K Fourier *cosine* partial
sum of g(theta) = f(cos theta). Hence every classical Gibbs result transfers
exactly. For the ideal low-pass step

        h(lambda) = 1 if lambda < lambda0 else 0

with  x0 = lambda0 - 1  and  theta0 = arccos(x0), the projection coefficients are
(derivation in stage_1_theory_score/task_1_1_ringing_band_theory.md):

        c_0 = 1 - theta0/pi
        c_k = -(2/(pi k)) sin(k theta0),     k = 1..K

and the degree-K approximation is p_K(x) = sum_{k=0}^K c_k T_k(x), evaluated
in theta as S_K(theta) = c_0 + sum_{k=1}^K c_k cos(k theta).

Everything is fp64.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# domain maps  (lambda in [0,2]  <->  x in [-1,1]  <->  theta in [0,pi])
# ---------------------------------------------------------------------------

def lam_to_x(lam):
    return np.asarray(lam, dtype=np.float64) - 1.0

def x_to_lam(x):
    return np.asarray(x, dtype=np.float64) + 1.0

def lam_to_theta(lam):
    """theta = arccos(lambda - 1); lambda in [0,2] -> theta in [0,pi]."""
    x = np.clip(lam_to_x(lam), -1.0, 1.0)
    return np.arccos(x)

def theta_to_lam(theta):
    return 1.0 + np.cos(np.asarray(theta, dtype=np.float64))


# ---------------------------------------------------------------------------
# closed-form Chebyshev-projection coefficients of an ideal step
# ---------------------------------------------------------------------------

def step_cheb_coeffs(lambda0: float, K: int, kind: str = "lowpass") -> np.ndarray:
    """Closed-form degree-K Chebyshev projection coefficients c[0..K] of a step.

    kind='lowpass'  : h = 1 for lambda < lambda0, else 0   (keeps low frequencies)
    kind='highpass' : h = 1 for lambda > lambda0, else 0   (= 1 - lowpass)
    """
    if not (0.0 < lambda0 < 2.0):
        raise ValueError("lambda0 must be interior to (0,2) for an interior jump")
    theta0 = float(np.arccos(lambda0 - 1.0))
    k = np.arange(0, K + 1, dtype=np.float64)
    c = np.empty(K + 1, dtype=np.float64)
    c[0] = 1.0 - theta0 / np.pi
    c[1:] = -(2.0 / (np.pi * k[1:])) * np.sin(k[1:] * theta0)
    if kind == "lowpass":
        return c
    elif kind == "highpass":
        # highpass = 1 - lowpass: negate all, add 1 to the constant term
        c = -c
        c[0] += 1.0
        return c
    else:
        raise ValueError("kind must be 'lowpass' or 'highpass'")


def step_cheb_coeffs_dct(lambda0: float, K: int, N: int = 200_000,
                         kind: str = "lowpass") -> np.ndarray:
    """Independent coefficient path: Gauss-Chebyshev (DCT-type) discrete projection.

    Uses nodes x_j = cos((j+1/2)pi/N), j=0..N-1. Converges to the true projection
    coefficients as N -> inf (error O(1/N) because the target is discontinuous).
    This is a DIFFERENT code path from step_cheb_coeffs, used only to cross-check it.
    """
    j = np.arange(N, dtype=np.float64)
    theta_nodes = (j + 0.5) * np.pi / N
    x_nodes = np.cos(theta_nodes)
    lam_nodes = x_nodes + 1.0
    if kind == "lowpass":
        f = (lam_nodes < lambda0).astype(np.float64)
    else:
        f = (lam_nodes > lambda0).astype(np.float64)
    c = np.empty(K + 1, dtype=np.float64)
    c[0] = f.mean()
    for kk in range(1, K + 1):
        c[kk] = (2.0 / N) * np.sum(f * np.cos(kk * theta_nodes))
    return c


# ---------------------------------------------------------------------------
# evaluate the degree-K partial sum
# ---------------------------------------------------------------------------

def partial_sum_theta(coeffs: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """S_K(theta) = c0 + sum_{k>=1} c_k cos(k theta), vectorized over theta."""
    theta = np.asarray(theta, dtype=np.float64)
    k = np.arange(1, coeffs.shape[0], dtype=np.float64)
    # matrix [len(theta) x K] of cos(k theta); fp64
    M = np.cos(np.outer(theta, k))
    return coeffs[0] + M @ coeffs[1:]


def partial_sum_lambda(coeffs: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Evaluate the same polynomial at spectral points lambda in [0,2]."""
    return partial_sum_theta(coeffs, lam_to_theta(lam))


# ---------------------------------------------------------------------------
# overshoot / undershoot measurement near the jump
# ---------------------------------------------------------------------------

def measure_overshoot(lambda0: float, K: int, kind: str = "lowpass",
                      n_lobes: int = 30, nwin: int = 8000) -> dict:
    """Peak overshoot above the top of the step and peak undershoot below its floor.

    Scans a fine theta window of half-width n_lobes * (pi/K) on each side of the
    jump theta0 (the first Gibbs lobe sits at theta0 + pi/K, well inside).
    Returns overshoot = max(S)-1 (high side) and undershoot = min(S)-0 (low side),
    plus the lambda locations of the extrema.
    """
    coeffs = step_cheb_coeffs(lambda0, K, kind=kind)
    theta0 = float(np.arccos(lambda0 - 1.0))
    w = n_lobes * np.pi / K

    # For kind='lowpass' the step in theta rises 0->1 as theta increases past
    # theta0, so the overshoot (>1) is just ABOVE theta0 and undershoot (<0) just
    # below. For highpass it is mirrored; measure_overshoot handles both by
    # taking the true max and min over a symmetric window around theta0.
    hi = np.linspace(theta0, min(theta0 + w, np.pi), nwin)
    lo = np.linspace(max(theta0 - w, 0.0), theta0, nwin)
    S_hi = partial_sum_theta(coeffs, hi)
    S_lo = partial_sum_theta(coeffs, lo)
    S_all = np.concatenate([S_lo, S_hi])
    over = float(S_all.max() - 1.0)
    under = float(S_all.min() - 0.0)
    i_over = int(S_all.argmax())
    i_under = int(S_all.argmin())
    theta_all = np.concatenate([lo, hi])
    return {
        "lambda0": lambda0, "K": K, "kind": kind,
        "overshoot": over, "undershoot": under,
        "lambda_at_overshoot": float(theta_to_lam(theta_all[i_over])),
        "lambda_at_undershoot": float(theta_to_lam(theta_all[i_under])),
        "theta0": theta0,
    }


# Gibbs-Wilbraham constant: overshoot fraction of the jump height.
def gibbs_constant() -> float:
    """G = Si(pi)/pi - 1/2 = 0.0894898722...  (fraction of the jump height)."""
    from scipy.special import sici
    Si_pi, _ = sici(np.pi)
    return float(Si_pi / np.pi - 0.5)
