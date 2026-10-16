"""Spectral primitives for the label-spectral-energy score.

Conventions (fixed throughout):
  - Normalized Laplacian  L = I - D^{-1/2} A D^{-1/2},  spectrum in [0, 2].
  - fp64 throughout; eigen-solves via numpy.linalg.eigh on the symmetrized L.
  - Eigenvector sign fixed deterministically (largest-|entry| made positive) for
    reproducibility. NOTE: within a degenerate eigenspace the individual eigenvectors are
    basis-arbitrary, so the SCORE only ever uses rotation-invariant quantities (energy summed
    over an eigenvalue, Parseval totals) -- never a single raw eigenvector.

The cycle graph C_n is the analytic anchor: eigenvalues lambda_k = 1 - cos(2 pi k / n) and the
eigenvectors are the Fourier modes, so a polynomial filter p(L) acts on eigenvalue lambda_i by
multiplication by p(lambda_i) -- the discrete-graph realization of the Chebyshev/Fourier Gibbs
theory above (a step-approximating p_K rings on C_n exactly as derived there).
"""
from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# graphs
# ---------------------------------------------------------------------------
def cycle_adjacency(n: int) -> np.ndarray:
    A = np.zeros((n, n), dtype=np.float64)
    idx = np.arange(n)
    A[idx, (idx + 1) % n] = 1.0
    A[idx, (idx - 1) % n] = 1.0
    return A


def cycle_closed_form_eigenvalues(n: int) -> np.ndarray:
    """lambda_k = 1 - cos(2 pi k / n), k = 0..n-1 (unsorted; k and n-k are degenerate)."""
    k = np.arange(n, dtype=np.float64)
    return 1.0 - np.cos(2.0 * np.pi * k / n)


# ---------------------------------------------------------------------------
# spectrum
# ---------------------------------------------------------------------------
def normalized_laplacian(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    deg = A.sum(axis=1)
    dinv = np.zeros_like(deg)
    nz = deg > 0
    dinv[nz] = 1.0 / np.sqrt(deg[nz])
    L = np.eye(A.shape[0]) - (dinv[:, None] * A * dinv[None, :])
    return 0.5 * (L + L.T)          # symmetrize away fp asymmetry


def eig_fp64(L: np.ndarray):
    """Return (eigenvalues sorted ascending in [0,2], eigenvectors) in fp64,
    with a deterministic eigenvector sign convention."""
    w, U = np.linalg.eigh(np.asarray(L, dtype=np.float64))
    w = np.clip(w, 0.0, 2.0)
    # deterministic sign: largest-magnitude entry positive
    j = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[j, np.arange(U.shape[1])])
    signs[signs == 0] = 1.0
    U = U * signs[None, :]
    return w, U


# ---------------------------------------------------------------------------
# label spectral energy (graph Fourier transform of the labels)
# ---------------------------------------------------------------------------
def label_spectral_energy(U: np.ndarray, Y: np.ndarray, center: bool = True) -> np.ndarray:
    """Per-eigenvalue label energy  e_i = sum_c (u_i^T y_c)^2.

    Y: (n,) integer class labels  OR  (n, C) indicator/one-hot columns.
    center=True removes each class column's mean (drops the trivial lambda=0 / DC component,
    which carries class balance, not discriminative spectral structure). This is a DESIGN
    choice.
    Returns e of shape (n,), aligned to the columns of U (i.e., to its eigenvalues).
    """
    Y = np.asarray(Y)
    if Y.ndim == 1:
        classes = np.unique(Y)
        Ymat = np.zeros((Y.shape[0], classes.size), dtype=np.float64)
        for c, cls in enumerate(classes):
            Ymat[:, c] = (Y == cls).astype(np.float64)
    else:
        Ymat = Y.astype(np.float64)
    if center:
        Ymat = Ymat - Ymat.mean(axis=0, keepdims=True)
    C = U.T @ Ymat                    # (n, n_classes) GFT coefficients
    return (C ** 2).sum(axis=1)       # (n,)


def total_energy(Y: np.ndarray, center: bool = True) -> float:
    """Spatial-domain total energy sum_c ||y_c (centered)||^2 (for Parseval checks)."""
    Y = np.asarray(Y)
    if Y.ndim == 1:
        classes = np.unique(Y)
        Ymat = np.zeros((Y.shape[0], classes.size), dtype=np.float64)
        for c, cls in enumerate(classes):
            Ymat[:, c] = (Y == cls).astype(np.float64)
    else:
        Ymat = Y.astype(np.float64)
    if center:
        Ymat = Ymat - Ymat.mean(axis=0, keepdims=True)
    return float((Ymat ** 2).sum())


# ---------------------------------------------------------------------------
# eigenvalue density (density of states)
# ---------------------------------------------------------------------------
def spectral_density(eigenvalues: np.ndarray, grid: np.ndarray, bandwidth: float = 0.02) -> np.ndarray:
    """Gaussian-KDE density of states rho(lambda) on `grid`, normalized to integrate to 1
    over the grid. (Exact-eigenvalue path.)"""
    ev = np.asarray(eigenvalues, dtype=np.float64)
    g = np.asarray(grid, dtype=np.float64)
    d = (g[:, None] - ev[None, :]) / bandwidth
    dens = np.exp(-0.5 * d * d).sum(axis=1) / (bandwidth * np.sqrt(2 * np.pi) * ev.size)
    area = np.trapezoid(dens, g)
    return dens / area if area > 0 else dens
