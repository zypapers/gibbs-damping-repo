"""Contextual Stochastic Block Model generator for this study.

Balanced k-block SBM + optional cSBM Gaussian features. The score is GNN-free and uses only the
graph + labels, so this generator controls and verifies the GRAPH SPECTRUM and the LABEL SPECTRAL
ENERGY; features are generated for later GNN training.

Spectral structure of a balanced 2-block SBM (normalized Laplacian L = I - D^{-1/2} A D^{-1/2}):
  - a BULK around lambda=1 (width ~ 2/sqrt(avg_degree));
  - a community SIGNAL outlier eigenvector (the +-1 block indicator) at
        lambda_signal ~ 1 - (p_in - p_out)/(p_in + p_out) = 1 - (2h - 1) = 2(1 - h),
    where h = p_in/(p_in+p_out) is the edge homophily.
So in a plain 2-block SBM, lambda_signal and homophily are coupled by construction -- decorrelating
them needs richer structure. fp64 throughout; reproducible from `seed`.
"""
from __future__ import annotations
import os, sys
import numpy as np


def generate_sbm(n, homophily, avg_degree, seed, n_blocks=2):
    """Balanced n_blocks SBM. For n_blocks=2, (p_in,p_out) set from (homophily, avg_degree):
    p_in+p_out = 2*avg_degree/n,  p_in/(p_in+p_out) = homophily.
    Returns dict with A (adjacency), comm (block labels), p_in, p_out."""
    rng = np.random.default_rng(seed)
    sizes = [n // n_blocks] * n_blocks
    sizes[0] += n - sum(sizes)
    comm = np.concatenate([np.full(s, b) for b, s in enumerate(sizes)]).astype(np.int64)

    # 2-block closed form; k>2 uses the same p_in on the diagonal, p_out off-diagonal.
    p_sum = 2.0 * avg_degree / n                      # expected within+cross prob budget (2-block)
    p_in = homophily * p_sum
    p_out = (1.0 - homophily) * p_sum

    same = comm[:, None] == comm[None, :]
    P = np.where(same, p_in, p_out)
    U = rng.random((n, n))
    A = (U < P).astype(np.float64)
    A = np.triu(A, 1)
    A = A + A.T                                        # symmetric, no self-loops
    return {"A": A, "comm": comm, "p_in": float(p_in), "p_out": float(p_out),
            "n": n, "homophily": float(homophily), "avg_degree": float(avg_degree), "seed": seed}


def csbm_features(comm, feat_dim, feat_snr, seed):
    """cSBM Gaussian features: X = sqrt(feat_snr/n) * b u^T + noise, b=+-1 (2-block sign)."""
    rng = np.random.default_rng(seed + 10_000)
    n = comm.shape[0]
    b = (2 * (comm == comm[0]).astype(np.float64) - 1.0) if comm.max() == 1 else (comm - comm.mean())
    u = rng.standard_normal(feat_dim); u /= np.linalg.norm(u)
    X = np.sqrt(feat_snr / n) * np.outer(b, u) + rng.standard_normal((n, feat_dim)) / np.sqrt(feat_dim)
    return X


def band_edge_label(w, U, a, b, taper=0.0, seed=0, binary=True):
    """Engineered band-edge label: a node signal built from the graph's eigenvectors with spectral
    energy in [a,b] (sharp edges at a,b unless `taper`>0 gives a smooth cosine roll-off of that
    width). Community labels have SMOOTH spectral edges and do not ring; band-edge labels create the
    sharp implied transition the score/ringing theory needs (the synthetic-grid label design; the
    real-task ringing question is separate). binary=True returns 2-class sign labels.

    Returns (y, mask) where y is (n,) continuous or {0,1}, mask selects the band eigenvectors.
    """
    w = np.asarray(w, float)
    mask = (w >= a) & (w <= b)
    lam = w[mask]
    wt = np.ones(lam.shape[0])
    if taper > 0:
        left = 0.5 * (1 - np.cos(np.clip((lam - a) / taper, 0, 1) * np.pi))
        right = 0.5 * (1 - np.cos(np.clip((b - lam) / taper, 0, 1) * np.pi))
        wt = left * right
    coeff = wt * np.random.default_rng(seed).standard_normal(lam.shape[0])
    y_cont = U[:, mask] @ coeff
    if binary:
        return (y_cont > np.median(y_cont)).astype(np.int64), mask
    return y_cont, mask


def spectrum_and_label_energy(A, comm, center=True):
    """fp64 normalized-Laplacian spectrum + per-eigenvalue label spectral energy of the community
    labels. Returns (w, U, e) with w ascending in [0,2]. (Not on the paper's reproduction path.)"""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "score"))
    import graph_spectrum as gs
    L = gs.normalized_laplacian(A)
    w, U = gs.eig_fp64(L)
    e = gs.label_spectral_energy(U, comm, center=center)
    return w, U, e


def edge_homophily(A, comm):
    """Fraction of edges connecting same-community nodes."""
    iu = np.triu_indices_from(A, 1)
    edges = A[iu] > 0
    same = comm[iu[0]] == comm[iu[1]]
    return float(np.sum(edges & same) / max(np.sum(edges), 1))


def label_dirichlet_energy(w, e):
    """y^T L y / y^T y = sum_i lambda_i e_i / sum_i e_i  (label smoothness = mean spectral position)."""
    tot = float(e.sum())
    return float(np.sum(w * e) / tot) if tot > 0 else 0.0
