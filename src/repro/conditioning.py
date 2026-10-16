"""Conditioning analysis: is the KPM taper literally a preconditioner? (data-only, no training)

Wang & Zhang (2022): for a linear polynomial filter, GD convergence is governed by the condition number
of the Hessian H_{k1,k2} = sum_i f(mu_i) T_{k1}(mu_i) T_{k2}(mu_i), where {mu_i} are the eigenvalues of
the shift operator S = A~ and f is the input (feature) spectral density. Applying a fixed taper g
reparameterizes the basis to {g_k T_k}, giving H_g = diag(g) H diag(g). We compute kappa(H) vs kappa(H_g)
for the KPM tapers and an anti-low-pass control. If a KPM (decaying) taper lowers kappa and the
anti-low-pass raises it, the taper acts as an order-graded PRECONDITIONER -- the conditioning channel the
paper attributes the gain to, measured directly and without training.
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from damping_cgn import taper_cgn


def cheb_T(K, x):                       # T_0..T_K at points x in [-1,1]; returns (len(x), K+1)
    th = np.arccos(np.clip(x, -1, 1))
    return np.stack([np.cos(k * th) for k in range(K + 1)], axis=1)


def hessian_kappa(dataset, K=10):
    import torch
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T
    from gpr_gnn import gcn_norm_adj
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "cache", "planetoid")
    ds = Planetoid(root, dataset, transform=T.NormalizeFeatures())
    d = ds[0]
    n = d.num_nodes
    A = gcn_norm_adj(d.edge_index, n, "cpu").numpy()          # S = A~ (dense, cpu)
    mu, U = np.linalg.eigh(A)                                  # eigenvalues of the shift operator, in [-1,1]
    f = (U.T @ d.x.numpy()) ** 2                               # feature energy per eigenvector
    fdens = f.sum(1)                                           # signal spectral density f(mu_i)
    Tm = cheb_T(K, mu)                                         # (n, K+1)
    H = Tm.T @ (fdens[:, None] * Tm)                           # H_{k1,k2} = sum_i f_i T_k1 T_k2
    def kappa(g):
        Hg = (g[:, None] * H) * g[None, :]
        w = np.linalg.eigvalsh(Hg)
        w = w[w > w.max() * 1e-14]                             # drop numerical zeros
        return float(w.max() / w.min())
    k = np.arange(K + 1, dtype=float)
    tapers = {"none": np.ones(K + 1), "jackson": taper_cgn("jackson", K),
              "lanczos": taper_cgn("lanczos", K, m=3), "increasing(anti)": 0.6 ** (K - k)}
    return {name: kappa(g) for name, g in tapers.items()}


if __name__ == "__main__":
    print("Hessian condition number kappa (data-only) with vs without the taper, K=10")
    print("preconditioner test: KPM (decaying) should LOWER kappa; anti-low-pass should RAISE it\n")
    out = {}
    print(f"  {'dataset':>9} | {'none':>12} {'jackson':>12} {'lanczos':>12} {'increasing':>12}")
    for ds in ["Cora", "CiteSeer", "PubMed"]:
        k = hessian_kappa(ds, K=10)
        out[ds] = k
        print(f"  {ds:>9} | {k['none']:12.2e} {k['jackson']:12.2e} {k['lanczos']:12.2e} {k['increasing(anti)']:12.2e}")
        print(f"  {'':>9} |  (ratio vs none) {k['jackson']/k['none']:11.3f} {k['lanczos']/k['none']:11.3f} {k['increasing(anti)']/k['none']:11.3f}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "..", "results", "conditioning_v001.json"), "w"), indent=2)
    print("\n  saved results/conditioning_v001.json")
