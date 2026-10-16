"""Unified decoupled polynomial-filter GNN for the controlled MECHANISM DECOMPOSITION.

One model, three orthogonal knobs, so a factorial ablation at EQUAL budget separates the mechanisms:
  basis      in {mono, cheb, bern}  : all span degree-K polynomials in S -> SAME filter class for FREE
                                       coeffs (absorption); differ only in CONDITIONING.
  constraint in {free, nonneg}       : nonneg (ReLU on coeffs, BernNet-style) is a GENUINE filter-class
                                       restriction -> a real (non-absorbed) delta, the control contrast.
  init       in {ppr, flat, damped, random} : a pure conditioning/optimization knob at fixed class.

S = D~^{-1/2} A~ D~^{-1/2} (GCN norm, self-loops), spectrum ~[-1,1]. H = MLP(X) (decoupled). Filter
Z = sum_k c_k B_k(S) H.  This subsumes GPR-GNN (mono/free/ppr), a Chebyshev filter (cheb/free), the
absorbed Gibbs taper (cheb/free with a damped init), and BernNet (bern/nonneg).

Decomposition target: delta(method) over the baseline (mono/free/ppr) = basis-conditioning + init +
constraint(filter-class) + weight-decay. For absorbed 'innovations' the filter-class term ~ 0.
"""
from __future__ import annotations
import os, sys, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from damping_cgn import taper_cgn
from gpr_gnn import gcn_norm_adj


def basis_apply(basis, S, H, K):
    """Return list [B_0 H, ..., B_K H] for the chosen basis (all degree-K polynomials in S)."""
    if basis == "mono":
        out = [H]
        for k in range(1, K + 1):
            out.append(S @ out[-1])
        return out
    if basis == "cheb":
        out = [H, S @ H] if K >= 1 else [H]
        for k in range(2, K + 1):
            out.append(2.0 * (S @ out[-1]) - out[-2])
        return out
    if basis == "bern":
        # b_k = (1/2^K) C(K,k) (I+S)^{K-k} (I-S)^k ; apply to H
        # precompute (I-S)^k H and (I+S)^j H by recurrence
        ImS = [H]
        for k in range(1, K + 1):
            ImS.append(ImS[-1] - S @ ImS[-1])            # (I-S)^k H
        out = []
        for k in range(K + 1):
            v = ImS[k]
            for _ in range(K - k):
                v = v + S @ v                            # multiply by (I+S)
            coef = math.comb(K, k) / (2 ** K)
            out.append(coef * v)
        return out
    raise ValueError(basis)


def init_coeffs(kind, K, alpha=0.1):
    k = np.arange(K + 1)
    if kind == "ppr":
        c = alpha * (1 - alpha) ** k; c[-1] = (1 - alpha) ** K
    elif kind == "flat":
        c = np.ones(K + 1) / (K + 1)
    elif kind == "damped":
        c = taper_cgn("jackson", K) / (K + 1)             # order-graded low-pass init
    elif kind == "random":
        c = np.random.default_rng(0).standard_normal(K + 1) / np.sqrt(K + 1)
    else:
        raise ValueError(kind)
    return c.astype(np.float32)


class PolyFilterNet(nn.Module):
    def __init__(self, cin, hid, cout, K, basis, constraint, init, dropout, taper_arr=None, freeze=False):
        super().__init__()
        self.lin1, self.lin2 = nn.Linear(cin, hid), nn.Linear(hid, cout)
        self.theta = nn.Parameter(torch.tensor(init_coeffs(init, K), dtype=torch.float32),
                                  requires_grad=not freeze)
        g = np.ones(K + 1) if taper_arr is None else np.asarray(taper_arr, float)
        self.register_buffer("g", torch.tensor(g, dtype=torch.float32))     # fixed per-order taper
        self.K, self.basis, self.constraint, self.dropout = K, basis, constraint, dropout

    def coeffs(self):
        c = F.relu(self.theta) if self.constraint == "nonneg" else self.theta
        return c * self.g

    def forward(self, x, S):
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, self.dropout, self.training)
        h = self.lin2(x)
        B = basis_apply(self.basis, S, h, self.K)
        c = self.coeffs()
        out = c[0] * B[0]
        for k in range(1, self.K + 1):
            out = out + c[k] * B[k]
        return out


def run(dataset, basis="mono", constraint="free", init="ppr", K=10, hid=64, dropout=0.5,
        lr=1e-2, wd=5e-4, epochs=300, patience=50, seeds=(0, 1, 2, 3, 4), taper_arr=None, freeze=False, device=None):
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "cache", "planetoid")
    ds = Planetoid(root, dataset, transform=T.NormalizeFeatures())
    d = ds[0].to(device)
    S = gcn_norm_adj(d.edge_index, d.num_nodes, device)
    accs = []
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        model = PolyFilterNet(ds.num_features, hid, ds.num_classes, K, basis, constraint, init, dropout,
                              taper_arr=taper_arr, freeze=freeze).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        bv, bt, bad = 0.0, 0.0, 0
        for ep in range(epochs):
            model.train(); opt.zero_grad()
            out = model(d.x, S)
            F.cross_entropy(out[d.train_mask], d.y[d.train_mask]).backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                pred = model(d.x, S).argmax(1)
                va = (pred[d.val_mask] == d.y[d.val_mask]).float().mean().item()
                te = (pred[d.test_mask] == d.y[d.test_mask]).float().mean().item()
            if va > bv: bv, bt, bad = va, te, 0
            else:
                bad += 1
                if bad >= patience: break
        accs.append(bt)
    return 100 * np.mean(accs), 100 * np.std(accs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer", "PubMed"])
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # factorial: the mechanism cells. baseline = mono/free/ppr.
    cells = [
        ("baseline mono/free/ppr", dict(basis="mono", constraint="free", init="ppr")),
        ("cheb/free/ppr  (basis)", dict(basis="cheb", constraint="free", init="ppr")),
        ("bern/free/ppr  (basis)", dict(basis="bern", constraint="free", init="ppr")),
        ("bern/NONNEG/ppr (class)", dict(basis="bern", constraint="nonneg", init="ppr")),
        ("mono/free/flat  (init) ", dict(basis="mono", constraint="free", init="flat")),
        ("mono/free/damped(init) ", dict(basis="mono", constraint="free", init="damped")),
    ]
    print(f"Controlled mechanism decomposition (K={args.K}, {len(seeds)} seeds, {dev})")
    print("delta vs baseline isolates: basis=conditioning, NONNEG=filter-class, init=init/optim\n")
    for dsname in args.datasets:
        res = {name: run(dsname, K=args.K, seeds=seeds, device=dev, **kw) for name, kw in cells}
        base = res["baseline mono/free/ppr"][0]
        print(f"  {dsname}:")
        for name, _ in cells:
            m, sd = res[name]
            dtag = "" if name.startswith("baseline") else f"  (delta {m-base:+.2f})"
            print(f"     {name:>26}: {m:5.2f} ± {sd:4.2f}{dtag}")
