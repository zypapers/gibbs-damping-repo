"""GPR-GNN (Chien et al., ICLR 2021) -- the canonical PLAIN free-coefficient polynomial GNN, with a
CONFIGURABLE gamma init, to test the central de-risk claim: are spectral-GNN differences a
conditioning/INIT effect rather than a filter-class effect?

Model: H = MLP(X); Z = sum_{k=0}^K gamma_k P^k H ; P = D~^{-1/2}(A+I)D~^{-1/2} (GCN norm, self-loops);
gamma_k learnable. Init options:
  ppr   : gamma_k = alpha (1-alpha)^k  (default GPR-GNN, a low-pass-ish prior)
  lowpass: gamma_k = jackson/lanczos taper  (an order-graded low-pass conditioning init)
  const : gamma_k = 1/(K+1)             (flat)
  random: gamma_k ~ N(0, 1/(K+1))
Same free-coefficient CLASS regardless of init (absorption) -- so any accuracy difference across inits
is a pure conditioning/optimization effect. If a low-pass init reproduces the damping gain, the
"innovation" is conditioning, not filter class.
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from damping_cgn import taper_cgn


def gcn_norm_adj(edge_index, num_nodes, device):
    A = torch.zeros(num_nodes, num_nodes, device=device)
    A[edge_index[0], edge_index[1]] = 1.0
    A = ((A + A.t()) > 0).float()
    A = A + torch.eye(num_nodes, device=device)        # self-loops
    d = A.sum(1)
    dinv = torch.where(d > 0, d.pow(-0.5), torch.zeros_like(d))
    return dinv[:, None] * A * dinv[None, :]


def init_gamma(kind, K, alpha=0.1):
    k = np.arange(K + 1)
    if kind == "ppr":
        g = alpha * (1 - alpha) ** k; g[-1] = (1 - alpha) ** K
    elif kind == "jackson":
        g = taper_cgn("jackson", K)
    elif kind == "lanczos":
        g = taper_cgn("lanczos", K, m=3)
    elif kind == "const":
        g = np.ones(K + 1) / (K + 1)
    elif kind == "random":
        g = np.random.default_rng(0).standard_normal(K + 1) / np.sqrt(K + 1)
    else:
        raise ValueError(kind)
    return g.astype(np.float32)


class GPRGNN(nn.Module):
    def __init__(self, cin, hid, cout, K, gamma0, dropout):
        super().__init__()
        self.lin1 = nn.Linear(cin, hid)
        self.lin2 = nn.Linear(hid, cout)
        self.gamma = nn.Parameter(torch.tensor(gamma0, dtype=torch.float32))
        self.K, self.dropout = K, dropout

    def forward(self, x, P):
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, self.dropout, self.training)
        h = self.lin2(x)
        out = self.gamma[0] * h
        hk = h
        for k in range(1, self.K + 1):
            hk = P @ hk
            out = out + self.gamma[k] * hk
        return out


def run(dataset="Cora", init="ppr", K=10, hid=64, dropout=0.5, lr=1e-2, wd=5e-4,
        epochs=300, patience=50, seeds=(0, 1, 2, 3, 4), device=None):
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "cache", "planetoid")
    ds = Planetoid(root, dataset, transform=T.NormalizeFeatures())
    d = ds[0].to(device)
    P = gcn_norm_adj(d.edge_index, d.num_nodes, device)
    g0 = init_gamma(init, K)
    accs = []
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        model = GPRGNN(ds.num_features, hid, ds.num_classes, K, g0, dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        bv, bt, bad = 0.0, 0.0, 0
        for ep in range(epochs):
            model.train(); opt.zero_grad()
            out = model(d.x, P)
            F.cross_entropy(out[d.train_mask], d.y[d.train_mask]).backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                pred = model(d.x, P).argmax(1)
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
    print(f"GPR-GNN (plain free-coeff poly, K={args.K}, {len(seeds)} seeds, {dev}) — INIT is a pure")
    print("conditioning knob (same filter class). Does a low-pass init match the damping gain?")
    print(f"  {'dataset':>9} | " + " ".join(f"{i:>11}" for i in ("ppr", "jackson", "lanczos", "const", "random")))
    for dsname in args.datasets:
        r = {i: run(dsname, init=i, K=args.K, seeds=seeds, device=dev)
             for i in ("ppr", "jackson", "lanczos", "const", "random")}
        print(f"  {dsname:>9} | " + " ".join(f"{r[i][0]:5.2f}±{r[i][1]:4.2f}" for i in
                                               ("ppr", "jackson", "lanczos", "const", "random")))
