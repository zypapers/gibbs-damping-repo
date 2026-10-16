"""LOW-PASS vs KPM: is ChebGibbsNet's damping gain GENERIC low-pass conditioning, or Gibbs-specific?

ChebGibbsNet's deployed filter Z = sum_k w_k g_k T_k(L~) H has w_k FREE (Eq.13), so the taper g_k is
representationally ABSORBED and cannot change the reachable filter. NOTE the model below (ChebFree)
is NOT that deployed filter: it uses a weight MATRIX per Chebyshev order and two propagation layers
with a ReLU between them, so results here speak to trained polynomial GNNs generally, not to the
decoupled scalar-coefficient architecture. On PubMed the gain (+3.6) is not reproduced by matching the initialization alone, which points to the
per-order forward scaling applied throughout training
acting as an effective-per-order-learning-rate / conditioning prior. The question: is the
specific Jackson/Lanczos (KPM/Gibbs) SHAPE what matters, or do generic decaying (low-pass) per-order
profiles reproduce it? If generic low-pass profiles match Jackson/Lanczos and an INCREASING profile
hurts, the gain is generic low-pass conditioning, which the "Gibbs suppression" attribution does
not account for.

Forward per-order profiles compared (all applied like the damping, w_k free, equal budget, 8 seeds):
  none        : undamped reference
  jackson/lanczos : the KPM/Gibbs tapers (what ChebGibbsNet claims is special)
  geometric   : 0.6^k  (generic low-pass, nothing to do with Gibbs)
  ppr         : 0.1*0.9^k (generic low-pass)
  increasing  : (k+1)  (anti-low-pass control -> should HURT if it's about low-pass)
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from damping_cgn import taper_cgn
from chebnet_gibbs import sym_norm_adj


def profile(name, K):
    k = np.arange(K + 1, dtype=float)
    if name == "none":       return np.ones(K + 1)
    if name in ("jackson", "lanczos", "fejer"): return taper_cgn(name, K, m=3)
    if name == "geometric":  return 0.6 ** k
    if name.startswith("geo"): return float(name[3:]) ** k        # geo0.5 etc. (tuned generic low-pass)
    if name.startswith("arr:"): return np.array([float(v) for v in name[4:].split(",")])  # explicit taper
    if name == "ppr":        g = 0.1 * 0.9 ** k; g[0] = 1.0; return g
    if name == "increasing": return (k + 1) / (K + 1)
    raise ValueError(name)


class ChebFree(nn.Module):
    def __init__(self, cin, hid, cout, K, gf, dropout):
        super().__init__()
        self.K, self.dropout = K, dropout
        self.register_buffer("gf", torch.tensor(gf, dtype=torch.float32))
        self.l1 = nn.ModuleList([nn.Linear(cin, hid, bias=(k == 0)) for k in range(K + 1)])
        self.l2 = nn.ModuleList([nn.Linear(hid, cout, bias=(k == 0)) for k in range(K + 1)])

    def _prop(self, x, Ahat, lins):
        Tx = [x] + ([-(Ahat @ x)] if self.K >= 1 else [])
        for k in range(2, self.K + 1):
            Tx.append(-2.0 * (Ahat @ Tx[-1]) - Tx[-2])
        return sum(lins[k](self.gf[k] * Tx[k]) for k in range(self.K + 1))

    def forward(self, x, Ahat):
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self._prop(x, Ahat, self.l1))
        x = F.dropout(x, self.dropout, self.training)
        return self._prop(x, Ahat, self.l2)


def run_one(dataset, prof, K=2, hid=64, dropout=0.5, lr=1e-2, wd=5e-4,
            epochs=200, patience=30, seeds=range(8), device=None):
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "cache", "planetoid")
    ds = Planetoid(root, dataset, transform=T.NormalizeFeatures())
    d = ds[0].to(device)
    Ahat = sym_norm_adj(d.edge_index, d.num_nodes, device)
    gf = profile(prof, K)
    accs = []
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        model = ChebFree(ds.num_features, hid, ds.num_classes, K, gf, dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        bv, bt, bad = 0.0, 0.0, 0
        for ep in range(epochs):
            model.train(); opt.zero_grad()
            F.cross_entropy(model(d.x, Ahat)[d.train_mask], d.y[d.train_mask]).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                pred = model(d.x, Ahat).argmax(1)
                va = (pred[d.val_mask] == d.y[d.val_mask]).float().mean().item()
                te = (pred[d.test_mask] == d.y[d.test_mask]).float().mean().item()
            if va > bv: bv, bt, bad = va, te, 0
            else:
                bad += 1
                if bad >= patience: break
        accs.append(bt)
    return np.array(accs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["PubMed", "CiteSeer"])
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    seeds = range(args.seeds)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    profs = ["none", "jackson", "lanczos", "geometric", "ppr", "increasing"]
    print(f"LOW-PASS vs KPM: is the damping gain Gibbs-specific or generic low-pass? K={args.K}, "
          f"{args.seeds} seeds, {dev}")
    print("forward per-order profile (w_k free, equal budget); gain = profile - none\n")
    for dsname in args.datasets:
        r = {p: run_one(dsname, p, K=args.K, seeds=seeds, device=dev) for p in profs}
        base = r["none"].mean()
        print(f"  {dsname}:")
        for p in profs:
            g = 100 * (r[p].mean() - base)
            tag = "  (KPM/Gibbs)" if p in ("jackson", "lanczos") else ("  (generic low-pass)" if p in ("geometric", "ppr") else ("  (anti-low-pass)" if p == "increasing" else ""))
            print(f"     {p:>11}: {100*r[p].mean():5.2f} ± {100*r[p].std():4.2f}  gain {g:+.2f}{tag}")
