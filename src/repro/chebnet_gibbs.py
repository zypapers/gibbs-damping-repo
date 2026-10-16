"""ChebNet-with-Gibbs (ChebGibbsNet ablation, Eq. 12) reproduction + the absorption test.

Two readings of Eq. 12  Z = sigma( sum_k g_{k,K} T_k(Ltilde) Z W ):
  share_W=True  : ONE shared W across orders -> coefficients are FIXED (all 1, times g_k) -> the
                  damping is NON-absorbable (it reweights a fixed polynomial filter). This is the
                  reading under which Table 2's damping effect and the high-K collapse make sense.
  share_W=False : per-order W_k (standard ChebNet, Defferrard 2016) -> g_k is absorbed into the free
                  W_k -> damping should be ~inert (our absorption claim, tested empirically here).

Chebyshev propagation on the normalized Laplacian: Ltilde = 2L/lmax - I; for the normalized Laplacian
lmax<=2, we use lmax=2 => Ltilde = L - I = -D^{-1/2} A D^{-1/2} (matches ChebNet).

Reproduction target: ChebGibbsNet Table 2 (K=2). Cora w/o G 78.39, w/J 75.86, w/La 73.08 (damping
HURTS on homophily); PubMed w/o G 67.77, w/J 73.98, w/La 75.60 (damping HELPS).
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from damping_cgn import taper_cgn


def sym_norm_adj(edge_index, num_nodes, device):
    """Ahat = D^{-1/2} A D^{-1/2} (no self loop); returns dense for simplicity (small graphs)."""
    A = torch.zeros(num_nodes, num_nodes, device=device)
    A[edge_index[0], edge_index[1]] = 1.0
    A = ((A + A.t()) > 0).float()
    d = A.sum(1)
    dinv = torch.where(d > 0, d.pow(-0.5), torch.zeros_like(d))
    return dinv[:, None] * A * dinv[None, :]


class ChebGibbsLayer(nn.Module):
    def __init__(self, cin, cout, K, g, share_W):
        super().__init__()
        self.K, self.share_W = K, share_W
        self.register_buffer("g", torch.tensor(g, dtype=torch.float32))
        if share_W:
            self.lin = nn.Linear(cin, cout)
        else:
            self.lins = nn.ModuleList([nn.Linear(cin, cout, bias=(k == 0)) for k in range(K + 1)])

    def forward(self, x, Ahat):
        # Ltilde x = -Ahat x  (lmax=2). Chebyshev recurrence in Ltilde.
        Tx = [x]                                  # T_0 = x
        if self.K >= 1:
            Tx.append(-(Ahat @ x))                # T_1 = Ltilde x
        for k in range(2, self.K + 1):
            Tx.append(-2.0 * (Ahat @ Tx[-1]) - Tx[-2])   # 2 Ltilde T_{k-1} - T_{k-2}
        if self.share_W:
            acc = sum(self.g[k] * Tx[k] for k in range(self.K + 1))
            return self.lin(acc)
        return sum(self.lins[k](self.g[k] * Tx[k]) for k in range(self.K + 1))


class ChebGibbsNet(nn.Module):
    def __init__(self, cin, hid, cout, K, g, share_W, dropout):
        super().__init__()
        self.l1 = ChebGibbsLayer(cin, hid, K, g, share_W)
        self.l2 = ChebGibbsLayer(hid, cout, K, g, share_W)
        self.dropout = dropout

    def forward(self, x, Ahat):
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.l1(x, Ahat))
        x = F.dropout(x, self.dropout, self.training)
        return self.l2(x, Ahat)


def run(dataset="Cora", damping="none", K=2, m=1, share_W=True, hid=64, dropout=0.5,
        lr=1e-2, wd=5e-4, epochs=200, patience=30, seeds=(0, 1, 2, 3, 4), device=None):
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "cache", "planetoid")
    ds = Planetoid(root, dataset, transform=T.NormalizeFeatures())
    d = ds[0].to(device)
    Ahat = sym_norm_adj(d.edge_index, d.num_nodes, device)
    g = taper_cgn(damping, K, m=m)
    accs = []
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        model = ChebGibbsNet(ds.num_features, hid, ds.num_classes, K, g, share_W, dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        best_val, best_test, bad = 0.0, 0.0, 0
        for ep in range(epochs):
            model.train(); opt.zero_grad()
            out = model(d.x, Ahat)
            F.cross_entropy(out[d.train_mask], d.y[d.train_mask]).backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                pred = model(d.x, Ahat).argmax(1)
                va = (pred[d.val_mask] == d.y[d.val_mask]).float().mean().item()
                te = (pred[d.test_mask] == d.y[d.test_mask]).float().mean().item()
            if va > best_val:
                best_val, best_test, bad = va, te, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        accs.append(best_test)
    return 100 * np.mean(accs), 100 * np.std(accs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Cora"])
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"ChebNet-Gibbs reproduction (K={args.K}, Lanczos m={args.m}, {len(seeds)} seeds, {dev})")
    print("Table 2 targets: Cora 78.39/75.86/73.08  CiteSeer 68.43/68.37/67.38  PubMed 67.77/73.98/75.60")
    for share_W in (True, False):
        tag = "shared-W (fixed coeffs, Eq.12 literal)" if share_W else "per-k W_k (standard ChebNet -> absorbable)"
        print(f"\n=== {tag} ===")
        print(f"  {'dataset':>9} | {'none':>12} {'jackson':>12} {'lanczos':>12} | damping effect")
        for dsname in args.datasets:
            r = {dmp: run(dsname, dmp, K=args.K, m=args.m, share_W=share_W, seeds=seeds, device=dev)
                 for dmp in ("none", "jackson", "lanczos")}
            eff = f"J {r['jackson'][0]-r['none'][0]:+.1f}  La {r['lanczos'][0]-r['none'][0]:+.1f}"
            print(f"  {dsname:>9} | " + " ".join(f"{r[d][0]:5.2f}±{r[d][1]:4.2f}" for d in ("none","jackson","lanczos"))
                  + f" | {eff}")
