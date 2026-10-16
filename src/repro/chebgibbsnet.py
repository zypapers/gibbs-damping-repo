"""ChebGibbsNet (Zhang & Sun, arXiv:2412.01789, Eq. 13) reproduction.

  Y = Softmax( sum_{k=0}^K w_k g_{k,K} T_k(S) f_Theta(X) )
  - f_Theta: decoupled MLP applied to X FIRST (GPR-GNN style): Linear-SiLU-dropout-Linear.
  - S = A~ = D~^{-1/2}(A+I)D~^{-1/2} (homophily) or S = -A~ (heterophily). eigenvalues in ~[-1,1].
  - T_k(S): Chebyshev recurrence in S (no rescaling; A~ in [-1,1]).
  - w_k learnable, initialized to the Chebyshev-quadrature weights 1/(K+1) and 2/(K+1) as in the
    released code; g_{k,K} fixed (Jackson/Lanczos).

Reproduction target: match ChebGibbsNet Table 2 (ablation) and Table 6 (full) within a tolerance
fixed before the reproduction is quoted. Splits: the public Planetoid split is the default and is what
the reported table uses; `--split dense` gives dense random splits instead (their ~88% Cora is above
the public 20/class ceiling). HPs/splits/GSO-per-dataset follow the reproduction specification.
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from damping_cgn import taper_cgn
from gpr_gnn import gcn_norm_adj
from splits import split


class ChebGibbsNet(nn.Module):
    def __init__(self, cin, hid, cout, K, damping, m, gso_sign, dropout, taper_arr=None, freeze_w=False):
        super().__init__()
        self.lin1, self.lin2 = nn.Linear(cin, hid), nn.Linear(hid, cout)
        w0 = np.full(K + 1, 2.0 / (K + 1)); w0[0] = 1.0 / (K + 1)   # their init (Chebyshev quadrature)
        self.w = nn.Parameter(torch.tensor(w0, dtype=torch.float32), requires_grad=not freeze_w)
        if taper_arr is not None:                                  # custom forward taper (low-pass vs KPM)
            g = np.asarray(taper_arr, dtype=float)
        else:
            g = taper_cgn(damping, K, m=m) if damping != "none" else np.ones(K + 1)
        self.register_buffer("g", torch.tensor(g, dtype=torch.float32))
        self.K, self.gso_sign, self.dropout = K, float(gso_sign), dropout

    def forward(self, x, Ahat):
        x = F.dropout(x, self.dropout, self.training)
        x = F.silu(self.lin1(x))                               # paper: SiLU
        x = F.dropout(x, self.dropout, self.training)
        h = self.lin2(x)                                       # f_Theta(X)
        S = self.gso_sign * Ahat                               # +A~ homophily / -A~ heterophily
        wg = self.w * self.g
        Tx = [h] + ([S @ h] if self.K >= 1 else [])
        for k in range(2, self.K + 1):
            Tx.append(2.0 * (S @ Tx[-1]) - Tx[-2])
        return sum(wg[k] * Tx[k] for k in range(self.K + 1))


def run(dataset, damping="jackson", m=1, gso_sign=1, K=10, hid=64, dropout=0.5, lr=1e-2, wd=5e-4,
        epochs=300, patience=50, seeds=range(5), split_kind="dense", trainfrac=0.6, taper_arr=None,
        freeze_w=False, device=None):
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "cache", "planetoid")
    ds = Planetoid(root, dataset, transform=T.NormalizeFeatures())
    d = ds[0].to(device)
    P = gcn_norm_adj(d.edge_index, d.num_nodes, device)
    accs = []
    for s in seeds:
        if split_kind == "dense":
            tr, va, te = split(d.num_nodes, trainfrac, min(0.2, (1 - trainfrac) / 2), seed=1000 + s)
            tr, va, te = tr.to(device), va.to(device), te.to(device)
        else:
            tr, va, te = d.train_mask, d.val_mask, d.test_mask
        torch.manual_seed(s); np.random.seed(s)
        model = ChebGibbsNet(ds.num_features, hid, ds.num_classes, K, damping, m, gso_sign, dropout,
                             taper_arr=taper_arr, freeze_w=freeze_w).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        bv, bt, bad = 0.0, 0.0, 0
        for ep in range(epochs):
            model.train(); opt.zero_grad()
            F.cross_entropy(model(d.x, P)[tr], d.y[tr]).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                pred = model(d.x, P).argmax(1)
                va_ = (pred[va] == d.y[va]).float().mean().item()
                te_ = (pred[te] == d.y[te]).float().mean().item()
            if va_ > bv: bv, bt, bad = va_, te_, 0
            else:
                bad += 1
                if bad >= patience: break
        accs.append(bt)
    return 100 * np.mean(accs), 100 * np.std(accs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Cora", "CiteSeer", "PubMed"])
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=10)  # 10 is what results/chebgibbsnet_repro_v001.json
    # and Table 2 report; the previous default of 5 silently produced different numbers.
    ap.add_argument("--split", default="public")
    args = ap.parse_args()
    seeds = range(args.seeds)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Table 7 per-dataset dropout; published Table 6 ChebGibbsNet (Jackson) target
    DROP = {"Cora": 0.6, "CiteSeer": 0.0, "PubMed": 0.1}
    PUB = {"Cora": 82.42, "CiteSeer": 71.14, "PubMed": 80.18}
    print(f"Faithful ChebGibbsNet (Eq.13, K={args.K}, {args.split} split, SiLU, patience 30, {args.seeds} seeds, {dev})")
    print("Fidelity: OURS(Jackson) vs published Table 6 ChebGibbsNet row.\n")
    print(f"  {'dataset':>9} | {'ours w/Jackson':>15} {'published':>10} {'gap':>6} | "
          f"{'w/Lanczos':>10} {'w/o damp':>10}  damp effect")
    for dsname in args.datasets:
        dp = DROP.get(dsname, 0.5)
        rj = run(dsname, "jackson", K=args.K, dropout=dp, patience=30, seeds=seeds, split_kind=args.split, device=dev)
        rl = run(dsname, "lanczos", m=3, K=args.K, dropout=dp, patience=30, seeds=seeds, split_kind=args.split, device=dev)
        rn = run(dsname, "none", K=args.K, dropout=dp, patience=30, seeds=seeds, split_kind=args.split, device=dev)
        gap = rj[0] - PUB.get(dsname, float('nan'))
        print(f"  {dsname:>9} | {rj[0]:7.2f}±{rj[1]:4.2f}   {PUB.get(dsname,0):>8.2f} {gap:>+6.2f} | "
              f"{rl[0]:6.2f}±{rl[1]:3.1f} {rn[0]:6.2f}±{rn[1]:3.1f}  J{rj[0]-rn[0]:+.2f} La{rl[0]-rn[0]:+.2f}")
