"""Platonov heterophily breadth (CLEAN suite) — long-running (>10 min on the large graphs).

Tests the paper's claims on the UNCONTAMINATED heterophily datasets (roman-empire, amazon-ratings,
minesweeper, tolokers, questions), not the contaminated Chameleon/Squirrel. Sparse propagation (graphs
up to ~49k nodes -> dense n^2 is infeasible). Metric per Platonov: ROC-AUC for the binary sets
(minesweeper/tolokers/questions), accuracy for the multi-class sets (roman-empire/amazon-ratings).
Their provided splits (train/val/test masks, multiple columns), averaged.

Reports for each dataset: ChebGibbsNet w/ Jackson vs w/o damping (the key ablation) + the GSO sign that
node-homophily selects. Expectation from the finite-sample-conditioning thesis: damping helps where the
labeled set is relatively small / the model is under-determined, and its benefit is NOT a Gibbs effect
(reproducible by a non-KPM low-pass taper; run isolation.py-style controls next).

Run (long, >10 min):  python3 src/repro/platonov_breadth.py --datasets roman-empire minesweeper --epochs 3000
"""
from __future__ import annotations
import os, sys, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from damping_cgn import taper_cgn
from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import homophily as pyg_homophily

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "cache", "platonov")
BINARY = {"minesweeper", "tolokers", "questions"}          # ROC-AUC; others = accuracy
DROP = {"roman-empire": 0.2, "amazon-ratings": 0.2, "minesweeper": 0.2, "tolokers": 0.2, "questions": 0.2}


class ChebGibbsSparse(nn.Module):
    def __init__(self, cin, hid, cout, K, damping, m, gso_sign, dropout, taper_arr=None):
        super().__init__()
        self.lin1, self.lin2 = nn.Linear(cin, hid), nn.Linear(hid, cout)
        w0 = np.full(K + 1, 2.0 / (K + 1)); w0[0] = 1.0 / (K + 1)
        self.w = nn.Parameter(torch.tensor(w0, dtype=torch.float32))
        if taper_arr is not None:                                  # custom forward taper (generic low-pass vs KPM)
            g = np.asarray(taper_arr, dtype=float)
        else:
            g = taper_cgn(damping, K, m=m) if damping != "none" else np.ones(K + 1)
        self.register_buffer("g", torch.tensor(g, dtype=torch.float32))
        self.K, self.gso_sign, self.dropout = K, float(gso_sign), dropout

    def forward(self, x, S):                                # S: sparse normalized adjacency (signed)
        x = F.dropout(x, self.dropout, self.training)
        x = F.silu(self.lin1(x))
        x = F.dropout(x, self.dropout, self.training)
        h = self.lin2(x)
        wg = self.w * self.g
        Tx = [h] + ([torch.sparse.mm(S, h)] if self.K >= 1 else [])
        for k in range(2, self.K + 1):
            Tx.append(2.0 * torch.sparse.mm(S, Tx[-1]) - Tx[-2])
        return sum(wg[k] * Tx[k] for k in range(self.K + 1))


def build_S(edge_index, n, gso_sign, device):
    ei, ew = gcn_norm(edge_index, None, n, improved=False, add_self_loops=True, flow="source_to_target")
    return torch.sparse_coo_tensor(ei, gso_sign * ew, (n, n)).coalesce().to(device)


def metric(logits, y, mask, binary):
    if binary:
        from sklearn.metrics import roc_auc_score
        p = torch.softmax(logits[mask], 1)[:, 1].detach().cpu().numpy()
        return roc_auc_score(y[mask].cpu().numpy(), p)
    return (logits[mask].argmax(1) == y[mask]).float().mean().item()


def run(name, damping="jackson", m=3, K=10, hid=64, lr=1e-2, wd=5e-4, epochs=3000, patience=200, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = HeterophilousGraphDataset(ROOT, name)
    d = ds[0].to(device)
    binary = name in BINARY
    h = float(pyg_homophily(d.edge_index, d.y, method="node"))
    gso = 1.0 if h > 0.5 else -1.0
    S = build_S(d.edge_index, d.num_nodes, gso, device)
    dp = DROP.get(name, 0.2)
    nsplits = d.train_mask.shape[1]
    scores = []
    for i in range(nsplits):
        tr, va, te = d.train_mask[:, i], d.val_mask[:, i], d.test_mask[:, i]
        torch.manual_seed(3407); np.random.seed(3407)
        model = ChebGibbsSparse(ds.num_features, hid, int(d.y.max()) + 1, K, damping, m, gso, dp).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        bv, bt, bad = -1.0, 0.0, 0
        for ep in range(epochs):
            model.train(); opt.zero_grad()
            F.cross_entropy(model(d.x, S)[tr], d.y[tr]).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                out = model(d.x, S)
                va_ = metric(out, d.y, va, binary); te_ = metric(out, d.y, te, binary)
            if va_ > bv: bv, bt, bad = va_, te_, 0
            else:
                bad += 1
                if bad >= patience: break
        scores.append(bt)
    return 100 * np.mean(scores), 100 * np.std(scores), h, ("ROC-AUC" if binary else "acc")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"])
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=3000)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Platonov heterophily breadth (clean suite, K={args.K}, sparse, {dev})")
    print(f"  {'dataset':>15} {'metric':>8} {'homophily':>9} | {'w/Jackson':>12} {'w/o damp':>12}  effect")
    out = {}
    for name in args.datasets:
        mj, sj, h, met = run(name, "jackson", K=args.K, epochs=args.epochs, device=dev)
        mn, sn, _, _ = run(name, "none", K=args.K, epochs=args.epochs, device=dev)
        out[name] = {"metric": met, "homophily": h, "jackson": mj, "jackson_std": sj,
                     "none": mn, "none_std": sn, "damp_effect": mj - mn}
        print(f"  {name:>15} {met:>8} {h:>9.3f} | {mj:6.2f}±{sj:4.2f} {mn:6.2f}±{sn:4.2f}  J{mj-mn:+.2f}")
        json.dump(out, open(os.path.join(ROOT, "..", "..", "..", "results", "platonov_breadth_v001.json"), "w"), indent=2)
    print("  saved results/platonov_breadth_v001.json")
