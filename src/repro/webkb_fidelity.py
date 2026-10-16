"""Reproduction on WebKB heterophily (Cornell/Texas/Wisconsin). Extends the citation fidelity.

Faithful ChebGibbsNet on WebKB: node-homophily < 0.5 -> S = -A~ (auto, like their code); geom-gcn 10
splits (WebKB provides train/val/test masks with 10 columns); average over the 10 splits (model seed
fixed, like their main.py). Per-dataset dropout from Table 7 (Cornell 0.2, Texas 0, Wisconsin 0.1).
Compare to published Table 6 (Cornell 78.11, Texas 85.68, Wisconsin 81.18) and Table 2 (K=2 ablation).
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chebgibbsnet import ChebGibbsNet
from gpr_gnn import gcn_norm_adj
from torch_geometric.datasets import WebKB
from torch_geometric.utils import homophily as pyg_homophily
import torch_geometric.transforms as T

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "cache", "webkb")
DROP = {"Cornell": 0.2, "Texas": 0.0, "Wisconsin": 0.1}
PUB6 = {"Cornell": 78.11, "Texas": 85.68, "Wisconsin": 81.18}   # Table 6 ChebGibbsNet


def run(name, damping="jackson", m=3, K=10, hid=64, lr=1e-2, wd=5e-4, epochs=3000, patience=200, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = WebKB(ROOT, name, transform=T.NormalizeFeatures())
    d = ds[0].to(device)
    P = gcn_norm_adj(d.edge_index, d.num_nodes, device)
    hnode = float(pyg_homophily(d.edge_index, d.y, method="node"))
    gso_sign = 1.0 if hnode > 0.5 else -1.0
    dp = DROP.get(name, 0.5)
    nsplits = d.train_mask.shape[1]
    accs = []
    for i in range(nsplits):
        tr, va, te = d.train_mask[:, i], d.val_mask[:, i], d.test_mask[:, i]
        torch.manual_seed(3407); np.random.seed(3407)               # their fixed model seed
        model = ChebGibbsNet(ds.num_features, hid, ds.num_classes, K, damping, m, gso_sign, dp).to(device)
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
    return 100 * np.mean(accs), 100 * np.std(accs), hnode


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" WebKB fidelity (ChebGibbsNet Eq.13, K=10, geom-gcn 10 splits, {dev})")
    print(f"  {'dataset':>10} {'homophily':>9} {'S':>4} | {'ours(Jackson)':>14} {'published':>10} {'gap':>6} | "
          f"{'w/o damp':>9}  damp effect")
    out = {}
    for name in ["Cornell", "Texas", "Wisconsin"]:
        mj, sj, h = run(name, "jackson", dev=dev) if False else run(name, "jackson", device=dev)
        mn, sn, _ = run(name, "none", device=dev)
        sign = "-A~" if h <= 0.5 else "+A~"
        gap = mj - PUB6[name]
        out[name] = {"ours_jackson": mj, "std": sj, "published": PUB6[name], "gap": gap,
                     "none": mn, "damp_effect": mj - mn, "homophily": h}
        print(f"  {name:>10} {h:>9.3f} {sign:>4} | {mj:7.2f}±{sj:4.2f}   {PUB6[name]:>8.2f} {gap:>+6.2f} | "
              f"{mn:6.2f}±{sn:3.1f}  J{mj-mn:+.2f}")
    import json
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "..", "results", "webkb_repro_v001.json"), "w"), indent=2)
    print("  saved results/webkb_repro_v001.json")
