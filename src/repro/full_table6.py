"""Complete the Table-6 reproduction: the 4 datasets missing from the citation+WebKB set --
Film (Actor), Wiki-CS, Chameleon, Squirrel. Dense ChebGibbsNet (Eq.13), per-dataset Table-7 HPs, auto
GSO sign from node homophily, geom-gcn / provided splits. Chameleon/Squirrel are the contaminated
pre-Platonov versions -- reproduced only to check we match ChebGibbsNet's SETUP, quoted with
that caveat. Published Table 6: Film 37.90, Chameleon 68.64, Squirrel 33.83, Wiki-CS 80.19.

Run (long, >10 min): python3 src/repro/full_table6.py
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chebgibbsnet import ChebGibbsNet
from gpr_gnn import gcn_norm_adj
from torch_geometric.utils import homophily as pyg_homophily

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "cache")
# Table-7 HPs: (dropout, hidden, lr, wd) ; published Table-6 ChebGibbsNet accuracy
HP = {"Film": (0.6, 32, 1e-3, 5e-5, 37.90), "WikiCS": (0.2, 128, 1e-2, 5e-4, 80.19),
      "Chameleon": (0.5, 64, 1e-2, 5e-4, 68.64), "Squirrel": (0.3, 64, 1e-2, 5e-4, 33.83)}


def load(name):
    import torch_geometric.transforms as T
    if name == "Film":
        from torch_geometric.datasets import Actor
        ds = Actor(os.path.join(ROOT, "actor"), transform=T.NormalizeFeatures()); return ds, ds[0]
    if name == "WikiCS":
        from torch_geometric.datasets import WikiCS
        ds = WikiCS(os.path.join(ROOT, "wikics"), is_undirected=True); return ds, ds[0]
    from torch_geometric.datasets import WikipediaNetwork
    ds = WikipediaNetwork(os.path.join(ROOT, "wikipedia"), name.lower(), geom_gcn_preprocess=True,
                          transform=T.NormalizeFeatures()); return ds, ds[0]


def split_cols(d):
    """Return list of (train,val,test) masks across the provided splits."""
    tr, va, te = d.train_mask, d.val_mask, d.test_mask
    ns = tr.shape[1] if tr.dim() > 1 else 1
    cols = []
    for i in range(ns):
        t = tr[:, i] if tr.dim() > 1 else tr
        v = va[:, i] if va.dim() > 1 else va
        e = te[:, i] if te.dim() > 1 else te          # WikiCS: single test mask
        cols.append((t, v, e))
    return cols


PATIENCE = {"Film": 300}          # Actor validation is noisy; the recorded run used 300


def run(name, damping, K=10, epochs=3000, patience=None, device=None):
    patience = patience if patience is not None else PATIENCE.get(name, 200)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds, d = load(name); d = d.to(device)
    dp, hid, lr, wd, _ = HP[name]
    P = gcn_norm_adj(d.edge_index, d.num_nodes, device)
    h = float(pyg_homophily(d.edge_index, d.y, method="node")); gso = 1.0 if h > 0.5 else -1.0
    cols = split_cols(d); accs = []
    for (tr, va, te) in cols:
        tr, va, te = tr.to(device), va.to(device), te.to(device)
        torch.manual_seed(3407); np.random.seed(3407)
        model = ChebGibbsNet(ds.num_features, hid, int(d.y.max()) + 1, K, damping, 3, gso, dp).to(device)
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
    return 100 * np.mean(accs), 100 * np.std(accs), h


if __name__ == "__main__":
    names = sys.argv[1:] or ["Film", "Chameleon", "Squirrel", "WikiCS"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Full Table-6 reproduction, missing datasets (dense ChebGibbsNet Eq.13, K=10, {dev})")
    print(f"  {'dataset':>10} {'homophily':>9} | {'ours(Jackson)':>14} {'published':>10} {'gap':>6} | {'w/o damp':>10}")
    out = {}
    for name in names:
        mj, sj, h = run(name, "jackson", device=dev)
        mn, sn, _ = run(name, "none", device=dev)
        pub = HP[name][4]
        out[name] = {"ours_jackson": round(mj, 2), "std": round(sj, 2), "published": pub,
                     "gap": round(mj - pub, 2), "none": round(mn, 2), "damp_effect": round(mj - mn, 2), "homophily": round(h, 3)}
        print(f"  {name:>10} {h:>9.3f} | {mj:7.2f}±{sj:4.2f}   {pub:>8.2f} {mj-pub:>+6.2f} | {mn:6.2f}±{sn:4.2f}", flush=True)
        json.dump(out, open(os.path.join(ROOT, "..", "..", "results", "full_table6_v001.json"), "w"), indent=2)
    print("  saved results/full_table6_v001.json")
