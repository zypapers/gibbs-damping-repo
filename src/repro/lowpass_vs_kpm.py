"""K=10 low-pass vs KPM on the DEPLOYED ChebGibbsNet (Eq.13): is the damping gain the KPM shape, or any
low-pass? PubMed public split, per-order forward tapers, free w_k, 10 seeds, bootstrap
CI on jackson - best-generic-low-pass. Generic low-pass = geometric / raised-cosine (NOT KPM); anti =
increasing. Prediction: generic low-pass matches Jackson/Lanczos; increasing (anti-low-pass) hurts.
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chebgibbsnet import ChebGibbsNet
from gpr_gnn import gcn_norm_adj
from damping_cgn import taper_cgn


def tapers(K=10):
    k = np.arange(K + 1, dtype=float)
    return {
        "none": np.ones(K + 1),
        "jackson(KPM)": taper_cgn("jackson", K),
        "lanczos(KPM)": taper_cgn("lanczos", K, m=3),
        "geometric(low-pass)": 0.6 ** k,
        "cosine(low-pass)": 0.5 * (1 + np.cos(np.pi * k / (K + 1))),   # raised-cosine window, non-KPM
        "increasing(anti)": 0.6 ** (K - k),
    }


def run_taper(g, K=10, dropout=0.1, seeds=range(10), device=None):
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "cache", "planetoid")
    ds = Planetoid(root, "PubMed", transform=T.NormalizeFeatures())
    d = ds[0].to(device)
    P = gcn_norm_adj(d.edge_index, d.num_nodes, device)
    accs = []
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        m = ChebGibbsNet(ds.num_features, 64, ds.num_classes, K, "none", 1, 1, dropout, taper_arr=g).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=5e-4)
        bv, bt, bad = 0.0, 0.0, 0
        for ep in range(300):
            m.train(); opt.zero_grad()
            F.cross_entropy(m(d.x, P)[d.train_mask], d.y[d.train_mask]).backward(); opt.step()
            m.eval()
            with torch.no_grad():
                pred = m(d.x, P).argmax(1)
                va = (pred[d.val_mask] == d.y[d.val_mask]).float().mean().item()
                te = (pred[d.test_mask] == d.y[d.test_mask]).float().mean().item()
            if va > bv: bv, bt, bad = va, te, 0
            else:
                bad += 1
                if bad >= 30: break
        accs.append(bt)
    return np.array(accs)


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"K=10 low-pass vs KPM (deployed ChebGibbsNet, PubMed public split, 10 seeds, {dev})")
    T = tapers(10)
    res = {name: run_taper(g, device=dev) for name, g in T.items()}
    base = res["none"].mean()
    print(f"  {'taper':>22} {'acc':>7} {'gain':>7}")
    for name in T:
        print(f"  {name:>22} {100*res[name].mean():7.2f} {100*(res[name].mean()-base):+7.2f}", flush=True)
    # bootstrap CI: jackson - best generic low-pass
    gens = ["geometric(low-pass)", "cosine(low-pass)"]
    best = max(gens, key=lambda n: res[n].mean())
    diff = res["jackson(KPM)"] - res[best]
    rng = np.random.default_rng(0)
    bs = diff[rng.integers(0, len(diff), (10000, len(diff)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    verdict = "generic low-pass matches KPM within CI" if lo < 0 < hi else "KPM has an edge"
    print(f"\n  jackson - best_generic({best}) = {100*diff.mean():+.2f}  95% CI [{100*lo:+.2f},{100*hi:+.2f}]  -> {verdict}")
    out = {"K": 10, "dataset": "PubMed", "split": "public",
           "acc": {n: float(res[n].mean() * 100) for n in T}, "gain_vs_none": {n: float((res[n].mean()-base)*100) for n in T},
           "jackson_minus_best_generic": {"best": best, "diff": float(diff.mean()*100), "ci": [float(lo*100), float(hi*100)]}}
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "..", "results", "lowpass_vs_kpm_v001.json"), "w"), indent=2)
    print("  saved results/lowpass_vs_kpm_v001.json")
