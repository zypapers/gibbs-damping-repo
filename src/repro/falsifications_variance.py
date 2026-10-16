"""Per-seed variance re-run for Fig 4 (falsifications) so both panels get error bars.

Panel (a) sign-flip: cSBM (hard config d=4, feat_snr=0), h in {0.85, 0.15}, tapers {decaying, flat,
    increasing} -> per-seed test acc mean/std. Reuses signflip_synthetic.run verbatim (it already returns
    mean, std over seeds).
Panel (b) required-K: dense 60%-train split, Cora/CiteSeer, K in {1,2,4,8,16}, flat taper (free
    coefficients) -> per-seed test acc mean/std. Reuses isolation.ChebFree + signflip_synthetic.split.

Qualitative claims in the paper are unchanged (flat wins both; accuracy peaks at K=1-2); this run only
adds the std for error bars. ~5-8 min on one GPU. Run: python3 src/repro/falsifications_variance.py
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from signflip_synthetic import run as signflip_run, split as dense_split
from isolation import ChebFree
from chebnet_gibbs import sym_norm_adj

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RES = os.path.join(ROOT, "results")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def panel_a(seeds=5):
    """cSBM sign-flip, hard config (d=4, feat_snr=0), per (h, taper) mean/std."""
    out = {}
    for h, tag in [(0.85, "h=0.85_homophily"), (0.15, "h=0.15_heterophily")]:
        out[tag] = {}
        for tk in ["decaying", "flat", "increasing"]:
            mu, sd = signflip_run(h, tk, d=4, feat_snr=0.0, seeds=range(seeds), device=DEV)
            out[tag][tk] = {"mean": round(float(mu), 2), "std": round(float(sd), 2)}
            print(f"  (a) h={h} {tk:>10}: {mu:5.2f} ± {sd:4.2f}", flush=True)
    return out


def run_dense_K(dataset, K, trainfrac=0.6, seeds=5, epochs=300, patience=50):
    """Free-coefficient ChebFree (flat taper) at degree K on a dense 60%-train split; per-seed test acc."""
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T
    ds = Planetoid(os.path.join(ROOT, "data", "cache", "planetoid"), dataset, transform=T.NormalizeFeatures())
    d = ds[0].to(DEV)
    Ahat = sym_norm_adj(d.edge_index, d.num_nodes, DEV)
    gf = np.ones(K + 1)                                            # flat taper (free w_k), degree-K only
    accs = []
    for s in range(seeds):
        tr, va, te = dense_split(d.num_nodes, trainfrac, 0.2, seed=1000 + s)
        tr, va, te = tr.to(DEV), va.to(DEV), te.to(DEV)
        torch.manual_seed(s); np.random.seed(s)
        model = ChebFree(ds.num_features, 64, ds.num_classes, K, gf, 0.5).to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
        bv, bt, bad = 0.0, 0.0, 0
        for ep in range(epochs):
            model.train(); opt.zero_grad()
            F.cross_entropy(model(d.x, Ahat)[tr], d.y[tr]).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                pred = model(d.x, Ahat).argmax(1)
                vacc = (pred[va] == d.y[va]).float().mean().item()
                tacc = (pred[te] == d.y[te]).float().mean().item()
            if vacc > bv: bv, bt, bad = vacc, tacc, 0
            else:
                bad += 1
                if bad >= patience: break
        accs.append(100 * bt)
    return float(np.mean(accs)), float(np.std(accs))


def panel_b(seeds=5):
    out = {}
    for dsn in ["Cora", "CiteSeer"]:
        out[dsn] = {}
        for K in [1, 2, 4, 8, 16]:
            mu, sd = run_dense_K(dsn, K, seeds=seeds)
            out[dsn][str(K)] = {"mean": round(mu, 2), "std": round(sd, 2)}
            print(f"  (b) {dsn} K={K:>2}: {mu:5.2f} ± {sd:4.2f}", flush=True)
    return out


if __name__ == "__main__":
    print(f"Falsifications variance (Fig 4 error bars), device={DEV}\n(a) cSBM sign-flip (d=4, snr=0):")
    A = panel_a()
    print("(b) required-K, dense 60% train:")
    B = panel_b()
    json.dump({"sign_flip_hard_cSBM_d4_snr0": A, "required_K_data_rich": B,
               "note": "per-seed mean/std for Fig 4 error bars; 5 seeds"},
              open(os.path.join(RES, "falsifications_seeds_v001.json"), "w"), indent=2)
    print("\nsaved results/falsifications_seeds_v001.json")
