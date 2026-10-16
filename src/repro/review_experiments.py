"""Three short (<10-min) supplementary experiments:

 A. Weight-decay ablation. Tests whether the damping benefit is mediated by weight decay
    (Appendix A's residual-channel account). Run {none,jackson} x {wd=0, wd=5e-4} on the
    small-label PUBLIC split (the only regime where damping helps). Report benefit(wd) with 5-seed std.
    Whatever the sign of d(benefit)/d(wd), report it as-is.

 B. Realized-coefficient profiles c_k = g_k * w_k for {none, jackson, generic low-pass, anti-low-pass}
    from trained ChebGibbsNet (Cora, best-val, 3-seed mean). Exact-from-coefficients extraction for
    ChebGibbsNet. Makes the low-pass prior visible.

 C. l2-norm-matched anti-low-pass control on PubMed. The anti taper is unnormalized (g_0 ~170x smaller),
    so a skeptic could call the 27-pt collapse a magnitude artifact. Rescale g so ||g||_2 = ||flat||_2
    and re-measure: if it still collapses, the taper's global norm alone does not explain the effect.
    This does not exclude per-coordinate scaling or a retuned anti-low-pass model.

All reuse the ChebGibbsNet (Eq.13) reproduction. Run: python3 src/repro/review_experiments.py  (~7 min, 1 GPU)
"""
import os, sys, json, copy
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chebgibbsnet import ChebGibbsNet, run
from gpr_gnn import gcn_norm_adj
from damping_cgn import taper_cgn
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RES = os.path.join(ROOT, "results")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DROP = {"Cora": 0.6, "CiteSeer": 0.0, "PubMed": 0.1}          # Table-7 per-dataset dropout


def wd_ablation():
    """A. Does the damping benefit require weight decay? Public (small-label) split, 5 seeds."""
    out = {}
    for ds in ["Cora", "CiteSeer", "PubMed"]:
        out[ds] = {}
        for wd in [0.0, 5e-4]:
            mj, sj = run(ds, "jackson", K=10, dropout=DROP[ds], wd=wd, patience=30,
                         seeds=range(5), split_kind="public", device=DEV)
            mn, sn = run(ds, "none", K=10, dropout=DROP[ds], wd=wd, patience=30,
                         seeds=range(5), split_kind="public", device=DEV)
            out[ds][f"wd={wd}"] = {"jackson": round(mj, 2), "jackson_std": round(sj, 2),
                                   "none": round(mn, 2), "none_std": round(sn, 2),
                                   "benefit": round(mj - mn, 2)}
            print(f"  {ds:>9} wd={wd:<6}: jackson {mj:5.2f}  none {mn:5.2f}  benefit {mj-mn:+5.2f}", flush=True)
    return out


def train_model(ds_name, taper_arr, dropout, seed, epochs=300, patience=30):
    """Train one deployed ChebGibbsNet (free w_k, fixed forward taper) and return the best-val model."""
    ds = Planetoid(os.path.join(ROOT, "data", "cache", "planetoid"), ds_name, transform=T.NormalizeFeatures())
    d = ds[0].to(DEV)
    P = gcn_norm_adj(d.edge_index, d.num_nodes, DEV)
    tr, va, te = d.train_mask, d.val_mask, d.test_mask
    torch.manual_seed(seed); np.random.seed(seed)
    m = ChebGibbsNet(ds.num_features, 64, ds.num_classes, 10, "none", 3, 1, dropout, taper_arr=taper_arr).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=5e-4)
    bv, best_state, bt, bad = 0.0, None, 0.0, 0
    for ep in range(epochs):
        m.train(); opt.zero_grad()
        F.cross_entropy(m(d.x, P)[tr], d.y[tr]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pred = m(d.x, P).argmax(1)
            va_ = (pred[va] == d.y[va]).float().mean().item()
            te_ = (pred[te] == d.y[te]).float().mean().item()
        if va_ > bv:
            bv, best_state, bt, bad = va_, copy.deepcopy(m.state_dict()), te_, 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    return m, 100 * bt


def coef_profiles():
    """B. Realized c_k = g_k * w_k for four tapers (Cora, 3-seed mean)."""
    K = 10; k = np.arange(K + 1.0)
    tapers = {"none": np.ones(K + 1), "jackson": taper_cgn("jackson", K),
              "geometric(low-pass)": 0.6 ** k, "increasing(anti)": 0.6 ** (K - k)}
    out = {}
    for name, g in tapers.items():
        cs = []
        for seed in range(3):
            m, _ = train_model("Cora", g, DROP["Cora"], seed)
            cs.append((m.g.detach().cpu().numpy()) * (m.w.detach().cpu().numpy()))
        c = np.mean(cs, axis=0)
        out[name] = {"c": [round(float(x), 5) for x in c],
                     "abs_norm": [round(float(x), 4) for x in (np.abs(c) / abs(c[0]))]}
        print(f"  {name:>20}: |c_k|/|c_0| = {np.round(np.abs(c)/abs(c[0]), 2)}", flush=True)
    return out


def anti_control():
    """C. l2-norm-matched anti-low-pass control (PubMed, 5 seeds)."""
    K = 10; k = np.arange(K + 1.0)
    g_anti = 0.6 ** (K - k)
    g_norm = g_anti * np.sqrt(K + 1) / np.linalg.norm(g_anti)      # ||g_norm||_2 = ||flat||_2 = sqrt(K+1)
    out = {}
    for name, g in [("none", np.ones(K + 1)), ("anti_raw", g_anti), ("anti_l2matched", g_norm)]:
        accs = [train_model("PubMed", g, DROP["PubMed"], s)[1] for s in range(5)]
        out[name] = {"acc": round(float(np.mean(accs)), 2), "std": round(float(np.std(accs)), 2)}
        print(f"  PubMed {name:>15}: {np.mean(accs):5.2f} ± {np.std(accs):4.2f}", flush=True)
    return out


if __name__ == "__main__":
    print(f"Review experiments (device={DEV})\n\nA. weight-decay ablation (does the damping benefit require weight decay?)")
    A = wd_ablation()
    json.dump({"wd_ablation": A, "note": "public/small-label split, K=10, 5 seeds; benefit=jackson-none"},
              open(os.path.join(RES, "wd_ablation_v001.json"), "w"), indent=2)
    print("\nB. realized-coefficient profiles c_k = g_k w_k (Cora, 3-seed mean)")
    B = coef_profiles()
    json.dump(B, open(os.path.join(RES, "coef_profile_v001.json"), "w"), indent=2)
    print("\nC. l2-norm-matched anti-low-pass control (PubMed, 5 seeds)")
    C = anti_control()
    json.dump({"control": C, "note": "raw vs l2-norm-matched anti taper; if both collapse, the taper's global norm "
                         "alone does not explain the effect"},
              open(os.path.join(RES, "anti_control_v001.json"), "w"), indent=2)
    print("\nsaved wd_ablation_v001.json, coef_profile_v001.json, anti_control_v001.json")
