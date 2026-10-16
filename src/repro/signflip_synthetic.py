"""Taper-direction SIGN-FLIP on synthetic cSBM (controllable).

Hypothesis under test (NOT supported by the result below): that the optimal per-order taper
DIRECTION is set by the task's spectral character. cSBM community signal sits at lambda_signal ~ 2(1-h) on the normalized
Laplacian [0,2]: homophily (h>0.5) -> low frequency; heterophily (h<0.5) -> high frequency. Prediction:
  h > 0.5 (homophily)   -> DECAYING profile (down-weight high-order Cheb terms) WINS
  h < 0.5 (heterophily) -> INCREASING profile (up-weight high-order terms) WINS
and the winner FLIPS as h crosses 0.5. The measured outcome is that the FLAT profile wins at both,
so the hypothesis is falsified; a per-order profile is not a graph-frequency response.

Tapers on T_k(L~), L~ = normalized Laplacian in [0,2]. These are profiles in polynomial ORDER, not
graph-frequency responses: a decaying g_k down-weights high Chebyshev orders, which is not the same
as a low-pass response in lambda (the graph-frequency adaptation would be S -> -S, tested separately).
  decaying   : rho^k        (rho<1) -> emphasizes LOW order
  flat       : 1
  increasing : rho^(K-k)             -> emphasizes HIGH order

The check the hypothesis implies -- argmax taper 'decaying' for h>0.5 and 'increasing' for h<0.5 --
FAILS: 'flat' is the argmax at both, so the hypothesis is not supported.
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "synth"))
import csbm
from splits import split
from chebnet_gibbs import sym_norm_adj
from isolation import ChebFree


def taper(kind, K, rho=0.5):
    k = np.arange(K + 1, dtype=float)
    if kind == "decaying":   return rho ** k
    if kind == "flat":       return np.ones(K + 1)
    if kind == "increasing": return rho ** (K - k)
    raise ValueError(kind)




def run(h, taper_kind, n=2000, d=20, feat_snr=2.0, feat_dim=64, K=10, hid=64, dropout=0.5,
        lr=1e-2, wd=5e-4, epochs=300, patience=50, seeds=range(5), device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    accs = []
    for s in seeds:
        g = csbm.generate_sbm(n, h, d, seed=s)
        X = csbm.csbm_features(g["comm"], feat_dim, feat_snr, seed=s)
        x = torch.tensor(X, dtype=torch.float32, device=device)
        y = torch.tensor(g["comm"], dtype=torch.long, device=device)
        edge = np.array(np.nonzero(np.triu(g["A"], 1)))
        edge = torch.tensor(np.concatenate([edge, edge[::-1]], 1), dtype=torch.long, device=device)
        Ahat = sym_norm_adj(edge, n, device)
        tr, va, te = split(n, 0.2, 0.2, seed=1000 + s)
        tr, va, te = tr.to(device), va.to(device), te.to(device)
        gf = taper(taper_kind, K)
        torch.manual_seed(s); np.random.seed(s)
        model = ChebFree(feat_dim, hid, 2, K, gf, dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        bv, bt, bad = 0.0, 0.0, 0
        for ep in range(epochs):
            model.train(); opt.zero_grad()
            F.cross_entropy(model(x, Ahat)[tr], y[tr]).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                pred = model(x, Ahat).argmax(1)
                vacc = (pred[va] == y[va]).float().mean().item()
                tacc = (pred[te] == y[te]).float().mean().item()
            if vacc > bv: bv, bt, bad = vacc, tacc, 0
            else:
                bad += 1
                if bad >= patience: break
        accs.append(bt)
    return 100 * np.mean(accs), 100 * np.std(accs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hs", nargs="+", type=float, default=[0.85, 0.70, 0.30, 0.15])
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--feat_snr", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    seeds = range(args.seeds)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tapers = ["decaying", "flat", "increasing"]
    print(f"SIGN-FLIP test (cSBM, K={args.K}, feat_snr={args.feat_snr}, {args.seeds} seeds, {dev})")
    print("predict: decaying wins for h>0.5 (homophily/low-freq); increasing wins for h<0.5 (hetero/high-freq)\n")
    print(f"  {'h':>5} {'regime':>10} | " + " ".join(f"{t:>11}" for t in tapers) + " | best | predicted | ok")
    all_ok = True
    for h in args.hs:
        r = {t: run(h, t, K=args.K, feat_snr=args.feat_snr, seeds=seeds, device=dev) for t in tapers}
        best = max(tapers, key=lambda t: r[t][0])
        pred = "decaying" if h > 0.5 else "increasing"
        ok = (best == pred); all_ok &= ok
        reg = "homophily" if h > 0.5 else "hetero"
        print(f"  {h:>5.2f} {reg:>10} | " + " ".join(f"{r[t][0]:5.2f}±{r[t][1]:4.2f}" for t in tapers) +
              f" | {best:>9} | {pred:>9} | {'PASS' if ok else 'FAIL'}")
    print(f"\n  SIGN-FLIP check: {'PASS' if all_ok else 'FAIL'}")
