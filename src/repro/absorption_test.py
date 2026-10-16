"""Absorption test on a REAL trained model (PubMed, per-k free-coefficient ChebNet).

Theorem (i) ABSORPTION AT THE OPTIMUM: with weight decay OFF, damped and undamped free-coefficient
training have the SAME attainable optima (g_k absorbed into free W_k), so at a shared optimum damping
is inert; any finite-budget difference enters through initialization and the optimization path. This
script does not sweep the budget, so it does not establish that the difference shrinks with training.
Theorem (ii) THE RESIDUAL CHANNEL: with weight decay ON the penalty is reshaped rho_k/g_k^2, so the
damped and undamped objectives differ and their optima need not coincide -- an order-graded
shrinkage/conditioning effect (NOT Gibbs). This says the optima may differ, not that they must.

We contrast SHORT (early-stopped, ChebGibbsNet-like) vs LONG (large-budget) training, wd on/off.
The long runs are not verified to have converged; they are simply given a much larger budget.
Expectation: wd=0 long -> damping effect ~0 (absorption); wd>0 -> a residual, placement-independent
regularization effect that is the reproducible mechanism (the +4-point PubMed help).
"""
from __future__ import annotations
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chebnet_gibbs import run

if __name__ == "__main__":
    DS = "PubMed"
    seeds = tuple(range(5))
    print(f"Absorption test on {DS}, per-k free-coefficient ChebNet, K=2, {len(seeds)} seeds")
    print("damped vs undamped test acc; effect = jackson/lanczos minus none\n")
    configs = [
        ("SHORT wd=5e-4 (early-stopped, ChebGibbsNet-like)", dict(wd=5e-4, epochs=200, patience=30)),
        ("LONG  wd=5e-4 (large budget, reg channel ON)", dict(wd=5e-4, epochs=1500, patience=400)),
        ("LONG  wd=0    (large budget, ABSORPTION test)", dict(wd=0.0, epochs=1500, patience=400)),
    ]
    print(f"{'regime':>52} | {'none':>11} {'jackson':>11} {'lanczos':>11} | effect")
    for name, kw in configs:
        r = {d: run(DS, d, K=2, share_W=False, seeds=seeds, **kw) for d in ("none", "jackson", "lanczos")}
        eff = f"J {r['jackson'][0]-r['none'][0]:+.1f}  La {r['lanczos'][0]-r['none'][0]:+.1f}"
        print(f"{name:>52} | " + " ".join(f"{r[d][0]:5.2f}±{r[d][1]:4.2f}" for d in ("none","jackson","lanczos")) + f" | {eff}")
