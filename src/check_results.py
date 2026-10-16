"""Cheap integrity check: load the cached result JSONs, verify a few headline values against the paper,
and regenerate the figures. No dataset download or training. Run: python src/check_results.py"""
import os, json, sys
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
def load(n): return json.load(open(os.path.join(R, n + "_v001.json")))
ok = True
def chk(name, got, exp, tol):
    global ok
    p = abs(got - exp) <= tol
    ok = ok and p
    print(f"  [{'ok' if p else 'FAIL'}] {name}: {got:.4f} (expected {exp} +/- {tol})")

lp = load("lowpass_seeds")["tapers"]
chk("Fig 1 Jackson(KPM) mean", lp["jackson(KPM)"]["mean"], 79.90, 0.5)
chk("Fig 1 anti-low-pass mean", lp["increasing(anti)"]["mean"], 48.18, 1.0)
pp = load("platonov_paired")["datasets"]["roman-empire"]
chk("Table 3 roman-empire Jackson", pp["jackson"]["mean"], 71.22, 0.3)
chk("Table 3 roman-empire none", pp["none"]["mean"], 72.46, 0.3)
fa = load("factorial_splits")["datasets"]["amazon-ratings"]["main_effects"]
chk("Table 4 amazon activation effect", fa["activation"]["mean"], -5.43, 0.2)
dr = load("data_regime_seeds")["datasets"]["Cora"][0]
chk("Fig 2 Cora benefit @2%", dr["jackson"]["mean"] - dr["none"]["mean"], 7.12, 0.5)

print("regenerating figures ...")
import paper_figures  # noqa: F401  (module regenerates all figures on import)
print("PASS" if ok else "SOME CHECKS FAILED"); sys.exit(0 if ok else 1)
