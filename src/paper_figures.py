"""Generate the paper's headline figures from results/ JSON. PDF (LaTeX) + PNG (markdown)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
# LoG requires the submitted PDF to contain Type-1 fonts only; matplotlib embeds
# Type-3 DejaVu by default, and mathtext ($K=10$, log-axis labels) stays Type 3 even
# under use14corefonts. make_type1.sh therefore converts figure text to vector
# outlines after generation, leaving no fonts in the figures at all. Sizes are raised
# because these render at ~0.61 scale on the page: ticks were near 6pt, legends 5pt.
matplotlib.rcParams.update({
    "pdf.use14corefonts": True,
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "mathtext.fontset": "stix",
})
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
blue, orange, green, vermillion, grey = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#666666"


def save(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=170)
    plt.close(fig)
    print("wrote", name)


# Shared geometry for the side-by-side pair (F1 low-pass + F2 data-regime): identical figsize AND identical
# axes rectangle => at equal \linewidth the on-page PLOT BOXES render identically (tight_layout, which sizes
# each figure's margins from its own content, would make them differ). Margins sized for the worst case
# (F1's 2-line x-tick labels + wider y-label); F2 just gets a little extra whitespace.
PAIR_FIGSIZE = (5.2, 3.4)
PAIR_BOX = dict(left=0.135, right=0.975, bottom=0.205, top=0.965)


# ---- F3 (headline): damping benefit decays as data grows (Cora + PubMed), 5-seed mean +/- paired std ----
drs = json.load(open(os.path.join(ROOT, "results", "data_regime_seeds_v001.json")))["datasets"]
fig, ax = plt.subplots(figsize=PAIR_FIGSIZE)   # shared with F2 (low-pass) so on-page plot boxes match exactly
ax.axhline(0, color=grey, lw=0.8, ls="-")
for label, col in [("Cora", vermillion), ("PubMed", blue)]:
    rows = drs[label]
    xf = [r["train_frac"] for r in rows]
    ben = [r["jackson"]["mean"] - r["none"]["mean"] for r in rows]
    err = [float(np.std(np.array(r["jackson"]["values"]) - np.array(r["none"]["values"]))) for r in rows]
    ax.errorbar(xf, ben, yerr=err, fmt="o-", color=col, lw=1.8, ms=6, capsize=2.5,
                elinewidth=1.0, label=label)
ax.annotate("a fixed-coefficient\nGibbs effect would\nnot decay", xy=(0.29, 0.9), xytext=(0.135, 3.9),
            fontsize=9, color="#222222", ha="left",
            arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.5, mutation_scale=14))
ax.set_xscale("log")
ax.set_ylim(-1.0, 9)
ax.tick_params(labelsize=9)
ax.set_xlabel("training-set fraction", fontsize=10)
ax.set_ylabel("damping benefit  (Jackson $-$ no taper)  [%]", fontsize=8.5)
ax.legend(fontsize=9, loc="upper right", title="ChebGibbsNet, $K{=}10$", title_fontsize=8)
ax.grid(True, alpha=0.25)
fig.subplots_adjust(**PAIR_BOX)
save(fig, "f3_data_regime", tight=False)


# ---- F2: on the DEPLOYED model (K=10), any low-pass ~ KPM; anti-low-pass COLLAPSES it (5-seed mean +/- std) ----
mtp = json.load(open(os.path.join(ROOT, "results", "lowpass_seeds_v001.json")))["tapers"]
bars = [("no taper", mtp["none"], grey), ("anti", mtp["increasing(anti)"], vermillion),
        ("geometric", mtp["geometric(low-pass)"], green),
        ("cosine", mtp["cosine(low-pass)"], green),
        ("Jackson", mtp["jackson(KPM)"], blue), ("Lanczos$^3$", mtp["lanczos(KPM)"], blue)]
fig, ax = plt.subplots(figsize=PAIR_FIGSIZE)   # shared with F3 (data-regime) so on-page plot boxes match exactly
xs = np.arange(len(bars))
ax.bar(xs, [b[1]["mean"] for b in bars], yerr=[b[1]["std"] for b in bars], color=[b[2] for b in bars],
       width=0.72, capsize=3, error_kw=dict(elinewidth=1.0))
ax.axhline(mtp["none"]["mean"], color=grey, lw=0.9, ls="--")
ax.set_xticks(xs)
ax.set_xticklabels([b[0] for b in bars], fontsize=8.5)
ax.set_ylim(45, 90)
ax.tick_params(axis="y", labelsize=9)
ax.set_ylabel("PubMed test accuracy [%]  ($K{=}10$)", fontsize=10)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=blue, label="KPM (Gibbs)"), Patch(color=green, label="generic low-pass"),
                   Patch(color=vermillion, label="anti-low-pass"), Patch(color=grey, label="no taper")],
          fontsize=9, loc="upper left", ncol=2)
fig.subplots_adjust(**PAIR_BOX)
save(fig, "f2_lowpass_matches_kpm", tight=False)

print("figures ->", os.path.relpath(FIG, ROOT))


# ---- F6: realized coefficients c_k = g_k w_k make the low-pass prior visible ----
cp = json.load(open(os.path.join(ROOT, "results", "coef_profile_v001.json")))
fig, ax = plt.subplots(figsize=(5.0, 3.2))
Kk = np.arange(len(cp["none"]["abs_norm"]))
series = [("none", grey, "o", "-"), ("jackson", blue, "s", "-"),
          ("geometric(low-pass)", green, "^", "-"), ("increasing(anti)", vermillion, "D", "--")]
labels = {"none": "no taper (free $w_k$)", "jackson": "Jackson (KPM)",
          "geometric(low-pass)": "geometric (low-pass)", "increasing(anti)": "anti"}
for name, col, mk, ls in series:
    y = np.array(cp[name]["abs_norm"])
    ax.semilogy(Kk, np.clip(y, 1e-3, None), mk + ls, color=col, lw=1.6, ms=5, label=labels[name])
ax.axhline(1.0, color="k", lw=0.6, ls=":")
ax.set_xlabel("polynomial order $k$")
ax.set_ylabel("realized $|c_k|/|c_0|$  ($c_k{=}w_k g_k$)")
ax.set_xticks(Kk[::2])
ax.legend(fontsize=6.8, loc="upper left", ncol=2)
ax.grid(True, alpha=0.25, which="both")
save(fig, "f6_coef_profile")


# ---- F5: the filter isn't the bottleneck (two falsifications), 5-seed mean +/- std ----
fx = json.load(open(os.path.join(ROOT, "results", "falsifications_seeds_v001.json")))
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.9))
# (a) sign-flip: flat wins at BOTH homophily and heterophily -> no frequency flip
sf = fx["sign_flip_hard_cSBM_d4_snr0"]
groups = ["homophily\n(h=0.85)", "heterophily\n(h=0.15)"]
keys = ["h=0.85_homophily", "h=0.15_heterophily"]
taps = [("decaying", "decaying", green), ("flat", "flat", blue), ("increasing", "anti", vermillion)]
xg = np.arange(2); w = 0.26
for i, (t, lab, c) in enumerate(taps):
    axa.bar(xg + (i - 1) * w, [sf[k][t]["mean"] for k in keys], w, color=c, label=lab,
            yerr=[sf[k][t]["std"] for k in keys], capsize=2, error_kw=dict(elinewidth=0.7))
axa.set_xticks(xg); axa.set_xticklabels(groups, fontsize=8)
axa.set_ylim(40, 100); axa.set_ylabel("cSBM test acc [%]")
axa.set_title("(a)", loc="left", fontweight="bold", fontsize=10)
axa.legend(fontsize=6.5, loc="upper center", ncol=3, columnspacing=1.0, handletextpad=0.4,
           title="per-order taper", title_fontsize=6.5)
# (b) required-K: trained accuracy peaks at K=1-2 and declines (variance), not approximation
rk = fx["required_K_data_rich"]
Ks = [1, 2, 4, 8, 16]
for name, col in [("Cora", blue), ("CiteSeer", orange)]:
    axb.errorbar(Ks, [rk[name][str(k)]["mean"] for k in Ks], yerr=[rk[name][str(k)]["std"] for k in Ks],
                 fmt="o-", color=col, lw=1.6, ms=5, capsize=2.5, elinewidth=1.0, label=name)
axb.set_xscale("log", base=2); axb.set_xticks(Ks); axb.set_xticklabels(Ks)
axb.set_xlabel("polynomial degree $K$"); axb.set_ylabel("test acc [%] (60% train)")
axb.set_title("(b)", loc="left", fontweight="bold", fontsize=10)
axb.legend(fontsize=7); axb.grid(True, alpha=0.25)
save(fig, "f5_filter_not_bottleneck")
