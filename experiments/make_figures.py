"""Research-grade figures from the results JSON (plan Sec. 8).

Design rules applied throughout:
* categorical hues assigned in FIXED order from a validated colorblind-safe palette
  (never cycled), always paired with a second encoding (marker + linestyle) because
  the palette's worst adjacent tritan separation sits in the 6-8 dE band;
* magnitude uses a single-hue sequential ramp; never a rainbow;
* NEVER a dual y-axis -- two measures become two stacked panels sharing an x-axis;
* legend for >=2 series, direct labels where <=4, recessive grid/axes, thin marks;
* the figures carry the result geometrically -- no prose annotations asserting a
  conclusion; only data, error bars (std over seeds), fitted curves and references.

Run:  python experiments/make_figures.py [--results results] [--figures figures]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

from hlq.budget import grad_snr_per_budget
from hlq.oracle import p_flip

# --- validated categorical palette (fixed order) + secondary encodings ------- #
PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834",
           "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)),
              (0, (4, 2, 1, 2))]
SEQ_CMAP = "Blues"                      # single hue, light -> dark
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e3e3e0"

# stable slot per method so colour follows the entity, never its rank
METHOD_SLOT = {"calibrated_hsja": 0, "fixed_hsja": 5, "popskipjump": 3,
               "pgd_whitebox": 1, "transfer": 6, "classical_hsja": 4}
METHOD_LABEL = {"calibrated_hsja": "Calibrated HSJA (ours, M1+M2)",
                "fixed_hsja": "Fixed-shot HSJA (naive port)",
                "popskipjump": "PopSkipJump (constant noise)",
                "pgd_whitebox": "White-box PGD (reference)",
                "transfer": "Classical-surrogate transfer",
                "classical_hsja": "HSJA on matched classical NN"}
# short two-line forms for categorical axes (the long ones collide)
METHOD_SHORT = {"calibrated_hsja": "Calibrated\n(ours)",
                "fixed_hsja": "Fixed-shot\n(naive)",
                "popskipjump": "PopSkipJump\n(const. noise)",
                "pgd_whitebox": "White-box PGD\n(reference)",
                "transfer": "Transfer\n(surrogate)",
                "classical_hsja": "Classical NN\n(anchor)"}


def bar_label(ax, x, mu, sd, fmt="{:.2f}"):
    """Direct label placed clear of the error-bar cap (the contrast relief rule)."""
    if not np.isfinite(mu):
        return
    top = mu + (sd if np.isfinite(sd) else 0.0)
    ax.annotate(fmt.format(mu), (x, top), xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=7.5, color=INK)


def style():
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "grid.alpha": 0.9, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.facecolor": "white",
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
    })


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {name}.png/.pdf")


def load(results, name):
    p = os.path.join(results, f"{name}.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def _mean_std(agg, cond, metric):
    d = agg[cond][metric]
    return d.get("mean", np.nan), d.get("std", np.nan)


# --------------------------------------------------------------------------- #
# T1 -- the Born-rule flip model (closed form vs Monte-Carlo)
# --------------------------------------------------------------------------- #
def fig_flip_model(sanity, out):
    t1 = sanity.get("T1")
    if not t1:
        return
    m = np.array(t1["margins"])
    S = np.array(t1["shots"])
    closed = np.array(t1["p_flip_closed"])
    mc = np.array(t1["p_flip_montecarlo"])

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1))
    ax = axes[0]
    cmap = plt.get_cmap(SEQ_CMAP)
    cols = cmap(np.linspace(0.35, 0.95, len(S)))          # sequential: shots = magnitude
    mm = np.linspace(1e-3, 0.99, 300)
    y_lab = 0.15                      # each curve meets this level at its own m ~ z/sqrt(S)
    for j, s in enumerate(S):
        ax.plot(mm, p_flip(mm, int(s)), color=cols[j], lw=1.6, zorder=2)
        ax.plot(m, mc[:, j], ls="none", marker="o", ms=3.4, mfc="white",
                mec=cols[j], mew=0.9, zorder=3)
        # direct-label along the curve so the labels spread instead of colliding at m->0
        m_lab = float(np.clip(1.036 / np.sqrt(float(s)), 0.012, 0.92))
        ax.annotate(f"S={s}", (m_lab, y_lab), xytext=(4, 6),
                    textcoords="offset points", color=cols[j], fontsize=7,
                    va="bottom", ha="left", rotation=0)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax.set_xlabel(r"Born margin  $m=|f_\theta(x)|$")
    ax.set_ylabel(r"flip probability  $p_{\mathrm{flip}}$")
    ax.set_xlim(-0.01, 0.8)
    ax.set_ylim(-0.02, 0.55)
    ax.set_title("Closed form (lines) vs Monte-Carlo (points)", color=INK)

    ax = axes[1]
    lim = max(closed.max(), mc.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=MUTED, lw=0.9, ls="--", zorder=1)
    for j, s in enumerate(S):
        ax.plot(closed[:, j], mc[:, j], ls="none", marker=MARKERS[j % len(MARKERS)],
                ms=3.6, mfc="white", mec=cols[j], mew=0.9, label=f"S={s}", zorder=3)
    ax.set_xlabel(r"predicted $p_{\mathrm{flip}}$ (CLT model)")
    ax.set_ylabel(r"empirical $p_{\mathrm{flip}}$ (Binomial MC)")
    ax.set_title("Model vs measurement", color=INK)
    ax.legend(ncol=2, loc="lower right")
    fig.suptitle("T1  Born-rule flip model validated against the exact stochastic oracle",
                 y=1.03, fontsize=10.5, color=INK)
    save(fig, out, "fig_T1_flip_model")


# --------------------------------------------------------------------------- #
# T2 / T3 -- limits and ground truth
# --------------------------------------------------------------------------- #
def fig_limits(sanity, out):
    t2, t3 = sanity.get("T2"), sanity.get("T3")
    if not (t2 or t3):
        return
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.1))

    ax = axes[0]
    if t2:
        sp = np.array(t2["probe_shots"], float)
        pert = np.array(t2["calibrated_median_perturbation"], float)
        det = t2["deterministic_hsja"]["median_perturbation"]
        ax.axhline(det, color=INK2, lw=1.2, ls="--", zorder=2)
        ax.annotate("deterministic HSJA  ($S\\to\\infty$)", (sp[0], det),
                    xytext=(2, 5), textcoords="offset points", fontsize=7.5, color=INK2)
        ax.plot(sp, pert, color=PALETTE[0], lw=1.8, marker=MARKERS[0], ms=5,
                mfc="white", mew=1.4, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("probe shots  $S_{\\mathrm{probe}}$   (decision cap $\\propto S$)")
        ax.set_ylabel(r"median $\ell_2$ perturbation")
        ax.set_title("T2  median vs shot budget", color=INK, fontsize=9.5)

    ax = axes[1]
    gaps = (t2 or {}).get("paired_relative_gap")
    if gaps:
        # the paired per-image gap is the statistic that actually measures convergence:
        # the median above is taken over whichever images succeeded, so it also moves
        # with the success set.
        sp = np.array(t2["probe_shots"], float)
        ax.plot(sp, gaps, color=PALETTE[5], lw=1.8, marker=MARKERS[1], ms=5,
                mfc="white", mew=1.4, ls=LINESTYLES[1], zorder=3)
        ax.axhline(0.15, color=MUTED, lw=1.0, ls=":", zorder=2)
        ax.annotate("15% tolerance", (sp[0], 0.15), xytext=(2, 4),
                    textcoords="offset points", fontsize=7.5, color=INK2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"probe shots  $S_{\mathrm{probe}}$")
        ax.set_ylabel("paired relative gap to deterministic")
        ax.set_title("T2  per-image convergence", color=INK, fontsize=9.5)

    ax = axes[2]
    if t3:
        rec = t3["per_point"]
        true = np.array([r["true_distance"] for r in rec])
        got = np.array([r["recovered"] for r in rec])
        lim = max(true.max(), got.max()) * 1.08
        ax.plot([0, lim], [0, lim], color=MUTED, lw=0.9, ls="--", zorder=1)
        ax.plot(true, got, ls="none", marker=MARKERS[1], ms=5.5, mfc="white",
                mec=PALETTE[1], mew=1.3, zorder=3)
        ax.set_xlabel("true point-to-boundary distance")
        ax.set_ylabel("perturbation recovered by attack")
        ax.set_title("T3  known analytic boundary", color=INK, fontsize=9.5)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    fig.tight_layout()
    save(fig, out, "fig_T2_T3_limits")


# --------------------------------------------------------------------------- #
# RQ1 -- feasibility across the attack suite
# --------------------------------------------------------------------------- #
def fig_rq1(rq1, out):
    if not rq1:
        return
    agg = rq1["aggregated"]
    order = [m for m in ["calibrated_hsja", "popskipjump", "fixed_hsja", "transfer",
                         "classical_hsja", "pgd_whitebox"] if m in agg]
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True)

    ax = axes[0]
    for i, mth in enumerate(order):
        mu, sd = _mean_std(agg, mth, "success_rate")
        ax.bar(i, mu, yerr=sd, width=0.62, color=PALETTE[METHOD_SLOT[mth]],
               ecolor=INK2, capsize=3, error_kw={"lw": 1.0}, zorder=3)
        bar_label(ax, i, mu, sd)                      # direct label (relief rule)
    ax.set_ylabel("verified attack success rate")
    ax.set_ylim(0, 1.18)

    ax = axes[1]
    for i, mth in enumerate(order):
        mu, sd = _mean_std(agg, mth, "median_perturbation")
        ax.bar(i, mu, yerr=sd, width=0.62, color=PALETTE[METHOD_SLOT[mth]],
               ecolor=INK2, capsize=3, error_kw={"lw": 1.0}, zorder=3)
        bar_label(ax, i, mu, sd)
    ax.set_ylabel(r"median $\ell_2$ perturbation")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([METHOD_SHORT[m] for m in order], fontsize=7.5)
    fig.suptitle("RQ1  Hard-label attacks on a variational quantum classifier\n"
                 "(success is verified against the exact model; bars are mean $\\pm$ s.d. over seeds)",
                 y=1.0, fontsize=10, color=INK)
    save(fig, out, "fig_RQ1_feasibility")


# --------------------------------------------------------------------------- #
# RQ2 -- budget economics
# --------------------------------------------------------------------------- #
def fig_rq2(rq2, out):
    if not rq2:
        return
    # (a) budget curve
    bc = rq2.get("budget_curve")
    if bc:
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        agg = bc["aggregated"]
        xs = sorted(int(k) for k in agg)
        mu = [agg[str(x)]["median_perturbation"]["mean"] if str(x) in agg
              else agg[x]["median_perturbation"]["mean"] for x in xs]
        sd = [agg[str(x)]["median_perturbation"]["std"] if str(x) in agg
              else agg[x]["median_perturbation"]["std"] for x in xs]
        mu, sd = np.array(mu, float), np.array(sd, float)
        ax.plot(xs, mu, color=PALETTE[0], lw=1.8, marker=MARKERS[0], ms=5,
                mfc="white", mew=1.4, zorder=3)
        ax.fill_between(xs, mu - sd, mu + sd, color=PALETTE[0], alpha=0.16, lw=0, zorder=2)
        ax.set_xscale("log")
        ax.set_xlabel("total measurement budget  $T$  (shots)")
        ax.set_ylabel(r"median $\ell_2$ perturbation")
        ax.set_title("RQ2  Perturbation vs measurement budget", color=INK)
        save(fig, out, "fig_RQ2_budget_curve")

    # (b) allocation sweep + theory  -- two stacked panels, NEVER a dual axis
    al = rq2.get("allocation_sweep")
    if al:
        fig, axes = plt.subplots(2, 1, figsize=(5.6, 5.0), sharex=True)
        xs = np.array(al["probe_shots"], float)
        ys = np.array(al["median_perturbation"], float)
        agg = al["aggregated"]
        sd = np.array([agg[str(int(x))]["median_perturbation"]["std"]
                       if str(int(x)) in agg else np.nan for x in xs], float)
        ax = axes[0]
        ax.plot(xs, ys, color=PALETTE[0], lw=1.8, marker=MARKERS[0], ms=5,
                mfc="white", mew=1.4, zorder=3)
        ax.fill_between(xs, ys - sd, ys + sd, color=PALETTE[0], alpha=0.16, lw=0, zorder=2)
        io = al.get("interior_optimum", {})
        if io.get("argmin_x") is not None and np.isfinite(io.get("min_y", np.nan)):
            ax.plot([io["argmin_x"]], [io["min_y"]], marker="*", ms=13,
                    color=PALETTE[5], ls="none", zorder=4)
        ax.set_xscale("log")
        ax.set_ylabel(r"median $\ell_2$ perturbation")
        ax.set_title("RQ2  Shot allocation of the normal estimate at fixed $T$", color=INK)

        ax = axes[1]                       # the closed-form objective being optimised
        Sg = np.unique(np.round(np.geomspace(1, max(xs.max(), 2), 200)).astype(int))
        for j, mgn in enumerate([0.02, 0.05, 0.10, 0.20]):
            g = np.array([grad_snr_per_budget(mgn, int(s)) for s in Sg])
            ax.plot(Sg, g / g.max(), lw=1.5, color=PALETTE[[1, 3, 4, 6][j]],
                    ls=LINESTYLES[j], label=f"$m$={mgn:g}", zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"probe shots  $S_{\mathrm{probe}}$   ($B=T_{\mathrm{grad}}/S$)")
        ax.set_ylabel("normalised  $(1-2p_{\\mathrm{flip}})^2/S$")
        ax.legend(ncol=4, loc="lower left")
        save(fig, out, "fig_RQ2_allocation")

    # (c) M1/M2 split
    sp = rq2.get("m1_m2_split")
    if sp:
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        xs = np.array(sp["grad_budget_frac"], float)
        ys = np.array(sp["median_perturbation"], float)
        ax.plot(xs, ys, color=PALETTE[0], lw=1.8, marker=MARKERS[0], ms=5,
                mfc="white", mew=1.4, zorder=3)
        io = sp.get("interior_optimum", {})
        if io.get("argmin_x") is not None and np.isfinite(io.get("min_y", np.nan)):
            ax.plot([io["argmin_x"]], [io["min_y"]], marker="*", ms=13,
                    color=PALETTE[5], ls="none", zorder=4)
        ax.set_xlabel("fraction of budget to normal estimation (M2)   $\\leftarrow$ M1 boundary search")
        ax.set_ylabel(r"median $\ell_2$ perturbation")
        ax.set_title("RQ2  Splitting the budget between boundary search and gradient",
                     color=INK)
        save(fig, out, "fig_RQ2_m1_m2_split")


# --------------------------------------------------------------------------- #
# RQ3 -- calibration payoff
# --------------------------------------------------------------------------- #
def fig_rq3(rq3, out):
    if not rq3:
        return
    agg = rq3["aggregated"]
    methods = rq3["methods"]
    budgets = rq3["budgets"]
    fig, axes = plt.subplots(2, 1, figsize=(5.8, 5.4), sharex=True)
    for i, mth in enumerate(methods):
        xs, su, sd_s, pe, sd_p = [], [], [], [], []
        for T in budgets:
            k = f"{mth}@T={T}"
            if k not in agg:
                continue
            xs.append(T)
            su.append(agg[k]["success_rate"]["mean"])
            sd_s.append(agg[k]["success_rate"]["std"])
            pe.append(agg[k]["median_perturbation"]["mean"])
            sd_p.append(agg[k]["median_perturbation"]["std"])
        c = PALETTE[METHOD_SLOT[mth]]
        axes[0].errorbar(xs, su, yerr=sd_s, color=c, lw=1.8, marker=MARKERS[i], ms=5,
                         mfc="white", mew=1.4, ls=LINESTYLES[i], capsize=2.5,
                         label=METHOD_LABEL[mth], zorder=3)
        axes[1].errorbar(xs, pe, yerr=sd_p, color=c, lw=1.8, marker=MARKERS[i], ms=5,
                         mfc="white", mew=1.4, ls=LINESTYLES[i], capsize=2.5, zorder=3)
    axes[0].set_ylabel("verified success rate")
    axes[0].set_ylim(-0.05, 1.1)
    axes[0].legend(loc="best")
    axes[1].set_ylabel(r"median $\ell_2$ perturbation")
    axes[1].set_xlabel("total measurement budget  $T$  (shots)")
    axes[1].set_xscale("log")
    fig.suptitle("RQ3  Born-rule calibration vs a constant-noise treatment at equal budget",
                 y=1.0, fontsize=10, color=INK)
    save(fig, out, "fig_RQ3_calibration")


# --------------------------------------------------------------------------- #
# RQ4 -- defenses vs a gradient-free attacker
# --------------------------------------------------------------------------- #
def fig_rq4(rq4, out):
    if not rq4:
        return
    agg = rq4["aggregated"]
    defs_ = ["none", "depolarizing", "randomized_encoding"]
    atks = ["calibrated_hsja", "pgd_whitebox"]

    def _key(d, a):
        """Resolve 'd|a', or the largest-n variant 'd|a|n=..' under an HLQ_RQ4_NS sweep."""
        if f"{d}|{a}" in agg:
            return f"{d}|{a}"
        cands = [k for k in agg if k.startswith(f"{d}|{a}|n=")]
        return max(cands, key=lambda s: int(s.split("n=")[1])) if cands else None

    if not any(_key(d, a) for d in defs_ for a in atks):
        return
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.4), sharex=True)
    w = 0.36
    for ai, a in enumerate(atks):
        xs, mu, sd, cl = [], [], [], []
        for di, d in enumerate(defs_):
            k = _key(d, a)
            if k is None:
                continue
            xs.append(di + (ai - 0.5) * w)
            mu.append(agg[k]["median_perturbation"]["mean"])
            sd.append(agg[k]["median_perturbation"]["std"])
            cl.append(agg[k]["clean_accuracy"]["mean"])
        axes[0].bar(xs, mu, yerr=sd, width=w, color=PALETTE[METHOD_SLOT[a]],
                    ecolor=INK2, capsize=3, error_kw={"lw": 1.0},
                    label=METHOD_LABEL[a], zorder=3)
    axes[0].set_ylabel(r"median $\ell_2$ perturbation")
    axes[0].legend(loc="best")

    # the RQ5 guardrail travels with every robustness number
    for di, d in enumerate(defs_):
        k = f"{d}|calibrated_hsja"
        if k not in agg:
            continue
        acc = agg[k]["clean_accuracy"]["mean"]
        sd_acc = agg[k]["clean_accuracy"]["std"]
        axes[1].bar(di, acc, yerr=sd_acc, width=0.5,
                    color=PALETTE[2], ecolor=INK2, capsize=3, zorder=3)
        bar_label(axes[1], di, acc, sd_acc)
    axes[1].axhline(0.5, color=INK2, lw=1.1, ls="--", zorder=4)
    axes[1].annotate("chance", (len(defs_) - 0.55, 0.5), xytext=(0, 3),
                     textcoords="offset points", fontsize=7.5, color=INK2)
    axes[1].set_ylabel("clean accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(range(len(defs_)))
    axes[1].set_xticklabels(["no defense", "depolarizing noise", "randomized encoding"])
    fig.suptitle("RQ4  Defenses validated against gradient attacks, re-tested gradient-free",
                 y=1.0, fontsize=10, color=INK)
    save(fig, out, "fig_RQ4_defenses")


# --------------------------------------------------------------------------- #
# RQ5 -- concentration vs robustness
# --------------------------------------------------------------------------- #
def fig_rq5(rq5, sanity, out):
    fits = (rq5 or {}).get("concentration_fits")
    t6 = (sanity or {}).get("T6")
    if not (fits or t6):
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    fig.subplots_adjust(wspace=0.32)

    ax = axes[0]
    if fits:
        for i, (obs, d) in enumerate(fits.items()):
            ns = np.array(d["n_qubits"], float)
            v = np.array(d["var_f"], float)
            c = PALETTE[[0, 5][i % 2]]
            ax.plot(ns, v, color=c, lw=1.8, marker=MARKERS[i], ms=5.5, mfc="white",
                    mew=1.4, ls=LINESTYLES[i], zorder=3,
                    label=("local $Z_0$" if obs == "local_z" else r"global $Z^{\otimes n}$"))
            f = d["exponential_fit"]
            if np.isfinite(f.get("decay_rate_b", np.nan)):
                xx = np.linspace(ns.min(), ns.max(), 50)
                ax.plot(xx, np.exp(f["intercept"] - f["decay_rate_b"] * xx), color=c,
                        lw=0.9, ls=":", zorder=2)
    elif t6:
        ns = np.array([r["n_qubits"] for r in t6["random_deep_global_z"]], float)
        v = np.array(t6["var_f"], float)
        ax.plot(ns, v, color=PALETTE[5], lw=1.8, marker=MARKERS[1], ms=5.5,
                mfc="white", mew=1.4, label=r"random deep, global $Z^{\otimes n}$", zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("qubits  $n$")
    ax.set_ylabel(r"$\mathrm{Var}[f_\theta(x)]$")
    ax.set_title("Decision-value variance (concentration)", color=INK, fontsize=9.5)
    ax.legend(loc="best")

    ax = axes[1]
    agg = (rq5 or {}).get("aggregated")
    if agg:
        for i, obs in enumerate(rq5["observables"]):
            ns, pert, acc = [], [], []
            for n in rq5["n_qubits"]:
                k = f"{obs}|n={n}"
                if k not in agg:
                    continue
                ns.append(n)
                pert.append(agg[k]["median_perturbation"]["mean"])
                acc.append(agg[k]["clean_accuracy"]["mean"])
            c = PALETTE[[0, 5][i % 2]]
            # robustness is only meaningful above chance -> size marks by that margin
            sizes = 20 + 260 * np.clip(np.array(acc) - 0.5, 0, None)
            ax.plot(ns, pert, color=c, lw=1.5, ls=LINESTYLES[i], zorder=2,
                    label=("local $Z_0$" if obs == "local_z" else r"global $Z^{\otimes n}$"))
            ax.scatter(ns, pert, s=sizes, facecolor="white", edgecolor=c, linewidth=1.4,
                       marker=MARKERS[i], zorder=3)
        ax.set_xlabel("qubits  $n$")
        ax.set_ylabel(r"median $\ell_2$ perturbation")
        ax.set_title(r"Apparent robustness (mark size $\propto$ acc. above chance)",
                     color=INK, fontsize=9.5)
        ax.legend(loc="best")
    fig.suptitle("RQ5  Separating genuine robustness from exponential concentration",
                 y=1.06, fontsize=10.5, color=INK)
    save(fig, out, "fig_RQ5_concentration")


# --------------------------------------------------------------------------- #
# T5 -- budget monotonicity on a common success set
# --------------------------------------------------------------------------- #
def fig_t5(sanity, out):
    t5 = (sanity or {}).get("T5")
    if not t5 or "median_perturbation_common_set" not in t5:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    T = np.array(t5["budgets"], float)
    ax.plot(T, t5["median_perturbation_common_set"], color=PALETTE[0], lw=1.8,
            marker=MARKERS[0], ms=5, mfc="white", mew=1.4, ls=LINESTYLES[0],
            label=f"common success set (n={t5.get('n_common_images','?')})", zorder=3)
    ax.plot(T, t5["median_perturbation_all_successes"], color=PALETTE[3], lw=1.5,
            marker=MARKERS[1], ms=4.5, mfc="white", mew=1.2, ls=LINESTYLES[1],
            label="all successes (selection-biased)", zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("total measurement budget  $T$  (shots)")
    ax.set_ylabel(r"median $\ell_2$ perturbation")
    ax.set_title("T5  Budget monotonicity", color=INK)
    ax.legend(loc="best")
    save(fig, out, "fig_T5_budget_monotonicity")


# --------------------------------------------------------------------------- #
# Ablations
# --------------------------------------------------------------------------- #
def fig_ablation(res, out, key, name, title, labels=None):
    if not res:
        return
    agg = res["aggregated"]
    conds = list(agg)
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 5.0), sharex=True)
    for i, c in enumerate(conds):
        mu, sd = _mean_std(agg, c, "median_perturbation")
        axes[0].bar(i, mu, yerr=sd, width=0.55, color=PALETTE[i % len(PALETTE)],
                    ecolor=INK2, capsize=3, zorder=3)
        bar_label(axes[0], i, mu, sd)
        mu2, sd2 = _mean_std(agg, c, "clean_accuracy")
        axes[1].bar(i, mu2, yerr=sd2, width=0.55, color=PALETTE[i % len(PALETTE)],
                    ecolor=INK2, capsize=3, zorder=3)
    axes[0].set_ylabel(r"median $\ell_2$ perturbation")
    axes[1].axhline(0.5, color=INK2, lw=1.0, ls="--", zorder=4)
    axes[1].set_ylabel("clean accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(range(len(conds)))
    axes[1].set_xticklabels([str(labels[c] if labels and c in labels else c)
                             for c in conds])
    fig.suptitle(title, y=1.0, fontsize=10, color=INK)
    save(fig, out, name)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--results", default=os.path.join(root, "results"))
    ap.add_argument("--figures", default=os.path.join(root, "figures"))
    args = ap.parse_args()
    style()
    print(f"[figures] reading {args.results}")

    sanity = load(args.results, "sanity") or {}
    if sanity:
        fig_flip_model(sanity, args.figures)
        fig_limits(sanity, args.figures)
    fig_rq1(load(args.results, "rq1"), args.figures)
    fig_rq2(load(args.results, "rq2"), args.figures)
    fig_rq3(load(args.results, "rq3"), args.figures)
    fig_rq4(load(args.results, "rq4"), args.figures)
    fig_rq5(load(args.results, "rq5"), sanity, args.figures)
    fig_t5(sanity, args.figures)
    fig_ablation(load(args.results, "ablation_encoding"), args.figures,
                 "encoding", "fig_ablation_encoding",
                 "Ablation C  Encoding vs hard-label attackability")
    fig_ablation(load(args.results, "ablation_depth"), args.figures,
                 "n_layers", "fig_ablation_depth",
                 "Ablation  Ansatz depth vs hard-label attackability")
    fig_ablation(load(args.results, "ablation_dataset"), args.figures,
                 "dataset", "fig_ablation_dataset",
                 "Ablation  Dataset vs hard-label attackability")
    print(f"[figures] done -> {args.figures}")


if __name__ == "__main__":
    main()
