"""Concentration diagnostics -- the RQ5 confound guardrail (plan Sec. 3, T6).

Exponential concentration [25,26] shrinks a VQC's output margin as qubit count grows.
A model deep in concentration is nearly *data-independent*: it looks spectacularly
"robust" (an attacker cannot move a decision value that barely varies) while being
useless as a classifier.  Every robustness number in this study is therefore
reported alongside:

* ``var_f``            -- variance of the decision value over the data (the
                          concentration order parameter; decays ~ exp(-b n));
* ``above_chance``     -- clean accuracy minus chance, the denominator that makes
                          a robustness claim meaningful;
* ``trivially_robust`` -- the T6 flag: clean accuracy at chance => report as
                          "trivially robust", never as "defended".
"""
from __future__ import annotations

import numpy as np
from scipy import stats

CHANCE = 0.5


def margin_stats(clf, X) -> dict:
    """Decision-value spread over a dataset: the concentration order parameter."""
    f = clf.decision_function(np.asarray(X))
    return {
        "var_f": float(np.var(f)),
        "std_f": float(np.std(f)),
        "mean_abs_f": float(np.mean(np.abs(f))),
        "median_abs_f": float(np.median(np.abs(f))),
        "max_abs_f": float(np.max(np.abs(f))),
    }


def trivially_robust(clean_acc: float, tol: float = 0.02) -> bool:
    """T6: at-chance clean accuracy means robustness is an artifact, not a defense."""
    return bool(clean_acc <= CHANCE + tol)


def robustness_guardrail(clean_acc: float, median_pert: float) -> dict:
    """Report robustness *relative to clean accuracy above chance* (H5)."""
    above = float(clean_acc - CHANCE)
    return {
        "clean_accuracy": float(clean_acc),
        "above_chance": above,
        "trivially_robust": trivially_robust(clean_acc),
        # perturbation per unit of genuine (above-chance) predictive power
        "robustness_per_above_chance": (float(median_pert) / above
                                        if above > 1e-6 and np.isfinite(median_pert)
                                        else float("nan")),
    }


def fit_exponential_concentration(ns, var_f) -> dict:
    """Fit Var[f] ~ exp(-b n): linear regression of log(Var) on n, with R^2 and p."""
    ns = np.asarray(ns, float)
    v = np.asarray(var_f, float)
    m = np.isfinite(v) & (v > 0)
    if m.sum() < 3:
        return {"decay_rate_b": float("nan"), "r_squared": float("nan"),
                "p_value": float("nan"), "n_points": int(m.sum())}
    res = stats.linregress(ns[m], np.log(v[m]))
    return {
        "decay_rate_b": float(-res.slope),          # Var ~ exp(-b n)
        "intercept": float(res.intercept),
        "r_squared": float(res.rvalue ** 2),
        "p_value": float(res.pvalue),
        "n_points": int(m.sum()),
        "model": "log(Var[f]) = intercept - b*n",
    }


def fit_inverse_n(ns, values) -> dict:
    """Fit y ~ a + c/n (the plan's 1/n scaling; requires R^2 > 0.95 to extrapolate)."""
    ns = np.asarray(ns, float)
    y = np.asarray(values, float)
    m = np.isfinite(y) & (ns > 0)
    if m.sum() < 3:
        return {"r_squared": float("nan"), "n_points": int(m.sum())}
    res = stats.linregress(1.0 / ns[m], y[m])
    return {
        "slope_c": float(res.slope), "intercept_a": float(res.intercept),
        "r_squared": float(res.rvalue ** 2), "p_value": float(res.pvalue),
        "n_points": int(m.sum()),
        "extrapolation_allowed": bool(res.rvalue ** 2 > 0.95),   # plan Sec. 6 gate
        "model": "y = a + c/n",
    }
