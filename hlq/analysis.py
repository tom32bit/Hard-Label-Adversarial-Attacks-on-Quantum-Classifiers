"""Derived analyses: budget curves, the RQ2 interior optimum, and model validation.

Complements :mod:`hlq.metrics` (per-run metrics) and :mod:`hlq.concentration`
(RQ5 diagnostics) -- this module turns collections of runs into the paper's claims.

``validate_p_flip_on_model`` is the second half of the double-step validation: T1
checks the closed-form flip model against Monte-Carlo at *synthetic* margins; this
checks it against the *real trained VQC's* margins, and reports Pearson r with a
p-value as the plan requires (|r| > 0.9 demands p < 0.001).
"""
from __future__ import annotations

import json
import os

import numpy as np

from .metrics import pearson_with_p
from .oracle import p_flip, sample_label


# --------------------------------------------------------------------------- #
# JSON I/O
# --------------------------------------------------------------------------- #
def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_default)
    return path


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# RQ2: interior optimum of the query/shot split
# --------------------------------------------------------------------------- #
def find_interior_optimum(xs, ys) -> dict:
    """Locate argmin of a budget-allocation curve and test whether it is *interior*.

    H2 predicts an interior optimum: too few shots -> coin-flip labels swamp the
    normal estimate; too many -> too few probes/queries to converge.  A boundary
    optimum is a legitimate (reportable) negative result for H2.
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    m = np.isfinite(ys)
    if m.sum() < 3:
        return {"is_interior": False, "reason": "insufficient finite points",
                "n_points": int(m.sum())}
    xs_f, ys_f = xs[m], ys[m]
    j = int(np.argmin(ys_f))
    interior = 0 < j < len(ys_f) - 1
    return {
        "argmin_x": float(xs_f[j]), "min_y": float(ys_f[j]),
        "is_interior": bool(interior),
        "index": j, "n_points": int(len(ys_f)),
        "edge_values": [float(ys_f[0]), float(ys_f[-1])],
        # how much better the optimum is than the worse edge (effect size)
        "gain_vs_worst_edge": float(max(ys_f[0], ys_f[-1]) - ys_f[j]),
    }


def budget_curve(records, x_key="total_budget", y_key="median_perturbation") -> dict:
    """Collect (x, y) from a list of aggregated cells, sorted by x."""
    pts = sorted(((r[x_key], r[y_key]) for r in records if np.isfinite(r.get(y_key, np.nan))),
                 key=lambda p: p[0])
    return {"x": [float(a) for a, _ in pts], "y": [float(b) for _, b in pts],
            "x_key": x_key, "y_key": y_key}


def monotone_non_increasing(xs, ys, tol=1e-3) -> bool:
    """T5: larger budget must never yield a *worse* median perturbation (within noise)."""
    order = np.argsort(np.asarray(xs, float))
    y = np.asarray(ys, float)[order]
    y = y[np.isfinite(y)]
    return bool(np.all(np.diff(y) <= tol))


# --------------------------------------------------------------------------- #
# Born-model validation on the real trained classifier
# --------------------------------------------------------------------------- #
def validate_p_flip_on_model(clf, X, shots=(10, 30, 100, 300, 1000), trials=400,
                             seed=0, max_points=200) -> dict:
    """Predicted vs empirical flip rate at the REAL model's margins (double-step step 2).

    For each test point the exact ``f`` gives a predicted ``p_flip``; the empirical
    rate comes from independently sampling ``trials`` S-shot labels.  Reports Pearson
    r with p-value per shot count and pooled.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)[:max_points]
    f = clf.decision_function(X)
    true_lab = np.where(f >= 0, 1, -1)
    per_shot, pred_all, emp_all = {}, [], []
    for S in shots:
        pred = p_flip(f, int(S))
        emp = np.empty(len(f))
        for i, fi in enumerate(f):
            labs = sample_label(np.full(trials, fi), int(S), rng)
            emp[i] = np.mean(labs != true_lab[i])
        per_shot[str(S)] = {
            "correlation": pearson_with_p(pred, emp),
            "mean_abs_error": float(np.mean(np.abs(pred - emp))),
            "predicted": pred.tolist(), "empirical": emp.tolist(),
        }
        pred_all.append(pred)
        emp_all.append(emp)
    pooled = pearson_with_p(np.concatenate(pred_all), np.concatenate(emp_all))
    return {
        "test": "p_flip_model_vs_empirical_on_trained_VQC",
        "shots": list(shots), "n_points": int(len(f)), "trials_per_point": int(trials),
        "per_shot": per_shot, "pooled_correlation": pooled,
        "margins": np.abs(f).tolist(),
        # plan Sec. 6 gate: |r| > 0.9 requires p < 0.001
        "passed": bool(abs(pooled["r"]) > 0.9 and pooled["p"] < 1e-3),
    }
