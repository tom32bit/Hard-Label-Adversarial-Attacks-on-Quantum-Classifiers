"""Evaluation metrics with ground-truth verification (plan Sec. 6).

A decision-based attack only ever sees *noisy* labels, so it can believe it has
crossed the boundary when it has not.  Every returned adversarial is therefore
re-checked against the exact (infinite-shot) model: a run counts as a success only
if the returned point is *truly* adversarial.  This is what exposes the failure of
the naive fixed-shot port and rewards the calibrated attack (RQ1/RQ3).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Ground-truth verification
# --------------------------------------------------------------------------- #
def verify(vqc, x0, x_adv, y0) -> dict:
    """Check a returned adversarial against the exact model.

    Returns verified success and the verified L2 perturbation (inf if the point is
    not truly adversarial).  ``exact_margin`` is the signed decision value at x_adv.
    """
    x0 = np.asarray(x0, float)
    x_adv = np.asarray(x_adv, float)
    f = float(vqc.decision_function(x_adv.reshape(1, -1))[0])
    truly_adv = (1 if f >= 0 else -1) != int(y0)
    pert = float(np.linalg.norm(x_adv - x0)) if truly_adv else float("inf")
    return {"verified_success": bool(truly_adv), "verified_perturbation": pert,
            "exact_margin_at_adv": abs(f)}


def shots_to_threshold(result_record: dict, verified: bool, eps: float) -> float:
    """Shots spent to first reach believed perturbation <= eps (inf if never / unverified)."""
    if not verified:
        return float("inf")
    dt = result_record.get("dist_trajectory", [])
    st = result_record.get("shot_trajectory", [])
    for d, s in zip(dt, st):
        if d <= eps:
            return float(s)
    return float("inf")


def queries_to_threshold(result_record: dict, verified: bool, eps: float) -> float:
    if not verified:
        return float("inf")
    dt = result_record.get("dist_trajectory", [])
    qt = result_record.get("query_trajectory", [])
    for d, q in zip(dt, qt):
        if d <= eps:
            return float(q)
    return float("inf")


# --------------------------------------------------------------------------- #
# Aggregation over an attacked image set
# --------------------------------------------------------------------------- #
def summarize_attack(perts, successes, *, cap: Optional[float] = None) -> dict:
    """Median/mean perturbation over verified successes + success rate + dispersion."""
    perts = np.asarray(perts, float)
    successes = np.asarray(successes, bool)
    ok = perts[successes & np.isfinite(perts)]
    out = {
        "success_rate": float(np.mean(successes)) if len(successes) else 0.0,
        "n_images": int(len(successes)),
        "n_success": int(len(ok)),
        "median_perturbation": float(np.median(ok)) if len(ok) else float("nan"),
        "mean_perturbation": float(np.mean(ok)) if len(ok) else float("nan"),
        "std_perturbation": float(np.std(ok)) if len(ok) else float("nan"),
    }
    if len(ok):
        out["dispersion"] = float(np.std(ok) / (abs(np.mean(ok)) + 1e-12))  # sigma/|mean|
    if cap is not None:                       # capped median counts failures at `cap`
        filled = np.where(successes & np.isfinite(perts), perts, cap)
        out["median_perturbation_capped"] = float(np.median(filled))
    return out


def robust_accuracy(perts, successes, eps: float) -> float:
    """Fraction of (clean-correct) images with no verified adversarial within eps."""
    perts = np.asarray(perts, float)
    successes = np.asarray(successes, bool)
    broken = successes & np.isfinite(perts) & (perts <= eps)
    return float(1.0 - np.mean(broken)) if len(successes) else float("nan")


def success_rate_at_eps(perts, successes, eps: float) -> float:
    perts = np.asarray(perts, float)
    successes = np.asarray(successes, bool)
    return float(np.mean(successes & np.isfinite(perts) & (perts <= eps)))


# --------------------------------------------------------------------------- #
# Seed aggregation (results.json schema) and correlation tests
# --------------------------------------------------------------------------- #
def seed_aggregate(values) -> dict:
    """{'mean','std','seeds':[...]} -- the plan's headline results schema."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return {"mean": float("nan"), "std": float("nan"), "seeds": list(values)}
    return {"mean": float(np.mean(v)), "std": float(np.std(v)),
            "sem": float(np.std(v) / np.sqrt(len(v))), "seeds": [float(x) for x in values]}


def pearson_with_p(x, y) -> dict:
    """Pearson r with p-value (correlation claims must report both, plan Sec. 6)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return {"r": float("nan"), "p": float("nan"), "n": int(m.sum())}
    r, p = stats.pearsonr(x[m], y[m])
    return {"r": float(r), "p": float(p), "n": int(m.sum())}
