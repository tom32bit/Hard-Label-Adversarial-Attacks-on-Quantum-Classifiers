"""Significance tests for the head-to-head claims (plan Sec. 6).

Reads the per-image records already stored in results/*.json and reports, for the
load-bearing contrasts, a bootstrap CI on the headline statistic plus a hypothesis
test with an effect size -- so "calibrated beats X" is backed by a p-value, not just
non-overlapping means. Pure post-processing: no experiments are re-run.

Run:  python experiments/significance.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hlq.analysis import load_json, save_json
from hlq.metrics import bootstrap_ci, compare_perturbations, proportion_test

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def _pool(cells):
    """Pool per-image (perturbation, success) across a list of cells."""
    perts, succ = [], []
    for c in cells:
        for r in c.get("per_image", []):
            perts.append(r.get("verified_perturbation", float("inf")))
            succ.append(bool(r.get("verified_success", False)))
    return np.array(perts, float), np.array(succ, bool)


def _by(cells, keyfn):
    out = {}
    for c in cells:
        out.setdefault(keyfn(c), []).append(c)
    return out


def _contrast(name, cells_a, cells_b):
    pa, sa = _pool(cells_a)
    pb, sb = _pool(cells_b)
    return {
        "contrast": name,
        "success_rate": proportion_test(int(sa.sum()), len(sa), int(sb.sum()), len(sb)),
        "perturbation_mannwhitney": compare_perturbations(pa[sa], pb[sb]),
        "median_ci_a": bootstrap_ci(pa[sa]),
        "median_ci_b": bootstrap_ci(pb[sb]),
    }


def analyse():
    out = {}

    rq1 = load_json(os.path.join(RESULTS, "rq1.json"))
    by = _by(rq1["cells"], lambda c: c["attack"]["name"])
    # our method vs each hard-label baseline
    out["rq1"] = {c: _contrast(f"calibrated_vs_{c}", by["calibrated_hsja"], by[c])
                  for c in ("popskipjump", "fixed_hsja") if c in by}

    rq3 = load_json(os.path.join(RESULTS, "rq3.json"))
    by = _by(rq3["cells"], lambda c: (c["attack"]["name"], c["attack"]["total_budget"]))
    budgets = sorted({b for (_, b) in by})
    out["rq3"] = {}
    for T in budgets:
        ca = by.get(("calibrated_hsja", T), [])
        for base in ("popskipjump", "fixed_hsja"):
            cb = by.get((base, T), [])
            if ca and cb:
                out["rq3"][f"T={T}:calibrated_vs_{base}"] = _contrast(
                    f"calibrated_vs_{base}@T={T}", ca, cb)

    rq4 = load_json(os.path.join(RESULTS, "rq4.json"))
    # defense effect on the calibrated (gradient-free) attacker: none vs each defense
    def is_cal(c):
        return c["attack"]["name"] == "calibrated_hsja"
    by = _by([c for c in rq4["cells"] if is_cal(c)], lambda c: c["defense"]["name"])
    out["rq4"] = {}
    if "none" in by:
        for d in ("depolarizing", "randomized_encoding"):
            if d in by:
                out["rq4"][f"none_vs_{d}"] = _contrast(f"none_vs_{d}", by["none"], by[d])
                # is the defended clean accuracy significantly below chance?
                accs = [c["summary"]["clean_accuracy"] for c in by[d]]
                out["rq4"][f"{d}_clean_acc_ci"] = bootstrap_ci(accs, stat="mean")

    path = os.path.join(RESULTS, "significance.json")
    save_json(out, path)

    # human-readable digest
    print(f"[significance] wrote {path}\n")
    for rq, blk in out.items():
        print(f"=== {rq} ===")
        for name, res in blk.items():
            if "success_rate" in res:
                sr = res["success_rate"]
                mw = res["perturbation_mannwhitney"]
                print(f"  {name}: success {sr.get('prop_a', float('nan')):.2f} vs "
                      f"{sr.get('prop_b', float('nan')):.2f} (p={sr.get('p_value', float('nan')):.1e}); "
                      f"pert MWU p={mw.get('p_value', float('nan')):.1e} "
                      f"effect={mw.get('rank_biserial_effect', float('nan')):+.2f}")
            elif "point" in res:
                print(f"  {name}: {res['point']:.3f}  CI[{res['lo']:.3f}, {res['hi']:.3f}]")
    return out


if __name__ == "__main__":
    analyse()
