"""Merge RQ2 fragments into one result file.

RQ2 is three independent sweeps that may be produced by different sessions (an
interrupted run salvaged from checkpoints, plus a later run that finishes the missing
sweep). This combines them, preferring the fragment with more seeds per point for each
sweep, so the merged file is the best available version of every panel.

Run:  python experiments/merge_rq2.py --inputs results/rq2.json results/rq2_partial_8seed.json \
                                      --out results/rq2.json
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hlq.analysis import load_json, save_json

SWEEPS = ("budget_curve", "allocation_sweep", "m1_m2_split")


def _seed_count(sweep: dict) -> int:
    """Minimum seeds per point in a sweep (0 if unknown) -- the quality we rank by."""
    sp = sweep.get("seeds_per_point")
    if sp:
        return min(int(v) for v in sp.values())
    # fall back to the seed spread recorded in the cells
    cells = sweep.get("cells") or []
    if not cells:
        return 0
    per = {}
    for c in cells:
        key = (c.get("force_probe_shots"), c["attack"].get("total_budget"),
               c["attack"].get("grad_budget_frac"))
        per[key] = per.get(key, 0) + 1
    return min(per.values()) if per else 0


def merge(paths) -> dict:
    frags = [(p, load_json(p)) for p in paths if os.path.exists(p)]
    out, provenance = {}, {}
    for sweep in SWEEPS:
        best, best_n, best_src = None, -1, None
        for p, d in frags:
            if sweep in d and d[sweep]:
                n = _seed_count(d[sweep])
                if n > best_n:
                    best, best_n, best_src = d[sweep], n, p
        if best is not None:
            out[sweep] = best
            provenance[sweep] = {"source": best_src, "min_seeds_per_point": best_n}
    out["_merged_from"] = [p for p, _ in frags]
    out["_provenance"] = provenance
    out["_meta"] = {"rq": "rq2", "merged": True}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = merge(args.inputs)
    save_json(res, args.out)
    print(f"[merge_rq2] -> {args.out}")
    for sweep, info in res.get("_provenance", {}).items():
        print(f"  {sweep:18s} from {info['source']}  (min seeds/point = {info['min_seeds_per_point']})")
    missing = [s for s in SWEEPS if s not in res]
    if missing:
        print(f"  still missing: {missing}")


if __name__ == "__main__":
    main()
