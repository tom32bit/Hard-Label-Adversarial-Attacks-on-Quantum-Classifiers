"""Calibrated hard-label attack -- the paper's core method (plan Sec. 4.3, M1+M2).

Two Born-rule-calibrated modifications over the naive HSJA port.

**M1 -- shot-calibrated boundary search.**  A hard-label binary search drives its
midpoints *onto* the boundary, where the margin ``m -> 0`` and the Born-rule flip
probability ``-> 1/2``: the sign there is unresolvable at *any* finite shot budget,
and a single false-positive "adversarial" reading collapses the search toward ``x0``
and never recovers (the noise-induced inward bias).  We therefore do not attempt the
impossible.  The shot budget defines a *boundary-localization resolution*

    tau(S) = z_delta * sqrt( (1 - f^2) / S )      (z_delta = Phi^{-1}(1 - delta)),

the Born-rule standard error of the S-shot estimator.  A point is declared
adversarial only when its estimate clears that resolution (``|f_hat| > tau``);
inside the band the decision is "too close to call" and is resolved *conservatively*
(treated as safe), which keeps the search outside the true boundary.  Shots are spent
**adaptively**: a cheap pilot settles easy, high-margin points immediately, and the
budget is escalated (doubling) only for points near the boundary, up to a cap.  This
is what a uniform-shot scheme (PopSkipJump) cannot do -- it pays the same price
everywhere.

**M2 -- budget-aware normal estimation.**  Given the per-iteration gradient budget
``T_grad``, the ``(B, S_probe)`` split is chosen by
:func:`hlq.budget.optimal_probe_split` from the locally estimated probe margin,
realising the interior optimum of RQ2.  ``force_probe_shots`` / ``force_num_probes``
let the RQ2 driver sweep the split by hand to trace the empirical curve.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ..budget import optimal_probe_split
from .base import DecisionBasedAttack

_Z_CACHE = {}


def _z(delta: float) -> float:
    """Phi^{-1}(1-delta), memoised: scipy's ppf is slow and this runs per decision."""
    z = _Z_CACHE.get(delta)
    if z is None:
        z = float(norm.ppf(1.0 - delta))
        _Z_CACHE[delta] = z
    return z


class CalibratedHSJA(DecisionBasedAttack):
    def __init__(self, cfg, decision_cap: int = None):
        super().__init__(cfg)
        self.force_probe_shots = None      # set by RQ2 sweep to force S_probe
        self.force_num_probes = None
        self._decision_cap_override = decision_cap
        self._m2_log = []                  # per-iteration (B, S_probe, margin) diagnostics
        self._decision_shots = []          # per-decision shot spend (M1 diagnostics)
        self._pilot_rng = np.random.default_rng(cfg.seed + 99)

    # -- M1: Born-calibrated adaptive shot escalation ----------------------- #
    def _pilot_shots(self) -> int:
        return int(max(8, min(32, self.cfg.fixed_shots // 3)))

    def _decision_cap(self) -> int:
        """Finest affordable resolution: tau_min ~ z/sqrt(cap). Bounds one decision's cost."""
        if self._decision_cap_override is not None:
            return int(self._decision_cap_override)
        return int(max(512, self.cfg.total_budget // 50))

    def _decide_adversarial(self, x, phase) -> bool:
        delta = self.cfg.delta_decision
        if phase == "init":
            delta = min(delta, 0.02)                    # very reliable initialisation
        z = _z(delta)
        cap = self._decision_cap()
        S_tot, f_acc, S = 0, 0.0, self._pilot_shots()
        while True:
            f_new = self._oracle.estimate_f(x, S)       # independent Binomial draw at same f
            f_acc = (f_acc * S_tot + f_new * S) / (S_tot + S)   # pooled estimate
            S_tot += S
            tau = z * np.sqrt(max(1.0 - f_acc * f_acc, 1e-6) / S_tot)
            if abs(f_acc) > tau:                        # resolved at confidence 1-delta
                self._decision_shots.append(S_tot)
                return (1 if f_acc >= 0 else -1) != self._y0
            if S_tot >= cap:
                # Unresolvable at the affordable budget: we have localised the boundary
                # to the shot-limited resolution.  Answer conservatively (treat as safe)
                # and stop bisecting -- pushing further only burns shots on coin flips.
                self._decision_shots.append(S_tot)
                self._halt_bisect = True
                return False
            S = min(S_tot, cap - S_tot)                 # escalate: double the budget

    # -- M2: budget-aware (B, S_probe) split -------------------------------- #
    def _estimate_probe_margin(self, x_b, x0) -> float:
        """Cheap pilot estimate of the typical |f| a probe sees at this boundary point."""
        d = x_b.size
        dist = float(np.linalg.norm(x_b - x0))
        delta_r = max(dist / d, 10 * self.cfg.bin_search_tol)
        u = self._pilot_rng.normal(size=(5, d))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        X = self._domain.project(x_b[None, :] + delta_r * u)
        fhat = self._oracle.estimate_f_batch(X, self._pilot_shots())
        return float(np.clip(np.median(np.abs(fhat)), 1e-3, 0.999))

    def _grad_budget(self, t, x_b, x0) -> tuple:
        T_iter = self.cfg.total_budget / max(1, self.cfg.iterations)
        T_grad = int(max(64, self.cfg.grad_budget_frac * T_iter))
        if self.force_probe_shots is not None:              # RQ2 manual sweep
            S = int(self.force_probe_shots)
            B = int(self.force_num_probes) if self.force_num_probes else max(8, T_grad // S)
            self._m2_log.append({"t": t, "S_probe": S, "B": B, "forced": True})
            return B, S
        margin = self._estimate_probe_margin(x_b, x0)
        B, S, _ = optimal_probe_split(T_grad, margin)
        self._m2_log.append({"t": t, "S_probe": S, "B": B, "margin_est": margin})
        return B, S

    def attack(self, *a, **k):
        res = super().attack(*a, **k)
        res.meta["m2_log"] = self._m2_log
        if self._decision_shots:
            res.meta["mean_decision_shots"] = float(np.mean(self._decision_shots))
            res.meta["n_decisions"] = int(len(self._decision_shots))
        return res
