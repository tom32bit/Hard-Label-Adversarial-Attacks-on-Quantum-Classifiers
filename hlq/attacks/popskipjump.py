"""PopSkipJump baseline: decision-based attack for a *constant*-noise oracle [15].

The key comparison for RQ3.  PopSkipJump makes each decision reliable by repeating
the query and majority-voting, with the repeat count fixed from a single *global*
flip-rate estimate ``p0`` -- it is oracle-agnostic and does **not** use the Born-rule
margin structure.  Consequences at equal total budget: it over-spends far from the
boundary (where one shot would suffice) and under-spends near it (where the true
flip rate ~ 1/2 exceeds the global ``p0`` its repeat count was calibrated for).
The calibrated attack should beat it (H3).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom

from .base import DecisionBasedAttack


def repeats_for_target(p0: float, delta: float, r_max: int = 51) -> int:
    """Smallest odd R with majority-vote error P(Bin(R,p0) > R/2) <= delta."""
    for R in range(1, r_max + 1, 2):
        if binom.sf(R // 2, R, p0) <= delta:
            return R
    return r_max


class PopSkipJump(DecisionBasedAttack):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._p0 = None
        self._R = None

    def _calibrate_global_noise(self):
        """Fix the repeat count from the ASSUMED constant flip rate p0.

        This is the baseline's defining limitation and must not be estimated at a
        convenient point: measuring the flip rate far from the boundary returns ~0 and
        collapses the method to a single unrepeated query (i.e. the naive fixed-shot
        port).  PopSkipJump's premise is a constant, oracle-agnostic noise level, so we
        give it exactly that -- one repeat count applied uniformly everywhere.
        """
        self._p0 = float(min(max(self.cfg.psj_assumed_p0, 0.02), 0.45))
        self._R = repeats_for_target(self._p0, self.cfg.delta_decision)

    def _reliable_label(self, x) -> int:
        if self._p0 is None:
            self._calibrate_global_noise()
        S = self.cfg.fixed_shots
        labs = np.array([int(self._oracle.label(x, S)) for _ in range(self._R)])
        return 1 if labs.sum() >= 0 else -1

    def _decide_adversarial(self, x, phase) -> bool:
        return self._reliable_label(x) != self._y0

    def _grad_budget(self, t, x_b, x0) -> tuple:
        return self.cfg.num_probes, self.cfg.probe_shots

    def attack(self, *a, **k):
        res = super().attack(*a, **k)
        res.meta["popskipjump_p0"] = self._p0
        res.meta["popskipjump_repeats"] = self._R
        return res
