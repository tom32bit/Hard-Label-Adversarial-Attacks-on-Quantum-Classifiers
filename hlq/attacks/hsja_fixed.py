"""Naive HSJA port: fixed shots everywhere, no Born-rule calibration (baseline).

This is the "port an attack mechanically" control (plan Sec. 4.4).  Every decision
uses a constant ``S`` and every gradient estimate a constant ``(B, S_probe)``.
With ``fixed_shots=None`` the oracle is deterministic, giving the *classical
anchor* (HSJA on a classical NN) for free from the same skeleton.
"""
from __future__ import annotations

from .base import DecisionBasedAttack


class FixedShotHSJA(DecisionBasedAttack):
    def _decide_adversarial(self, x, phase) -> bool:
        S = self.cfg.init_eval_shots if phase == "init" else self.cfg.fixed_shots
        return int(self._oracle.label(x, S)) != self._y0

    def _grad_budget(self, t, x_b, x0) -> tuple:
        return self.cfg.num_probes, self.cfg.probe_shots
