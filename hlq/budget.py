"""Query/shot economics of the boundary-normal estimate (RQ2, plan Sec. 4.3-M2).

At a boundary point the attacker estimates the normal by averaging the +-1 labels of
``B`` random probes, each measured with ``S_probe`` shots.  A probe's label is the
true side with probability ``1 - p_flip(m, S_probe)``, so its signed contribution is
attenuated by ``(1 - 2 p_flip)``.  Averaging ``B = T_grad / S_probe`` such probes gives

    SNR^2(direction)  ∝  B (1 - 2 p_flip)^2  =  T_grad * (1 - 2 p_flip(m, S))^2 / S.

For fixed total gradient budget ``T_grad`` the optimum is therefore the ``S`` that
maximises ``g(S) = (1 - 2 p_flip(m, S))^2 / S`` -- an interior optimum (RQ2/H2):
too few shots -> labels near the boundary are coin-flips (numerator -> 0); too many
shots -> too few probes (denominator grows).  This module is the single source of
that objective, used both by the calibrated attack (M2) and by the RQ2 theory curve.
"""
from __future__ import annotations

import numpy as np

from .oracle import p_flip


def grad_snr_per_budget(margin: float, S) -> np.ndarray:
    """The objective g(S) = (1 - 2 p_flip(m, S))^2 / S (SNR^2 per unit T_grad)."""
    S = np.asarray(S, dtype=float)
    p = p_flip(float(margin), S.astype(int) if S.ndim else int(S))
    return (1.0 - 2.0 * p) ** 2 / S


def optimal_probe_split(T_grad: int, margin: float, *, min_probes: int = 8,
                        max_probes: int = 256, shots_grid=None) -> tuple:
    """Choose (B, S_probe) maximising the gradient SNR at fixed budget T_grad.

    ``max_probes`` bounds ``B`` (equivalently floors ``S_probe`` at
    ``T_grad / max_probes``).  This is a *practical* constraint, not a statistical
    one: the unconstrained objective is flat as ``S -> 1`` (see module docstring), so
    the optimiser would otherwise request tens of thousands of single-shot probes per
    iteration.  Each probe is still a circuit evaluation, and a real deployed API also
    rate-limits queries, so B is capped.  The RQ2 sweep reports the *unconstrained*
    curve so the plateau/boundary optimum is visible regardless of this cap.

    Returns ``(B, S_probe, info)`` with the swept grid and curve for plotting/auditing.
    """
    T_grad = int(max(T_grad, min_probes))
    s_hi = max(1, T_grad // min_probes)                    # B >= min_probes
    s_lo = max(1, T_grad // max(1, max_probes))            # B <= max_probes
    if shots_grid is None:
        shots_grid = np.unique(np.round(np.geomspace(1, max(2, s_hi), 40)).astype(int))
    shots_grid = np.asarray(shots_grid, dtype=int)
    g = np.array([grad_snr_per_budget(margin, int(S)) for S in shots_grid])
    feasible = (shots_grid <= s_hi) & (shots_grid >= s_lo)
    g_feas = np.where(feasible, g, -np.inf)
    j = int(np.argmax(g_feas))
    S_probe = int(shots_grid[j])
    B = int(np.clip(T_grad // S_probe, min_probes, max_probes))
    info = {"shots_grid": shots_grid.tolist(),
            "snr": [float(v) for v in g],                  # unconstrained curve
            "feasible": [bool(v) for v in feasible],
            "margin": float(margin), "T_grad": int(T_grad)}
    return B, S_probe, info


def predicted_optimal_shots(T_grad: int, margin: float, **kw) -> int:
    """The theory-predicted S* (used to overlay on the empirical RQ2 curve)."""
    return optimal_probe_split(T_grad, margin, **kw)[1]
