"""Shared decision-based attack skeleton (HopSkipJump geometry, plan Sec. 4.3).

All hard-label variants inherit :class:`DecisionBasedAttack`.  The skeleton -- init
-> boundary binary-search -> Monte-Carlo normal estimate -> geometric step -> repeat
-- lives here exactly once.  Concrete attacks override only *policy* hooks:

* ``_decide_adversarial(x, phase)`` : the side-of-boundary decision and how many
  shots / repeats it costs  (this is where the stochastic oracle bites -- M1);
* ``_grad_budget(t)``               : the (B, S_probe) split of the per-iteration
  gradient budget  (M2).

Every side-of-boundary decision goes through ``_decide_adversarial`` so the
stochastic-oracle handling is defined in one place per attack, never duplicated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..config import AttackConfig
from ..oracle import BudgetExhausted, StochasticOracle


@dataclass
class Domain:
    """Valid region for perturbed inputs (the encoder's input space)."""

    kind: str = "box"           # "box" (clip to [lo,hi]) | "sphere" (L2-normalise) | "none"
    lo: float = 0.0
    hi: float = float(np.pi)

    def project(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "box":
            return np.clip(X, self.lo, self.hi)
        if self.kind == "sphere":
            n = np.linalg.norm(X, axis=-1, keepdims=True)
            n = np.where(n == 0, 1.0, n)
            return X / n
        return X


@dataclass
class AttackResult:
    success: bool
    perturbation: float                       # final L2 (np.inf if never adversarial)
    x_adv: np.ndarray
    queries: int
    shots: int
    dist_trajectory: list = field(default_factory=list)
    query_trajectory: list = field(default_factory=list)
    shot_trajectory: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        r = dict(success=bool(self.success), perturbation=float(self.perturbation),
                 queries=int(self.queries), shots=int(self.shots),
                 dist_trajectory=[float(d) for d in self.dist_trajectory],
                 query_trajectory=[int(q) for q in self.query_trajectory],
                 shot_trajectory=[int(s) for s in self.shot_trajectory])
        r.update(self.meta)
        return r


class DecisionBasedAttack:
    """HopSkipJump-style ell_2 untargeted attack; policy hooks left abstract."""

    def __init__(self, cfg: AttackConfig):
        self.cfg = cfg

    # --------------------------------------------------------------------- #
    # Policy hooks (overridden by subclasses)
    # --------------------------------------------------------------------- #
    def _decide_adversarial(self, x: np.ndarray, phase: str) -> bool:
        raise NotImplementedError

    def _grad_budget(self, t: int, x_b: np.ndarray, x0: np.ndarray) -> tuple:
        """Return (B, S_probe) for the normal estimate at iteration t.

        ``x_b`` (current boundary point) and ``x0`` are passed so a calibrated
        policy can estimate the local probe margin; fixed policies ignore them.
        """
        return self.cfg.num_probes, self.cfg.probe_shots

    # convenience shared by fixed/popskipjump policies
    def _labels_batch(self, X: np.ndarray, S: Optional[int]) -> np.ndarray:
        return self._oracle.label_batch(X, S)

    # --------------------------------------------------------------------- #
    # Shared geometry
    # --------------------------------------------------------------------- #
    def _binary_search(self, x_adv: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """Project to the boundary by bisecting the segment [x0 (safe), x_adv (adv)].

        Invariant: ``hi`` is always an accepted-adversarial alpha, ``lo`` is not, so the
        returned point is on the adversarial side.  A policy may set ``_halt_bisect``
        to stop early once the oracle can no longer resolve the side of the boundary
        at the affordable shot budget (the calibrated attack's resolution limit, M1).
        """
        lo, hi = 0.0, 1.0
        tol = self.cfg.bin_search_tol
        self._halt_bisect = False
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            xm = self._domain.project(x0 + mid * (x_adv - x0))
            if self._decide_adversarial(xm, "bsearch"):
                hi = mid
            else:
                lo = mid
            if self._halt_bisect:            # resolution limit reached; hi stays valid
                break
        return self._domain.project(x0 + hi * (x_adv - x0))

    def _estimate_gradient(self, x_b, x0, t, rng) -> np.ndarray:
        """HSJA Monte-Carlo boundary-normal estimate with batched probes (M2)."""
        d = x_b.size
        B, S_probe = self._grad_budget(t, x_b, x0)
        dist = float(np.linalg.norm(x_b - x0))
        delta = max(dist / d, 10 * self.cfg.bin_search_tol)      # HSJA ell_2 probe radius
        u = rng.normal(size=(B, d))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        X = self._domain.project(x_b[None, :] + delta * u)
        labels = self._labels_batch(X, S_probe)
        phi = np.where(labels != self._y0, 1.0, -1.0)            # +1 if adversarial
        phi = phi - phi.mean()                                   # baseline control variate
        g = (phi[:, None] * u).mean(axis=0)
        ng = np.linalg.norm(g)
        if ng < 1e-12:                                           # degenerate: random dir
            g = rng.normal(size=d)
            ng = np.linalg.norm(g)
        return g / ng

    def _geometric_step(self, x_b, x0, grad, t) -> np.ndarray:
        """Move along the estimated normal, halving until still adversarial."""
        dist = float(np.linalg.norm(x_b - x0))
        eps = dist / np.sqrt(t + 1.0)
        for _ in range(25):
            cand = self._domain.project(x_b + eps * grad)
            if self._decide_adversarial(cand, "step"):
                return cand
            eps *= 0.5
            if eps < self.cfg.bin_search_tol:
                break
        return x_b                                               # no improving step found

    def _init_adv(self, x0, y0, init_pool, rng) -> Optional[np.ndarray]:
        """Find a starting adversarial point: nearest opposite-class ref, else random."""
        if init_pool is not None and len(init_pool):
            order = np.argsort(np.linalg.norm(init_pool - x0[None, :], axis=1))
            for idx in order[: min(len(order), 40)]:
                if self._decide_adversarial(init_pool[idx], "init"):
                    return init_pool[idx].copy()
        for _ in range(200):                                     # random fallback
            if self._domain.kind == "sphere":
                c = rng.normal(size=x0.size)
            else:
                c = rng.uniform(self._domain.lo, self._domain.hi, size=x0.size)
            c = self._domain.project(c)
            if self._decide_adversarial(c, "init"):
                return c
        return None

    # --------------------------------------------------------------------- #
    # Main loop
    # --------------------------------------------------------------------- #
    def attack(self, oracle: StochasticOracle, x0, y0, init_pool, rng,
               domain: Domain) -> AttackResult:
        self._oracle = oracle
        self._y0 = int(y0)
        self._domain = domain
        oracle.budget = min(oracle.budget, self.cfg.total_budget)   # exact fixed-T stop
        x0 = np.asarray(x0, dtype=np.float64)

        current = x0.copy()
        cur_dist = np.inf
        best_dist = np.inf                    # recorded for reference only (see note)
        dist_traj, q_traj, s_traj = [], [], []
        interrupted = False
        try:
            # T4 trivial-case guard: if x0 is already adversarial there is nothing to do.
            if self._decide_adversarial(x0, "init"):
                return AttackResult(True, 0.0, x0.copy(), oracle.n_queries,
                                    oracle.n_shots, dist_trajectory=[0.0],
                                    query_trajectory=[oracle.n_queries],
                                    shot_trajectory=[oracle.n_shots],
                                    meta={"already_adversarial": True})
            x_adv = self._init_adv(x0, y0, init_pool, rng)
            if x_adv is None:
                return AttackResult(False, np.inf, x0.copy(), oracle.n_queries,
                                    oracle.n_shots, meta={"init_failed": True})
            x_b = self._binary_search(x_adv, x0)
            current, cur_dist = x_b.copy(), float(np.linalg.norm(x_b - x0))
            best_dist = cur_dist
            dist_traj, q_traj, s_traj = [cur_dist], [oracle.n_queries], [oracle.n_shots]

            for t in range(self.cfg.iterations):
                grad = self._estimate_gradient(x_b, x0, t, rng)
                x_step = self._geometric_step(x_b, x0, grad, t)
                x_b = self._binary_search(x_step, x0)
                current = x_b.copy()
                cur_dist = float(np.linalg.norm(x_b - x0))
                best_dist = min(best_dist, cur_dist)
                dist_traj.append(cur_dist)
                q_traj.append(oracle.n_queries)
                s_traj.append(oracle.n_shots)
        except BudgetExhausted:
            interrupted = True

        # NOTE: we return the FINAL iterate, as standard HopSkipJump does -- never the
        # minimum-*believed*-distance iterate.  Under a stochastic oracle each accepted
        # boundary point carries a ~delta chance of being a false positive, and a false
        # positive always lies much closer to x0; taking the minimum would therefore
        # systematically select the attack's own errors and return a point that is not
        # truly adversarial.
        return AttackResult(
            success=np.isfinite(cur_dist), perturbation=cur_dist, x_adv=current,
            queries=oracle.n_queries, shots=oracle.n_shots,
            dist_trajectory=dist_traj, query_trajectory=q_traj, shot_trajectory=s_traj,
            meta={"budget_interrupted": interrupted, "best_believed_distance": best_dist},
        )
