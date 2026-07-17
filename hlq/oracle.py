"""The stochastic label oracle -- the quantum-specific crux (plan Sec. 4.2).

A quantum classifier queried with a finite shot budget ``S`` is a *natively
stochastic* oracle.  For a diagonal observable ``M`` with eigenvalues in {-1,+1},
each shot is an independent Bernoulli with ``p(+1) = (1 + f)/2`` where
``f = <M>`` is the exact decision value.  Hence the S-shot empirical mean and its
sign are governed *exactly* by a Binomial(S, (1+f)/2) draw -- this reproduces real
shot-based measurement without materialising S samples, and is what every attack
queries.

Two models live here, and the paper's whole premise is the gap between them:

* ``sample_label`` -- the EXACT Binomial oracle (ground truth the attacks face);
* ``p_flip`` -- the closed-form CLT/Gaussian model the *calibrated* attacker uses
  to choose its shot budget ``S(delta, m_hat)``.

``validate_p_flip`` performs the double-step check (T1): closed-form vs an
independent Monte-Carlo estimate of the flip rate, plus the three analytic limits.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import norm


# --------------------------------------------------------------------------- #
# Closed-form Born-rule flip model (the attacker's calibration knowledge)
# --------------------------------------------------------------------------- #
def p_flip(f: np.ndarray, S: int) -> np.ndarray:
    """P(sign(f_hat) != sign(f)) under the CLT model:  Phi( -|f| sqrt(S/(1-f^2)) ).

    Analytic limits (VERIFY): S->inf => 0 ; |f|->0 => 1/2 ; monotone down in S,|f|.
    """
    f = np.asarray(f, dtype=np.float64)
    m = np.abs(f)
    var = np.clip(1.0 - f * f, 1e-15, None)          # Var[f_hat] = (1-f^2)/S
    z = m * np.sqrt(S / var)
    return norm.cdf(-z)


def shots_for_delta(f_hat: float, delta: float, s_min: int = 1,
                    s_max: int = 100_000) -> int:
    """Smallest S guaranteeing p_flip <= delta at estimated decision value f_hat.

    Invert the CLT model:  S = ceil( (1 - f_hat^2) * (Phi^{-1}(delta) / |f_hat|)^2 ).
    """
    m = abs(float(f_hat))
    if m < 1e-9:                                     # on the boundary: never reliable
        return s_max
    z = norm.ppf(delta)                              # < 0 for delta < 0.5
    s = int(np.ceil((1.0 - f_hat * f_hat) * (z / m) ** 2))
    return int(np.clip(s, s_min, s_max))


# --------------------------------------------------------------------------- #
# Exact stochastic oracle
# --------------------------------------------------------------------------- #
def sample_label(f: np.ndarray, S: Optional[int], rng: np.random.Generator) -> np.ndarray:
    """Exact S-shot label(s) from decision value(s) f. S=None => deterministic sign.

    k ~ Binomial(S, (1+f)/2);  f_hat = 2k/S - 1;  label = sign(f_hat), ties broken
    by a fair coin (matches an undefined sign on the measured boundary).
    """
    f = np.asarray(f, dtype=np.float64)
    scalar = f.ndim == 0
    f = np.atleast_1d(f)
    if S is None or S == np.inf:
        lab = np.where(f >= 0, 1, -1)
    else:
        p = np.clip((1.0 + f) / 2.0, 0.0, 1.0)
        k = rng.binomial(int(S), p)
        fhat = 2.0 * k / S - 1.0
        lab = np.sign(fhat).astype(np.int64)
        ties = lab == 0
        if ties.any():
            lab[ties] = rng.choice([-1, 1], size=int(ties.sum()))
    lab = lab.astype(np.int64)
    return lab[0] if scalar else lab


class BudgetExhausted(Exception):
    """Raised when a query would exceed the oracle's total measurement budget T."""


class StochasticOracle:
    """Hard-label oracle over a trained classifier, with query/shot accounting.

    The attacks see ONLY ``label`` / ``label_batch`` (top-1 hard labels) and the
    original label; they never receive ``f``.  ``estimate_f`` is available for the
    calibrated attack's shot controller (it costs shots, which are accounted).

    A finite ``budget`` (total shots ``T``) makes every attack stop at *exactly* the
    same measurement budget: a query that would exceed ``T`` raises
    :class:`BudgetExhausted`, which the attack skeleton catches and returns best-so-far.
    This is what makes the fixed-``T`` comparisons (RQ2/RQ3) fair.
    """

    def __init__(self, classifier, rng: np.random.Generator, *, stochastic=False,
                 budget: float = float("inf"), cache_f: bool = True, cache_max=50_000):
        self.clf = classifier
        self.rng = rng
        self.stochastic = stochastic          # True for randomized-encoding defense
        self.budget = budget
        self.n_queries = 0
        self.n_shots = 0
        self.n_circuit_evals = 0              # wall-clock cost (cache misses only)
        # f(x) is a deterministic pure function, so repeated queries at the SAME point
        # (the calibrated attack's shot escalation, PopSkipJump's repeats) need only one
        # simulation.  Shots are still independent Binomial draws from that same f, so
        # the statistics are unchanged -- this is purely a compute optimisation.
        # Disabled for per-query-stochastic defenses, whose f genuinely varies per call.
        self._fcache = {} if (cache_f and not stochastic) else None
        self._cache_max = cache_max

    def reset_counters(self):
        self.n_queries = 0
        self.n_shots = 0
        self.n_circuit_evals = 0

    def _charge(self, cost: int):
        if self.n_shots + cost > self.budget:
            raise BudgetExhausted
        self.n_shots += int(cost)

    def _f_single(self, x) -> float:
        if self._fcache is None:
            self.n_circuit_evals += 1
            return float(self.clf.decision_function(np.atleast_2d(x))[0])
        key = np.asarray(x, dtype=np.float64).tobytes()
        v = self._fcache.get(key)
        if v is None:
            self.n_circuit_evals += 1
            v = float(self.clf.decision_function(np.atleast_2d(x))[0])
            if len(self._fcache) >= self._cache_max:
                self._fcache.clear()
            self._fcache[key] = v
        return v

    # -- exact decision value (no measurement noise; not charged as a query) -- #
    def f_exact(self, X):
        return self.clf.decision_function(X)

    # -- hard-label queries (charged) -------------------------------------- #
    def label(self, x, S: Optional[int]) -> int:
        self._charge(int(S) if S not in (None, np.inf) else 0)
        self.n_queries += 1
        return int(sample_label(self._f_single(x), S, self.rng))

    def label_batch(self, X, S: Optional[int]) -> np.ndarray:
        X = np.atleast_2d(X)
        self._charge(len(X) * (int(S) if S not in (None, np.inf) else 0))
        self.n_queries += len(X)
        self.n_circuit_evals += len(X)                  # batched, but still len(X) evals
        f = self.clf.decision_function(X)
        return sample_label(f, S, self.rng)

    def estimate_f(self, x, S: int) -> float:
        """Empirical f_hat over S shots (single Binomial draw). Charged as 1 query."""
        self._charge(int(S))
        self.n_queries += 1
        p = np.clip((1.0 + self._f_single(x)) / 2.0, 0.0, 1.0)
        k = self.rng.binomial(int(S), p)
        return 2.0 * k / S - 1.0

    def estimate_f_batch(self, X, S: int) -> np.ndarray:
        """Empirical f_hat over S shots for a batch (one circuit call). Charged per point."""
        X = np.atleast_2d(X)
        self._charge(len(X) * int(S))
        self.n_queries += len(X)
        self.n_circuit_evals += len(X)
        f = self.clf.decision_function(X)
        p = np.clip((1.0 + f) / 2.0, 0.0, 1.0)
        k = self.rng.binomial(int(S), p)
        return 2.0 * k / S - 1.0


# --------------------------------------------------------------------------- #
# Double-step validation of the flip model  (sanity test T1)
# --------------------------------------------------------------------------- #
def validate_p_flip(margins=None, shots=None, trials=20000, seed=0) -> dict:
    """Closed-form p_flip vs an independent Monte-Carlo flip rate over an (m,S) grid.

    Returns a JSON-serialisable record with the grid, both surfaces, the max
    absolute discrepancy, and explicit checks of the three analytic limits.
    """
    rng = np.random.default_rng(seed)
    if margins is None:
        margins = np.linspace(0.02, 0.98, 25)
    if shots is None:
        shots = np.array([10, 30, 100, 300, 1000, 3000, 10000])
    margins = np.asarray(margins, dtype=float)
    shots = np.asarray(shots, dtype=int)

    closed = np.zeros((len(margins), len(shots)))
    mc = np.zeros_like(closed)
    for i, m in enumerate(margins):
        f = m                                        # true label +1 (f>0)
        for j, S in enumerate(shots):
            closed[i, j] = float(p_flip(f, int(S)))
            labs = sample_label(np.full(trials, f), int(S), rng)
            mc[i, j] = float(np.mean(labs != 1))     # empirical flip rate vs true +1

    mc_se = np.sqrt(np.clip(mc * (1 - mc), 1e-12, None) / trials)
    within = np.abs(closed - mc) <= (3 * mc_se + 5e-3)

    # analytic limits
    lim_inf = float(p_flip(0.5, 10_000_000))                       # S->inf
    lim_boundary = float(p_flip(1e-6, 1000))                       # m->0
    mono_S = bool(np.all(np.diff([p_flip(0.3, s) for s in [10, 100, 1000, 10000]]) <= 1e-9))
    mono_m = bool(np.all(np.diff([p_flip(mm, 200) for mm in [0.1, 0.3, 0.5, 0.7, 0.9]]) <= 1e-9))

    return {
        "test": "T1_p_flip",
        "margins": margins.tolist(),
        "shots": shots.tolist(),
        "p_flip_closed": closed.tolist(),
        "p_flip_montecarlo": mc.tolist(),
        "max_abs_discrepancy": float(np.abs(closed - mc).max()),
        "fraction_within_3se": float(np.mean(within)),
        "limit_S_to_inf": lim_inf,
        "limit_margin_to_0": lim_boundary,
        "monotone_decreasing_in_S": mono_S,
        "monotone_decreasing_in_margin": mono_m,
        "trials": int(trials),
        "passed": bool(lim_inf < 1e-3 and abs(lim_boundary - 0.5) < 0.05
                       and mono_S and mono_m and np.mean(within) > 0.9),
    }
