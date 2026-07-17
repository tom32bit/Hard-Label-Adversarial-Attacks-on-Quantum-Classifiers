"""hlq -- Hard-Label Quantum Attacks.

A shot-budget-aware, Born-rule-calibrated decision-based (hard-label) adversarial
attack framework for variational quantum classifiers (VQCs), plus baselines,
defenses, concentration diagnostics, and research-grade analysis.

Reference: hard_label_quantum_attacks_research_plan.md (Quantum Machine Intelligence).

Design principles
-----------------
* Double-step validation: every numeric component is checked (1) against a
  closed-form / analytic limit and (2) against an independent Monte-Carlo /
  empirical simulation before it is trusted (see hlq.oracle and experiments T1-T6).
* Single source of truth: shared attack machinery lives in ``hlq.attacks.base``;
  concrete attacks override only *policy* hooks (shot budgeting, probe splitting,
  side-of-boundary decision) -- no duplicated boundary-walk code.
* Analytic Born-rule oracle: the exact decision value ``f = <M>`` is computed once
  per query point; the stochastic S-shot label is an *exact* Binomial draw with
  ``p(+1) = (1 + f) / 2`` (diagonal observable), which reproduces real shot-based
  measurement without materializing S samples.
"""

__version__ = "1.0.0"
