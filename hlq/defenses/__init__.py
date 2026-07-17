"""Inference-time defenses, re-evaluated for the FIRST time against a gradient-free
hard-label attacker (plan Sec. 4.5, RQ4).

Both defenses wrap a trained :class:`hlq.classifier.VQC` and expose the same
``decision_function`` interface, so the oracle and every attack run on them unchanged.
"""
from ..config import DefenseConfig
from .noise import NoisyVQC
from .randomized_encoding import RandomizedEncodingVQC


def wrap_defense(vqc, defense: DefenseConfig):
    """Return (defended_classifier, is_stochastic) for the requested defense."""
    if defense.name == "none":
        return vqc, False
    if defense.name == "depolarizing":
        return NoisyVQC(vqc, defense), False          # deterministic f given x
    if defense.name == "randomized_encoding":
        return RandomizedEncodingVQC(vqc, defense), bool(defense.randomized_per_query)
    raise ValueError(defense.name)


__all__ = ["NoisyVQC", "RandomizedEncodingVQC", "wrap_defense"]
