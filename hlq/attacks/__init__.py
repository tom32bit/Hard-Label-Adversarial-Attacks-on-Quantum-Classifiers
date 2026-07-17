"""Decision-based (hard-label) attacks and the white-box / transfer references.

The HSJA family (calibrated, fixed, PopSkipJump, classical anchor) all inherit the
single skeleton in :mod:`hlq.attacks.base`; each overrides only its shot-budget and
side-of-boundary *policy*.  PGD (white-box) and transfer are separate references.
"""
from .base import AttackResult, DecisionBasedAttack, Domain
from .hsja_fixed import FixedShotHSJA
from .hsja_quantum import CalibratedHSJA
from .popskipjump import PopSkipJump
from .pgd_whitebox import pgd_attack
from .transfer import transfer_attack

BUILDERS = {
    "calibrated_hsja": CalibratedHSJA,
    "fixed_hsja": FixedShotHSJA,
    "popskipjump": PopSkipJump,
    "classical_hsja": FixedShotHSJA,   # same skeleton, deterministic oracle (S=None)
}

__all__ = [
    "AttackResult", "DecisionBasedAttack", "Domain",
    "FixedShotHSJA", "CalibratedHSJA", "PopSkipJump",
    "pgd_attack", "transfer_attack", "BUILDERS",
]
