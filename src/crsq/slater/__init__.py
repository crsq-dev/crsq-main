""" This package provides functions to construct a qubit representation
    of the Slater determinant.
"""

from crsq.slater.aregister import ARegister, ARegisterFrame
from crsq.slater.bregister import BRegister, BRegisterFrame
from crsq.slater.sigma import build_sums

__all__ = [
    "ARegister",
    "BRegister",
    "ARegisterFrame",
    "BRegisterFrame",
    "build_sums"
]