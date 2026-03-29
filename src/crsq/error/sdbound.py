"""State dependent error bound"""

import math
import cmath
import numpy
import numpy.typing as npt

import logging

logger = logging.getLogger("crsq.error")


class RunData:
    def __init__(self):
        """Error data holder class"""

    def set_psi0_rs(self, psi0_rs):
        """Set the initial state in real space"""
        self._psi0_rs = psi0_rs

    @property
    def psi0_rs(self) -> npt.NDArray[numpy.complex128]:
        return self._psi0_rs

    def set_psi0_ks(self, psi0_ks):
        """Set the initial state in k-space"""
        self._psi0_ks = psi0_ks

    @property
    def psi0_ks(self) -> npt.NDArray[numpy.complex128]:
        return self._psi0_ks

    def set_psin_rs(self, psin_rs):
        """Set the state after time evolution in real space"""
        self._psin_rs = psin_rs

    @property
    def psin_rs(self) -> npt.NDArray[numpy.complex128]:
        return self._psin_rs

    def set_psin_ks(self, psin_ks):
        """Set the state after time evolution in k-space"""
        self._psin_ks = psin_ks

    @property
    def psin_ks(self) -> npt.NDArray[numpy.complex128]:
        return self._psin_ks

    def set_dh1_rs(self, dh1_rs):
        """Set the diagonal elements of the hamiltonian in real space"""
        self._dh1_rs = dh1_rs

    @property
    def dh1_rs(self) -> npt.NDArray[numpy.float64]:
        return self._dh1_rs

    def set_dh2_ks(self, dh2_ks):
        """Set the diagonal elements of the hamiltonian in k-space"""
        self._dh2_ks = dh2_ks

    @property
    def dh2_ks(self) -> npt.NDArray[numpy.float64]:
        return self._dh2_ks

    def set_h(self, h: float):
        """Set the energy levelh for the state psi"""
        self._h = h

    @property
    def h(self) -> float:
        return self._h

    def set_N(self, N: int):
        self._N = N

    @property
    def N(self) -> float:
        return self._N

    def set_t(self, t):
        self._t = t

    @property
    def t(self) -> float:
        return self._t


def sdbound1(rd: RunData, alpha: float) -> float:
    """Calculate the state-dependent error bound of first order trotterization
    for a state in real space (psi_rs) and k-space (psi_ks)
    and the diagonal elements of the hamiltonian H1 as DH1_rs and H2 as DH2_ks.
    psi is the eigen function for H1 + H2 with eigen value h, i.e. (H1+H2)psi = h*psi.
    alpha is an arbitrary real number. t is the total time for time evolution and N is the trotter round.
    The bound is defined as t**2/N * (norm((H1-alpha*h)**2 * psi) + norm((H2-(1-alpha)*h)**2 * psi)

    The caller is supposed to search for an alpha that provides the infimum.
    """
    h1sq = numpy.square(rd.dh1_rs - alpha * rd.h)
    h2sq = numpy.square(rd.dh2_ks - (1 - alpha) * rd.h)
    norm1 = numpy.linalg.norm(h1sq * rd.psi0_rs)
    norm2 = numpy.linalg.norm(h2sq * rd.psi0_ks)
    bound = rd.t**2 / rd.N * (norm1 + norm2)
    # logger.info(f"alpha: {alpha}, norm1: {norm1}, norm2: {norm2}, bound: {bound}")
    return bound


def sdbound2a(rd: RunData, alpha: float) -> float:
    """Calculate the state-dependent error bound of second order trotterization
    for a state in real space (psi_rs) and k-space (psi_ks)
    and the diagonal elements of the hamiltonian H1 as DH1_rs and H2 as DH2_ks.
    psi is the eigen function for H1 + H2 with eigen value h, i.e. (H1+H2)psi = h*psi.
    alpha is an arbitrary real number. t is the total time for time evolution and N is the trotter round.
    The bound is defined as t**2/N * (norm((H1-alpha*h)**2 * psi) + norm((H2-(1-alpha)*h)**2 * psi)

    The caller is supposed to search for an alpha that provides the infimum.
    """
    h1g = rd.dh1_rs - alpha * rd.h
    h2g = rd.dh2_ks - (1 - alpha) * rd.h
    norm1 = numpy.linalg.norm(h1g * h1g * h1g * rd.psi0_rs)
    norm2 = numpy.linalg.norm(h2g * h1g * h1g * rd.psi0_ks)
    norm3 = numpy.linalg.norm(h2g * h2g * h2g * rd.psi0_ks)
    bound = (
        rd.t**3 / rd.N**2 * ((1 / 24.0) * norm1 + (1 / 8.0) * norm2 + (1 / 12) * norm3)
    )
    # logger.info(f"alpha: {alpha}, norm1: {norm1}, norm2: {norm2}, bound: {bound}")
    return bound


def sderror(rd: RunData) -> float:
    """calculate the state-dependent error for a state in real space (psi_s_rs)
    by comparing it to the analytical solution (psi_a_rs)
    "so" is the split operator result. "an" is the analytical result.
    """
    psi_an_rs = rd.psi0_rs * numpy.exp(-1j * rd.h * rd.t)
    logger.info(f"at h={rd.h} t={rd.t} : psi_an_rs[0,0]: {psi_an_rs[0,0]:.6f}, psin_rs[0,0]: {rd.psin_rs[0,0]:.6f} ")
    return numpy.linalg.norm(rd.psin_rs - psi_an_rs)
