"""
    Hydrogen atom model in 2D.
"""

import math
import cupy as np
import cupy.typing as npt
import scipy.special as sp
import logging

logger = logging.getLogger(__name__)

class PsiH2D:
    """ Hydrogen atom, 2 dimensional model

        A callable object to calculate the wave function of the hydrogen atom in 2D model.

        Args:
            (Qx0, Qy0): float : center of the potential
            delta_q: offset added to avoid division by zero
            n: int : primary quantum number
            m: int : secondary quantum number
    """
    def __init__(self, Qx0: float, Qy0: float, dq: float, n: int, m: int):
        self._Qx0 = Qx0
        self._Qy0 = Qy0
        self._dq = dq
        if abs(m) > n:
            raise ValueError(f"Invalid quantum numbers: n={n}, m={m}. must be |m| <= n")
        self._n = n
        self._m = m
        logger.info("PsiH2D.__init__:   n = %d, m = %d", n, m)

    def __call__(self, qxv: np.ndarray, qyv: np.ndarray) -> np.ndarray:
        """ calculate the wave function of the hydrogen atom in 2D model.

        Args:
            qxv: np.ndarray[(M,M)] : x coordinate values
            qyv: np.ndarray[(M,M)] : y coordinate values
        Returns:
            psi: np.ndarray[(M,M)] : wave function values
        """
        logger.info("PsiH2D.__call__")
        n = self._n
        m = self._m
        absm = abs(m)
        q0 = 1/(n+1/2)
        dxv = qxv - self._Qx0
        dyv = qyv - self._Qy0
        rho = np.sqrt(np.square(dxv) + np.square(dyv))
        rho[self._Qx0//self._dq, self._Qy0//self._dq] = self._dq / 2
        A = math.sqrt((q0**3 * math.factorial(n-absm))/(math.pi*math.factorial(n+absm)))
        logger.info("PsiH2D:   A = ", A)
        q0rho = q0*rho
        q0rho2 = 2*q0rho

        np_q0rho2 = np.asnumpy(q0rho2)
        np_lg = sp.assoc_laguerre(np_q0rho2, n-absm, 2*absm)
        logger.info("PsiH2d:   np_lg = ", np_lg)
        lg = np.array(np_lg)

        psi = (A * np.power(q0rho2, absm) * np.exp(-q0rho) * lg * np.power(((dxv+1j*dyv)/rho),m))
        return psi

    @property
    def label(self):
        return f'H-2D:ψ{self._n}_{self._m}(q)'

    @property
    def name(self):
        return f'H2D_n_{self._n}_m_{self._m}_q0_{self._Qx0}_{self._Qy0}'
    


class VHAtom2:
    """V(x) for H atom. potential function object - 2D version"""

    def __init__(self, Qx0: float, Qy0: float, dq: float, Z: float):
        self._Qx0 = Qx0
        self._Qy0 = Qy0
        self._dq = dq
        self._Z = Z

    def __call__(self, qxv: np.ndarray, qyv: np.ndarray) -> np.ndarray:
        riA = np.sqrt(
            (
                np.square(qxv - self._Qx0)
                + np.square(qyv - self._Qy0)
            )
        )
        xq0 = int(self._Qx0 / self._dq)
        yq0 = int(self._Qy0 / self._dq)
        riA[xq0, yq0] = self._dq / 2
        qe = -1
        QA = 1
        varray = (qe * QA) / riA
        return varray

    @property
    def label(self):
        return f"V(q)=-1/sqrt((q-({self._Qx0},{self._Qy0}))^2+{self._delta_qQ}^2)"

