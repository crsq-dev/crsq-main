"""
    Hydrogen atom model in 2D.
"""

import math
import numpy
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

    def __call__(self, qxv: numpy.ndarray, qyv: numpy.ndarray) -> numpy.ndarray:
        """ calculate the wave function of the hydrogen atom in 2D model.

        Args:
            qxv: numpy.ndarray[(M,M)] : x coordinate values
            qyv: numpy.ndarray[(M,M)] : y coordinate values
        Returns:
            psi: numpy.ndarray[(M,M)] : wave function values
        """
        logger.info("PsiH2D.__call__")
        n = self._n
        m = self._m
        absm = abs(m)
        q0 = 1/(n+1/2)
        dxv = qxv - self._Qx0
        dyv = qyv - self._Qy0
        rho = numpy.sqrt(numpy.square(dxv) + numpy.square(dyv))
        x0 = int(self._Qx0 // self._dq)
        y0 = int(self._Qy0 // self._dq)
        A = math.sqrt((q0**3 * math.factorial(n-absm))/(math.pi*math.factorial(n+absm)))
        q0rho = q0*rho
        q0rho2 = 2*q0rho

        np_lg = sp.assoc_laguerre(q0rho2, n-absm, 2*absm)
        lg = numpy.array(np_lg)

        rho[x0, y0] = 1
        omega = (dxv+1j*dyv)/rho
        # suppress division by zero
        omega[x0, y0] = 1
        rho[x0, y0] = 0

        psi = (A * numpy.power(q0rho2, absm) * numpy.exp(-q0rho) * lg * numpy.power(omega, m))
        return psi

    @property
    def label(self):
        return f'H-2D:ψ{self._n}_{self._m}(q)'

    @property
    def name(self):
        return f'H2D_n_{self._n}_m_{self._m}_q0_{self._Qx0}_{self._Qy0}'

    @property
    def eigen_value(self):
        # Parfitt uses Rydberg energy units, which are double the Bohr energy units
        return -1/(2*(self._n+1/2)**2)
    
    @property
    def n(self):
        return self._n
    
    @property
    def m(self):
        return self._m
    
    @property
    def r0(self):
        if self._m > 0:
            return self._dq / 2
        else:
            q0 = 1/(self._n+1/2)
            dq = self._dq
            if self._n == 0:
                return dq*dq*q0/4/(-math.exp(-q0*dq)+1)
            else:
                return dq*dq*q0/4/((-1+q0*q0*dq*dq)*math.exp(-q0*dq)+1)

class VHAtom2:
    """V(x) for H atom. potential function object - 2D version"""

    def __init__(self, Qx0: float, Qy0: float, dq: float, r0: float, Z: float):
        self._Qx0 = Qx0
        self._Qy0 = Qy0
        self._dq = dq
        self._r0 = r0
        self._Z = Z
        logger.info("VHAtom2.__init__:   dq = %f, r0=%f  r0/dq=%f", dq, r0, r0/dq)


    def __call__(self, x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
        logger.info("VHAtom2.__call__")
        riA = numpy.sqrt(
            (
                numpy.square(x - self._Qx0)
                + numpy.square(y - self._Qy0)
            )
        )
        xq0 = int(self._Qx0 / self._dq)
        yq0 = int(self._Qy0 / self._dq)
        riA[xq0, yq0] = self._r0
        qe = -1
        QA = self._Z
        varray = (qe * QA) / riA
        return varray

    @property
    def label(self):
        return f"V(q)=-1/sqrt((q-({self._Qx0},{self._Qy0}))^2+{self._delta_qQ}^2)"

