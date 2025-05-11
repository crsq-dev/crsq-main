""" 1D hydrogen atom related model functions
"""
import math
import numpy
import numpy.typing as npyt

import scipy.special as sp

class VHAtom:
    """V(x) for H atom. potential function object"""

    def __init__(self, Q0: float, dq: float, Z: int):
        """
        Args:
            Q0: float : center of the potential
            inv0: float : ゼロ除算を防ぐために 1/0 の代わりに用いる値
            Z: float : charge of the nucleus
        """
        self._Q0 = Q0
        self._dq = dq
        self._Z = Z

    def __call__(self, x: npyt.NDArray[numpy.float64]) -> npyt.NDArray[numpy.float64]:
        # riA は、各格子点での原子核までの距離単位は格子間隔。
        riA = numpy.abs(numpy.subtract(x, self._Q0))
        # ゼロ除算を防ぐために、ゼロになるところは 1/r を inv0 で置き換える。
        xq0 = int(self._Q0 / self._dq)
        riA[xq0] = self._dq / 2
        qe = -1
        QA = self._Z
        varray = (qe*QA) / riA
        return varray

    @property
    def label(self):
        return f"V(q)=-1/|q-({self._Q0})|"

class VHAtomDiscrete:
    """V(x) for H atom. potential function object"""

    def __init__(self, xQ0: int, dq: float, nb: int, Z: float):
        """
        Args:
            Q0: float : center of the potential
            inv0: float : ゼロ除算を防ぐために 1/0 の代わりに用いる値
            Z: float : charge of the nucleus
        """
        self._xQ0 = xQ0
        self._dq = dq
        self._nb = nb
        self._Z = Z

    def __call__(self, x: npyt.NDArray[numpy.float64]) -> npyt.NDArray[numpy.float64]:
        # riA は、各格子点での原子核までの距離単位は格子間隔。
        xq = (x // self._dq).astype(int)
        rqiA = numpy.abs(xq - self._xQ0)
        # ゼロ除算を防ぐために、ゼロになるところは 1/r を inv0 で置き換える。
        rqiA[self._xQ0] = self._dq / 2
        one_nb = 1 << self._nb
        quotient_nb = one_nb // rqiA
        quotient = quotient_nb.astype(float) * 2**(-self._nb)
        qe = -1
        QA = self._Z
        varray = (qe*QA) * quotient
        return varray

    @property
    def label(self):
        return f"V(q)=-1/|q-({self._Q0})|"


class PsiH1D_Loudon:
    """Hydrogen atom 1s wave function, 1 dimensional version"""

    def __init__(self, Q0: float, N: int, odd: bool):
        self._Q0 = Q0
        self._N = N
        self._odd = odd
        if N <= 0:
            raise ValueError(f"N must be positive integer, but {N} is given.")

    def __call__(self, qv: npyt.NDArray[numpy.float64]) -> npyt.NDArray[numpy.float64]:
        N = self._N  # primary quantum number
        hbar = 1
        me = 1
        qe = 1
        a0 = 1
        a0 = hbar * hbar / (me * qe * qe)
        x = qv - self._Q0
        absx = numpy.abs(x)
        A = numpy.sqrt(2 / ((a0**3) * (N**5) * math.factorial(N) ** 2))
        lg = sp.assoc_laguerre(2 * absx / (N * a0), N - 1, 1)
        if self._odd:
            psi = A * numpy.exp(-absx / (N * a0)) * x * lg
        else:
            psi = A * numpy.exp(-absx / (N * a0)) * absx * lg
        return psi

    @property
    def parity(self):
        return "odd" if self._odd else "even"

    @property
    def label(self):
        return f"L:{self.parity}:ψ(z)=An*z*exp(-|z|/2)*L{self._N}^1(z)"

    @property
    def name(self):
        return f"H1D_L_{self._N}_{self.parity}_Q0_{self._Q0}"

    @property
    def eigen_value(self):
        return -1 / 2 * self._N**2

class PsiH1D_Palma:
    """Hydrogen atom 1s wave function, 1 dimensional version"""

    def __init__(self, Q0: float, N: int):
        self._Q0 = Q0
        self._N = N
        if N <= 0:
            raise ValueError(f"N must be a non-negative integer, but {N} is given.")

    def __call__(self, qv: numpy.ndarray) -> numpy.ndarray:
        N = self._N  # primary quantum number
        hbar = 1
        me = 1
        qe = 1
        a0 = 1
        a0 = hbar**2 / (me * qe**2)
        z = (2 / N / a0) * (qv - self._Q0)
        absz = numpy.abs(z)
        An = (
            (-1) ** N
            * math.sqrt(2 * N * a0)
            / (math.factorial(N) * (N**3) * (a0**2))
            * (N * a0)
            / 2
        )
        np_lg = sp.assoc_laguerre(absz, N - 1, 1)
        lg = numpy.array(np_lg)
        psi = An * z * numpy.exp(-absz / 2) * lg
        return psi

    @property
    def label(self):
        return f"P:ψ(z)=An*z*exp(-|z|/2)*L{self._N}^1(z)"

    @property
    def name(self):
        return f"H1D_P_{self._N}_Q0_{self._Q0}"

    @property
    def eigen_value(self):
        return -1 / 2 * self._N**2