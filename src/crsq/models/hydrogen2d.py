"""
Hydrogen atom model in 2D.
"""

import math
import numpy
import scipy.special as sp
import logging

logger = logging.getLogger(__name__)


def SA_0_0(dq, q0):
    """Hep0 function for 2D delta-q analysis."""
    return -(q0**2) * (-math.exp(-2 * q0 * dq) + 1)


def SB_0_0(dq, q0, N):
    h = dq / N
    S = 0
    dq2 = dq * dq
    A = q0**3 / math.pi * h**2
    for i in range(-N, N):
        x = h * i
        for j in range(-N, N):
            y = h * j
            if x * x + y * y > dq2:
                r = math.sqrt(x * x + y * y)
                S += -(A / r) * math.exp(-2 * q0 * r)
    return S


def Hep0_0_0(dq, q0, N):
    SA = SA_0_0(dq, q0)
    SB = SB_0_0(dq, q0, N)
    rt2 = math.sqrt(2)
    Hep0 = (
        math.pi / (q0**3 * dq**2) * (SA + SB)
        + 1 / (rt2 * dq) * math.exp(-2 * rt2 * q0 * dq)
        + 2 / dq * math.exp(-2 * q0 * dq)
    )
    return Hep0


def SA_1_0(dq, q0):
    """Hep0 function for 2D delta-q analysis."""
    return -(q0**2) * (-(1 - 4 * q0**2 * dq**2) * math.exp(-2 * q0 * dq) + 1)


def SB_1_0(dq, q0, N):
    h = dq / N
    S = 0
    dq2 = dq * dq
    A = q0**3 / math.pi * h**2
    for i in range(-N, N):
        x = h * i
        for j in range(-N, N):
            y = h * j
            if x * x + y * y > dq2:
                r = math.sqrt(x * x + y * y)
                S += -(A / r) * (1 - 2 * q0 * r) ** 2 * math.exp(-2 * q0 * r)
    return S


def Hep0_1_0(dq, q0, N):
    SA = SA_1_0(dq, q0)
    SB = SB_1_0(dq, q0, N)
    rt2 = math.sqrt(2)
    Hep0 = (
        math.pi / (q0**3 * dq**2) * (SA + SB)
        + 1 / (rt2 * dq) * (1 - (4 / 3) * rt2 * dq) ** 2 * math.exp(-2 * rt2 * q0 * dq)
        + 2 / dq * (1 - (4 / 3) * dq) ** 2 * math.exp(-2 * q0 * dq)
    )
    return Hep0


class PsiH2D:
    """Hydrogen atom, 2 dimensional model

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
        """calculate the wave function of the hydrogen atom in 2D model.

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
        q0 = 1 / (n + 1 / 2)
        dxv = qxv - self._Qx0
        dyv = qyv - self._Qy0
        rho = numpy.sqrt(numpy.square(dxv) + numpy.square(dyv))
        x0 = int(self._Qx0 // self._dq)
        y0 = int(self._Qy0 // self._dq)
        A = math.sqrt(q0**3 * math.factorial(n - absm)) / (
            math.pi * math.factorial(n + absm)
        )
        q0rho = q0 * rho
        q0rho2 = 2 * q0rho

        np_lg = sp.assoc_laguerre(q0rho2, n - absm, 2 * absm)
        lg = numpy.array(np_lg)

        rho[x0, y0] = 1
        omega = (dxv + 1j * dyv) / rho
        # suppress division by zero
        omega[x0, y0] = 1
        rho[x0, y0] = 0

        psi = (
            A
            * numpy.power(q0rho2, absm)
            * numpy.exp(-q0rho)
            * lg
            * numpy.power(omega, m)
        )
        return psi

    @property
    def label(self):
        return f"H-2D:ψ{self._n}_{self._m}(q)"

    @property
    def name(self):
        return f"H2D_n_{self._n}_m_{self._m}_q0_{self._Qx0}_{self._Qy0}"

    @property
    def eigen_value(self):
        # Parfitt uses Rydberg energy units, which are double the Bohr energy units
        return -1 / (2 * (self._n + 1 / 2) ** 2)

    @property
    def n(self):
        return self._n

    @property
    def m(self):
        return self._m

    @property
    def r0(self):
        """dq/4に至る式の一時近似をしない式。"""
        if self._m > 0:
            return self._dq / 2
        else:
            q0 = 1 / (self._n + 1 / 2)
            dq = self._dq
            if self._n == 0:
                return dq * dq * q0 / 4 / (-math.exp(-q0 * dq) + 1)
            else:
                return (
                    dq
                    * dq
                    * q0
                    / 4
                    / ((-1 + q0 * q0 * dq * dq) * math.exp(-q0 * dq) + 1)
                )

    @property
    def r0_new(self):
        """dq/4 に至る式の一時近似をしないで、さらに四角形の四隅の面積も数値計算した式"""
        n = self._n
        m = self._m
        q0 = 1 / (n + 1 / 2)
        dq = self._dq
        if self._m != 0:
            return dq / 4
        if n == 0:
            Hep0 = Hep0_0_0(dq, q0, 40)
            r0 = -1 / Hep0
            return r0
        elif n == 1:
            Hep0 = Hep0_1_0(dq, q0, 40)
            r0 = -1 / Hep0
            return r0
        else:
            raise ValueError(
                f"Invalid quantum number n={n}. r0_new is not defined for n > 1"
            )

    def r0_for_pole(self, pole_mitigation: str, eps: float = 0.25):
        """Calculate the r0 value for the pole mitigation method."""
        if pole_mitigation == "r0lim":
            return self._dq * eps
        elif pole_mitigation == "r0":
            return self.r0
        elif pole_mitigation == "r0new":
            return self.r0_new
        else:
            raise ValueError(f"Unknown pole mitigation method: {pole_mitigation}")


class PsiH2DRadial:
    """Hydrogen atom, 2 dimensional model, radial function

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
        logger.info("PsiH2DRadial.__init__:   n = %d, m = %d", n, m)

    def __call__(self, qxv: numpy.ndarray, qyv: numpy.ndarray) -> numpy.ndarray:
        """calculate the wave function of the hydrogen atom in 2D model.

        Args:
            qxv: numpy.ndarray[(M,M)] : x coordinate values
            qyv: numpy.ndarray[(M,M)] : y coordinate values
        Returns:
            psi: numpy.ndarray[(M,M)] : wave function values
        """
        logger.info("PsiH2DRadial.__call__")
        n = self._n
        m = self._m
        absm = abs(m)
        q0 = 1 / (n + 1 / 2)
        dxv = qxv - self._Qx0
        dyv = qyv - self._Qy0
        rho = numpy.sqrt(numpy.square(dxv) + numpy.square(dyv))
        x0 = int(self._Qx0 // self._dq)
        y0 = int(self._Qy0 // self._dq)
        A = math.sqrt(
            (q0**3 * math.factorial(n - absm)) / (math.pi * math.factorial(n + absm))
        )
        q0rho = q0 * rho
        q0rho2 = 2 * q0rho

        np_lg = sp.assoc_laguerre(q0rho2, n - absm, 2 * absm)
        lg = numpy.array(np_lg)

        rho[x0, y0] = 1

        psi = A * numpy.power(q0rho2, absm) * numpy.exp(-q0rho) * lg
        return psi

    @property
    def label(self):
        return f"H-2D:ψ{self._n}_{self._m}(q)"

    @property
    def name(self):
        return f"H2D_n_{self._n}_m_{self._m}_q0_{self._Qx0}_{self._Qy0}"

    @property
    def eigen_value(self):
        # Parfitt uses Rydberg energy units, which are double the Bohr energy units
        return -1 / (2 * (self._n + 1 / 2) ** 2)

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
            q0 = 1 / (self._n + 1 / 2)
            dq = self._dq
            if self._n == 0:
                return dq * dq * q0 / 4 / (-math.exp(-q0 * dq) + 1)
            else:
                return (
                    dq
                    * dq
                    * q0
                    / 4
                    / ((-1 + q0 * q0 * dq * dq) * math.exp(-q0 * dq) + 1)
                )


class VHAtom2:
    """V(x) for H atom. potential function object - 2D numpy version"""

    def __init__(
        self,
        Qx0: float,
        Qy0: float,
        dq: float,
        r0: float,
        Z: float,
        eps=0,
        frac_bits: int = -1,
    ):
        """
        arguments:
            Qx0, Qy0: float : center of the potential
            dq: float : offset added to avoid division by zero
            r0: float : Δ1 for Ha1 potential function (Ha1(r)=1/Δ1 for r = 0, 1/r for r > 0) for r0lim pole mitigation
            Z: float : charge of the nucleus
            eps: float : Δ1 for Ha2 potential function (Ha2(r)=1/sqrt(r^2+Δ1^2) for rofs pole mitigation
        """
        self._Qx0 = Qx0
        self._Qy0 = Qy0
        self._dq = dq
        self._r0 = r0
        self._Z = Z
        self._eps = eps
        self._frac_bits = frac_bits
        logger.info(
            "VHAtom2.__init__:   dq = %f, r0=%f  r0/dq=%f, eps=%f", dq, r0, r0 / dq, eps
        )

    def __call__(self, x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
        logger.info("VHAtom2.__call__")
        if self._eps > 0:
            return self._calc_with_eps(x, y)

        xq0 = int(self._Qx0 / self._dq)
        yq0 = int(self._Qy0 / self._dq)

        if self._r0 <= 0:
            riA = numpy.sqrt(
                (numpy.square(x - self._Qx0) + numpy.square(y - self._Qy0))
            )
        else:
            # emulate fixed point calculation with frac_bits
            scale = 2 ** self._frac_bits
            hscale = 2 ** (self._frac_bits // 2)
            xq = numpy.floor(x / self._dq).astype(numpy.int32)
            yq = numpy.floor(y / self._dq).astype(numpy.int32)
            rsqq = numpy.square(xq - xq0) + numpy.square(yq - yq0)
            riA = self._dq * numpy.floor(numpy.sqrt(rsqq * scale)) / hscale

        riA[xq0, yq0] = self._r0
        logger.info("VHAtom2.Hp0 = %f", 1 / self._r0)
        qe = -1
        QA = self._Z
        varray = (qe * QA) / riA
        return varray

    def _calc_with_eps(self, x: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
        logger.info("VHAtom2._calc_with_eps")
        riA = numpy.sqrt(
            (
                numpy.square(x - self._Qx0)
                + numpy.square(y - self._Qy0)
                + numpy.square(self._dq * self._eps)
            )
        )
        qe = -1
        QA = self._Z
        varray = (qe * QA) / riA
        return varray

    @property
    def label(self):
        return f"V(q)=-1/sqrt((q-({self._Qx0},{self._Qy0}))^2+{self._delta_qQ}^2)"
