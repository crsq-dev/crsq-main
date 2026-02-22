""" 3D-model of hydrogen

"""
import math
import numpy
import scipy.special as sp
import logging

logger = logging.getLogger(__name__)

class PsiH3D:
    """ Hydrogen atom, 3 dimensional model

        A callable object to calculate the wave function of the hydrogen atom in 2D model.

        Args:
            (Qx0, Qy0, Qz0): float : center of the potential
            n: int : primary quantum number
            l: int : azimuthal quantum number
            m: int : magnetic quantum number
    """
    def __init__(self, Qx0: float, Qy0: float, Qz0: float, n: int, l: int, m: int):
        logger.info("PsiH3D.__init__: Qx0=%f, Qy0=%f, Qz0=%f, n=%d, l=%d, m=%d", Qx0, Qy0, Qz0, n, l, m)
        self._Qx0 = Qx0
        self._Qy0 = Qy0
        self._Qz0 = Qz0
        self._a0 = 1.0
        self._n = n
        self._l = l
        self._m = m
        self._scale = numpy.sqrt((2 / n) ** 3 * math.factorial(n - l - 1) / (2 * n * math.factorial(n + l)))

    def __call__(self, x: numpy.ndarray, y: numpy.ndarray, z: numpy.ndarray) -> numpy.ndarray:
        """ calculate the wave function of the hydrogen atom in 3D model.

        Args:
            qxv: numpy.ndarray[(M,M)] : x coordinate values
            qyv: numpy.ndarray[(M,M)] : y coordinate values
            qzv: numpy.ndarray[(M,M)] : z coordinate values
        Returns:
            psi: numpy.ndarray[(M,M)] : wave function values
        """
        logger.info("PsiH3D.__call__")
        dxv = x - self._Qx0
        dyv = y - self._Qy0
        dzv = z - self._Qz0
        rv = numpy.sqrt(dxv * dxv + dyv * dyv + dzv * dzv)
        # radial part
        rho = 2 * rv / (self._n * self._a0)
        L = sp.genlaguerre(self._n - self._l - 1, 2 * self._l + 1)(rho)
        R = self._scale * numpy.exp(-rho / 2) * rho ** self._l * L
        # angular part
        theta = numpy.arccos(dzv / rv)
        phi = numpy.arctan2(dyv, dxv)
        Y = sp.sph_harm(self._m, self._l, phi, theta)
        psi = R * Y
        return psi

    def R(self, x, y, z):
        """ calculate the radial part of the wave function of the hydrogen atom in 3D model."""
        dxv = x - self._Qx0
        dyv = y - self._Qy0
        dzv = z - self._Qz0
        rv = numpy.sqrt(dxv * dxv + dyv * dyv + dzv * dzv)
        # radial part
        rho = 2 * rv / (self._n * self._a0)
        L = sp.genlaguerre(self._n - self._l - 1, 2 * self._l + 1)(rho)
        R = self._scale * numpy.exp(-rho / 2) * rho ** self._l * L
        return R

    def Y(self, x, y, z):
        """ calculate the angular part of the wave function of the hydrogen atom in 3D model."""
        dxv = x - self._Qx0
        dyv = y - self._Qy0
        dzv = z - self._Qz0
        rv = numpy.sqrt(dxv * dxv + dyv * dyv + dzv * dzv)
        theta = numpy.arccos(dzv / rv)
        phi = numpy.arctan2(dyv, dxv)
        Y = sp.sph_harm(self._m, self._l, phi, theta)
        return Y

    @property
    def label(self):
        return f'H-3D:ψ{self._n}_{self._l}_{self._m}(r)'

    @property
    def name(self):
        return f'H3D_n_{self._n}_l_{self._l}_m_{self._m}_q0_{self._Qx0}_{self._Qy0}_{self._Qz0}'
    
    @property
    def eigen_value(self):
        return -0.5 / self._n ** 2

class VHAtom3:
    """V(x) for H atom. potential function object - 2D numpy version
    Args:
    :param Qx0: x coordinate of the potential center
    :param Qy0: y coordinate of the potential center
    :param Qz0: z coordinate of the potential center
    :param dq: grid spacing
    :param reff: effective radius for pole mitigation (for r0lim method)
    :param Z: charge of the nucleus
    :param frac_bits: number of fractional bits for fixed point representation (if use_fixed_point is True)

    """

    def __init__(
        self,
        Qx0: float,
        Qy0: float,
        Qz0: float,
        dq: float,
        reff: float,
        Z: float,
        frac_bits: int = -1,
    ):
        """
        arguments:
            Qx0, Qy0, Qz0: float : center of the potential
            dq: float : offset added to avoid division by zero
            r0: float : Δ1 for Ha1 potential function (Ha1(r)=1/Δ1 for r = 0, 1/r for r > 0) for r0lim pole mitigation
            Z: float : charge of the nucleus
            eps: float : Δ1 for Ha2 potential function (Ha2(r)=1/sqrt(r^2+Δ1^2) for rofs pole mitigation
        """
        self._Qx0 = Qx0
        self._Qy0 = Qy0
        self._Qz0 = Qz0
        self._dq = dq
        self._reff = reff
        self._Z = Z
        self._frac_bits = frac_bits
        logger.info(
            "VHAtom3.__init__:   dq = %f, reff=%f", dq, reff
        )

    def __call__(self, x: numpy.ndarray, y: numpy.ndarray, z: numpy.ndarray) -> numpy.ndarray:
        logger.info("VHAtom3.__call__")

        xq0 = int(self._Qx0 / self._dq)
        yq0 = int(self._Qy0 / self._dq)
        zq0 = int(self._Qz0 / self._dq)

        if self._frac_bits <= 0:
            riA = numpy.sqrt(
                (numpy.square(x - self._Qx0) + numpy.square(y - self._Qy0) + numpy.square(z - self._Qz0))
            )
        else:
            # emulate fixed point calculation with frac_bits
            scale = 2 ** self._frac_bits
            hscale = 2 ** (self._frac_bits // 2)
            xq = numpy.floor(x / self._dq).astype(numpy.int32)
            yq = numpy.floor(y / self._dq).astype(numpy.int32)
            zq = numpy.floor(z / self._dq).astype(numpy.int32)
            rsqq = numpy.square(xq - xq0) + numpy.square(yq - yq0) + numpy.square(zq - zq0)
            riA = self._dq * numpy.floor(numpy.sqrt(rsqq * scale)) / hscale

        riA[xq0, yq0, zq0] = self._reff
        logger.info("VHAtom3.Hp0 = %f", 1 / self._reff)
        qe = -1
        QA = self._Z
        varray = (qe * QA) / riA
        return varray


    @property
    def label(self):
        return f"V(q)=-1/sqrt((q-({self._Qx0},{self._Qy0},{self._Qz0}))^2"
