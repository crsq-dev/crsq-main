""" 3D-model of hydrogen

"""
import math
import numpy
import scipy.special as sp
import logging

logger = logging.getLogger(__name__)

class PsiH3D:
    """ Hydrogen atom, 2 dimensional model

        A callable object to calculate the wave function of the hydrogen atom in 2D model.

        Args:
            (Qx0, Qy0): float : center of the potential
            delta_q: offset added to avoid division by zero
            n: int : primary quantum number
            m: int : secondary quantum number
    """
    def __init__(self, Qx0: float, Qy0: float, Qz0: float, dq: float):
        self._Qx0 = Qx0
        self._Qy0 = Qy0
        self._Qz0 = Qz0
        self._dq = dq
        logger.info("PsiH3D implementation is tentative.")

    def __call__(self, qxv: numpy.ndarray, qyv: numpy.ndarray, qzv: numpy.ndarray) -> numpy.ndarray:
        """ calculate the wave function of the hydrogen atom in 2D model.

        Args:
            qxv: numpy.ndarray[(M,M)] : x coordinate values
            qyv: numpy.ndarray[(M,M)] : y coordinate values
        Returns:
            psi: numpy.ndarray[(M,M)] : wave function values
        """
        logger.info("PsiH2D.__call__")
        dxv = qxv - self._Qx0
        dyv = qyv - self._Qy0
        dzv = qzv - self._Qz0
        # tentative
        r = numpy.sqrt(numpy.square(dxv) + numpy.square(dyv) + numpy.square(dzv))
        psi = numpy.exp(-r)
        return psi

    @property
    def label(self):
        return f'H-2D:ψ{self._n}_{self._m}(q)'

    @property
    def name(self):
        return f'H2D_n_{self._n}_m_{self._m}_q0_{self._Qx0}_{self._Qy0}'