""" 2-D wave packet related model functions
"""

import math
import numpy


class PsiWavePacket2:
    """Wave packet exp(-0.5*(q-q0)^2/sigma^2 + ik0q)"""

    def __init__(self, qx0: float, qy0: float, kx0: float, ky0: float, sigma: float):
        self._qx0 = qx0
        self._qy0 = qy0
        self._kx0 = kx0
        self._ky0 = ky0
        self._sigma = sigma

    def __call__(self, qxv: numpy.ndarray, qyv: numpy.ndarray) -> numpy.ndarray:
        sigma = self._sigma
        kx0 = self._kx0
        ky0 = self._ky0
        a = math.pow(1 / (math.pi * sigma * sigma), 0.25 * 2)
        psi = a * numpy.exp(
            -(numpy.square(qxv - self._qx0) + numpy.square(qyv - self._qy0))
            / (2 * sigma * sigma)
            + (1j * (kx0 * qxv + ky0 * qyv))
        )
        return psi

    @property
    def label(self):
        return f"ψ(q)=exp(-(q-({self._qx0},{self._qy0}))^2/{self._sigma}^2 + i*({self._kx0/math.pi:6.3f},{self._ky0/math.pi:6.3f})*πq)"

    @property
    def name(self):
        return f"wp_q0_{self._qx0}_{self._qy0}_k0_{self._kx0/math.pi}_{self._ky0/math.pi}_sig_{self._sigma}"
