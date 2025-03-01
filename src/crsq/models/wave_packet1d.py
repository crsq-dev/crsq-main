""" 1-D wave packet related model functions
"""

import math
import cupy as np

class PsiWavePacket:
    """Wave packet exp(-0.5*(q-q0)^2/sigma^2 + ik0q)"""

    def __init__(self, q0: float, k0: float, sigma: float):
        self._q0 = q0
        self._k0 = k0
        self._sigma = sigma

    def __call__(self, qv: np.ndarray) -> np.ndarray:
        sigma = self._sigma
        k0 = self._k0
        a = math.pow(1 / (math.pi * sigma * sigma), 0.25)
        psi = a * np.exp(
            -np.square(qv - self._q0) / (2 * sigma * sigma) + (1j * k0 * qv)
        )
        return psi

    @property
    def label(self):
        return f"ψ(q)=exp(-(q-{self._q0})^2/{self._sigma}^2 + i*{self._k0/math.pi:6.3f}*πq)"

    @property
    def name(self):
        return f"wp_q0_{self._q0}_k0_{self._k0/math.pi}_sig_{self._sigma}"

