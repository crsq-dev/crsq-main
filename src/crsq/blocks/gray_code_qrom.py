""" Uniformly controlled rotation (or gray-code based QROM)
"""

import time
import logging
import numpy as np

from qiskit import QuantumRegister
from crsq_heap.heap import Frame, Binding

logger = logging.getLogger(__name__)
LOG_TIME_THRESH = 1

def _bin2gray(i: int):
    return i ^ (i >> 1)

def _bitwise_dot(a: int, b: int):
    p = 0
    s = a & b
    while s > 0:
        if s & 1:
            p += 1
        s >>= 1
    return p

def _bit_diff_index(a: int, b: int):
    diff = a ^ b
    index = -1
    while diff > 0:
        index += 1
        diff >>= 1
    return index

def _m_k(i: int, j: int):
    """ the item of the M^k matrix. (not a power of k)
    """
    gj = _bin2gray(j)
    dp = _bitwise_dot(i, gj)
    if dp & 1:
        return -1
    else:
        return 1

class GrayCodeQrom(Frame):
    """ QROM gate based on the uniformly controlled rotation
        algorithm.
        
        [Möttönen M, Vartiainen JJ, Bergholm V, Salomaa MM.
        Quantum circuits for general multiqubit gates.
        Phys Rev Lett. 2004;93:130502.]
    """
    def __init__(
        self,
        k: int,
        alpha: np.ndarray,
        build=True
    ):
        super().__init__(label="GrayCodeQROM")
        logger.info("start: GrayCodeQrom()")
        t1 = time.time()
        self._k = k
        self._alpha = alpha
        self.allocate_registers()
        if build:
            self.build_circuit()
        t2 = time.time()
        dt = t2 - t1
        if dt > LOG_TIME_THRESH:
            logger.info("end  : GrayCodeQrom() %f msec", round(dt * 1000))
    
    def allocate_registers(self):
        k = self._k
        self._x = QuantumRegister(k, "x")
        self._target = QuantumRegister(1, "t")
        self.add_param(self._x, self._target)
    
    def build_circuit(self):
        k = self._k
        nk = 1 << k
        alpha = self._alpha
        theta = np.ndarray(nk, dtype = float)
        c1 = 1.0 / nk
        for i in range(nk):
            th = 0
            for j in range(nk):
                inv_mk_ij = c1 * _m_k(j, i)
                th += inv_mk_ij * alpha[j]
            theta[i] = th
        qc = self._circuit
        tb = self._target[0]
        g0 = 0
        for i in range(nk):
            qc.rz(theta[i], tb)
            g1 = _bin2gray((i+1) % nk)
            ctrl = _bit_diff_index(g0, g1)
            g0 = g1
            qc.cx(ctrl, tb)

    def bind(self, x: QuantumRegister, t: QuantumRegister):
        return Binding(self, {"x": x, "t": t})
