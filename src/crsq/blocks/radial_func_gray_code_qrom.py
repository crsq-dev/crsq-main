""" Coulomb potential term implemented with gray code QROM(uniformly controlled rotation gate)
"""

""" state preparation gates (unary iteration using ancilla qubits)
"""

from typing import List
import math
import cmath
import time
import logging
import numpy as np

from qiskit import QuantumRegister
from crsq_heap.heap import Frame, Binding
import crsq_arithmetic as ari
from crsq.blocks import gray_code_qrom

logger = logging.getLogger(__name__)
LOG_TIME_THRESH = 1

class RadialFuncGrayCodeQROM(Frame):
    """ 2D- Radial function implemented using gray code QROM
    """

    def __init__(
        self,
        num_coord_bits: int,
        dq: float,
        rfunc: callable,
        build = True
    ):
        super().__init__(label="RadialFuncGrayCodeQROM")
        logger.info("start: RadialFuncGrayCodeQROM")
        t1 = time.time()
        self._num_coord_bits = num_coord_bits
        self._dq = dq
        self._rfunc = rfunc
        self._prepare_data()
        self.allocate_registers()
        if build:
            self.build_circuit()
        t2 = time.time()
        dt = t2 - t1
        if dt > LOG_TIME_THRESH:
            logger.info("end : RadialFuncGrayCodeQROM() %f msec", round(dt * 1000))
    
    def _prepare_data(self):
        n = self._num_coord_bits
        M = 1 << n
        self._data = np.ndarray((M, M), dtype = float)
        for i in range(M):
            si = (i + M // 2) % M - (M // 2)
            y = (si + 0.5) * self._dq
            for j in range(M):
                sj = (j + M // 2) % M - (M // 2)
                x = (sj + 0.5) * self._dq
                r = math.sqrt(x * x + y * y)
                psi = self._rfunc(r)
                if abs(psi) > math.pi:
                    logger.warning("x=%f, y=%f, r=%f, psi=%f", x, y, r, psi)
                self._data[i, j] = -2.0 * psi

    def allocate_registers(self):
        n = self._num_coord_bits
        self._x = QuantumRegister(n, "x")
        self._y = QuantumRegister(n, "y")
        self._t = QuantumRegister(1, "target")
        self.add_param(self._x, self._y, self._t)
    
    def build_circuit(self):
        k = self._num_coord_bits * 2
        alpha = self._data.flatten()
        xbits = QuantumRegister(name="x", bits=self._x[:] + self._y[:])
        gcqrom = gray_code_qrom.GrayCodeQrom(k, alpha)
        self.invoke(gcqrom.bind(x=xbits, t=self._t))
    
    def bind(self, x: QuantumRegister, y: QuantumRegister, target: QuantumRegister):
        return Binding(self, {"x": x, "y": y, "target": target})
