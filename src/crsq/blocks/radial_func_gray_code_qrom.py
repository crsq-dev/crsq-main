""" Coulomb potential term implemented with gray code QROM(uniformly controlled rotation gate)
"""

""" state preparation gates (unary iteration using ancilla qubits)
"""

import math
import time
import logging
import numpy as np

from qiskit import QuantumRegister
from qiskit.circuit.library import UCRZGate
from crsq_heap.heap import Frame, Binding
from crsq.blocks import gray_code_qrom

logger = logging.getLogger(__name__)
LOG_TIME_THRESH = 1

class RadialFuncGrayCodeQrom(Frame):
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
        self._data = np.ndarray(M*M, dtype = float)
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
                self._data[i*M + j] = -2.0 * psi

    def allocate_registers(self):
        n = self._num_coord_bits
        self._x = QuantumRegister(n, "x")
        self._y = QuantumRegister(n, "y")
        self._t = QuantumRegister(1, "target")
        self.add_param(self._x, self._y, self._t)
        self._regs = self._x[:] + self._y[:] + self._t[:]
    
    @property
    def regs(self):
        return self._regs

    def build_circuit(self):
        k = self._num_coord_bits * 2
        alpha = self._data
        indexbits = QuantumRegister(name="x", bits=self._x[:] + self._y[:])
        use_ucrz_gate = True
        if use_ucrz_gate:
            ucrz = UCRZGate(alpha.tolist())
            self.circuit.append(ucrz, self._t[:] + indexbits[:])
        else:
            gcqrom = gray_code_qrom.GrayCodeQrom(k, alpha)
            self.invoke(gcqrom.bind(x=indexbits, t=self._t))
    
    def bind(self, x: QuantumRegister, y: QuantumRegister, target: QuantumRegister):
        return Binding(self, {"x": x, "y": y, "target": target})


class RadialFuncGrayCodeQromTestBoard(Frame):
    def __init__(
        self,
        n: int,
        dq: float,
        rfunc: callable,
        use_symmetry=True,
        use_transpose=True,
        verbose=True,
    ):
        super().__init__(label="RFQTest")
        self._n = n
        self._dq = dq
        self._rfunc = rfunc
        self._use_symmetry = use_symmetry
        self._use_transpose = use_transpose
        self._verbose = verbose
        self.allocate_registers()
        self.build_circuit()

    def allocate_registers(self):
        self._x = QuantumRegister(self._n, "x")
        self._y = QuantumRegister(self._n, "y")
        self._target = QuantumRegister(1, "target")
        self.add_param(self._x, self._y, self._target)

    def build_circuit(self):
        qc = self.circuit
        qc.h(self._x)
        qc.h(self._y)
        self._rfq = RadialFuncGrayCodeQrom(
            self._n,
            self._dq,
            self._rfunc
        )
        self.invoke(self._rfq.bind(x=self._x, y=self._y, target=self._target), invoke_as_instruction=True)

    @property
    def regs(self):
        return self._rfq.regs
