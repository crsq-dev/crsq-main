""" State preparation gate for 2D data on a pair of quantum registers
"""

from typing import List, Tuple
import cupy as np
import math, cmath
import time
import logging

from qiskit import QuantumRegister
from crsq_heap.heap import Frame, Binding

logger = logging.getLogger(__name__)
LOG_TIME_THRESH=1

def fix_polar(p: Tuple[float, float])-> Tuple[float, float]:
    """ fix polar form """
    if p[1] >= math.pi/2:
        return (-p[0], p[1]-math.pi)
    elif p[1] <= -math.pi/2:
        return (-p[0], p[1]+math.pi)
    else:
        return p

def to_fixed_polar(data: List[complex])-> List[Tuple[float, float]]:
    """ to fixed polar form """
    return [fix_polar(cmath.polar(x)) for x in data]

class StateEmbedGate2D(Frame):
    """ State embedding gate for 2D data in the form data[r,c]
        r is row index, c is column index
    """
    def __init__(self, data: np.ndarray, build = True):
        super().__init__()
        logger.info("start: StateEmbedGate2D()")
        t1 = time.time()
        nr, nc = data.shape
        num_rbits = math.ceil(math.log2(nr))
        if 2**num_rbits != nr:
            raise ValueError("row count must be a power of 2")
        num_cbits = math.ceil(math.log2(nc))
        if 2**num_cbits != nc:
            raise ValueError("column count must be a power of 2")
        self._num_rbits = num_rbits
        self._num_cbits = num_cbits
        self._num_bits = num_rbits + num_cbits
        self._label = f"emb2d({num_rbits},{num_cbits})"
        self._data = data.flatten()
        self._rreg: QuantumRegister = None
        self._creg: QuantumRegister = None
        self._qreg: QuantumRegister = None
        self._work: QuantumRegister = None
        self.allocate_registers()
        if build:
            self.build_circuit()
        t2 = time.time()
        dt = t2 - t1
        if dt > LOG_TIME_THRESH:
            logger.info("end  : StateEmbedGate2D() %f msec", round(dt*1000))

    def allocate_registers(self):
        """ allocate """
        self._rreg = QuantumRegister(self._num_rbits, "row")
        self._creg = QuantumRegister(self._num_rbits, "col")
        self._qreg = QuantumRegister(name="q", bits = self._creg[:] + self._rreg[:])
        self.add_param(self._rreg, self._creg)
        self._work = QuantumRegister(self._num_bits-1, "w")
        self.add_local(self._work)

    def build_circuit(self):
        """ build """
        cuarray = np.asarray(self._data)
        norm = math.sqrt(np.sum(np.square(np.abs(cuarray))))
        logger.info("sqrt(Σ|Ψ^2|) = %.16f", norm)

        pdata = to_fixed_polar(self._data)
        norms = self.build_norm_tree_from_polar(pdata)
        phases = self.build_phase_tree_from_polar(pdata)

        n = self._num_bits

        self._test = np.zeros(1 << n, dtype=np.float64)
        self._test[0] = 1.0
        cols = 1 << self._num_cbits
        for c in range(cols):
            logger.info("data[0][%d] = (%.16f,%.16f)", c, self._data[c].real, self._data[c].imag)

        # the top bit is treated differently from the rest,
        # so we cannot use the build_structure_for_bit method here.
        bit = n - 1
        p = 0
        hp = 1 << bit

        qc = self.circuit

        s0 = norms[0][0]
        s1 = norms[1][0]
        theta = 2*math.atan2(s1,s0)
        nm = math.sqrt(s0*s0 + s1*s1)
        self._test[p+hp] = self._test[p] * s1/nm
        self._test[p] = self._test[p] * s0/nm

        avg0 = phases[0][0]
        avg1 = phases[1][0]
        phi = avg1 - avg0

        global_phase = (avg0 + avg1)/2
        logger.info("global_phase = %.16f", global_phase)

        if theta == math.pi:
            qc.x(self._qreg[bit])
        elif theta == math.pi/2:
            qc.h(self._qreg[bit])
        elif theta != 0.0:
            qc.ry(theta, self._qreg[bit])
        if phi != 0.0:
            logger.info("rz(%f, q%d)", phi, bit)
            qc.rz(phi, self._qreg[bit])
        if bit >= 1:
            qc.cx(self._qreg[bit], self._work[bit-1], ctrl_state=0)
            self.build_structure_for_bit(bit-1, p, norms[0][1], phases[0][1])
            qc.x(self._work[bit-1])
            self.build_structure_for_bit(bit-1, p + hp, norms[1][1], phases[1][1])
            qc.cx(self._qreg[bit], self._work[bit-1])

        for c in range(cols):
            logger.info("test[%d] = %.16f", c, self._test[c])

    def build_square_tree(self):
        square0 = [(abs(x)*abs(x),) for x in self._data]
        while len(square0) >= 4:
            square1 = []
            for j in range(len(square0)//2):
                s0 = square0[2*j][0]
                s1 = square0[2*j+1][0]
                s = s0 + s1
                square1.append((s, (square0[2*j], square0[2*j+1])))
            square0 = square1
        return square0

    def build_phase_tree(self):
        avg0 = [(cmath.phase(x),) for x in self._data]
        while len(avg0) >= 4:
            avg1 = []
            for j in range(len(avg0)//2):
                phi0 = avg0[2*j][0]
                phi1 = avg0[2*j+1][0]
                avg = (phi0 + phi1)/2
                avg1.append((avg, (avg0[2*j], avg0[2*j+1])))
            avg0 = avg1
        return avg0

    def build_norm_tree_from_polar(self, pdata):
        norm0 = [(p[0],) for p in pdata]
        while len(norm0) >= 4:
            norm1 = []
            for j in range(len(norm0)//2):
                s0 = norm0[2*j][0]
                s1 = norm0[2*j+1][0]
                s = math.sqrt(s0*s0 + s1*s1)
                norm1.append((s, (norm0[2*j], norm0[2*j+1])))
            norm0 = norm1
        return norm0

    def build_phase_tree_from_polar(self, pdata):
        avg0 = [(p[1],) for p in pdata]
        while len(avg0) >= 4:
            avg1 = []
            for j in range(len(avg0)//2):
                phi0 = avg0[2*j][0]
                phi1 = avg0[2*j+1][0]
                avg = (phi0 + phi1)/2
                avg1.append((avg, (avg0[2*j], avg0[2*j+1])))
            avg0 = avg1
        return avg0

    def build_structure_for_bit(self, bit: int, p: int, norms, phases):
        qc = self.circuit

        hp = 1 << bit

        s0 = norms[0][0]
        s1 = norms[1][0]
        theta = 2*math.atan2(s1,s0)
        
        nm = math.sqrt(s0*s0 + s1*s1)
        self._test[p+hp] = self._test[p] * s1/nm
        self._test[p] = self._test[p] * s0/nm

        avg0 = phases[0][0]
        avg1 = phases[1][0]
        phi = avg1 - avg0

        if theta == math.pi:
            qc.cx(self._work[bit], self._qreg[bit])
        elif theta == math.pi/2:
            qc.ch(self._work[bit], self._qreg[bit])
        elif theta != 0.0:
            qc.cry(theta, self._work[bit], self._qreg[bit])
        if phi != 0.0:
            logger.info("crz(%f, w%d, q%d)", phi, bit, bit)
            qc.crz(phi, self._work[bit], self._qreg[bit])
        if bit >= 1:
            qc.ccx(self._work[bit], self._qreg[bit], self._work[bit-1], ctrl_state="01")
            self.build_structure_for_bit(bit-1, p, norms[0][1], phases[0][1])
            qc.cx(self._work[bit], self._work[bit-1])
            self.build_structure_for_bit(bit-1, p + hp, norms[1][1], phases[1][1])
            qc.ccx(self._work[bit], self._qreg[bit], self._work[bit-1])
        
    def bind(self, row: QuantumRegister, col: QuantumRegister)-> Binding:
        """ bind """
        return Binding(self, {"row": row, "col": col})

