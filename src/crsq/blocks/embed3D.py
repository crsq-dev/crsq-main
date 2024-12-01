""" State preparation gate for 3D data on a trio of quantum registers
"""

import cupy as np
import math, cmath
import time
import logging

from qiskit import QuantumRegister
from crsq_heap.heap import Frame, Binding

logger = logging.getLogger(__name__)
LOG_TIME_THRESH=1

class StateEmbedGate2D(Frame):
    """ State embedding gate for 2D data in the form data[p,r,c]
        p is plane index, r is row index, c is column index
    """
    def __init__(self, data: np.ndarray, build = True):
        super().__init__()
        logger.info("start: StateEmbedGate3D()")
        t1 = time.time()
        np, nr, nc = data.shape
        num_pbits = math.ceil(math.log2(np))
        if 2**num_pbits != np:
            raise ValueError("plane count must be a power of 2")
        num_rbits = math.ceil(math.log2(nr))
        if 2**num_rbits != nr:
            raise ValueError("row count must be a power of 2")
        num_cbits = math.ceil(math.log2(nc))
        if 2**num_cbits != nc:
            raise ValueError("column count must be a power of 2")
        self._num_rbits = num_rbits
        self._num_pbits = num_pbits
        self._num_cbits = num_cbits
        self._num_bits = num_pbits + num_rbits + num_cbits
        self._label = f"emb3d({num_pbits},{num_rbits},{num_cbits})"
        self._data = data.flatten()
        self._preg: QuantumRegister = None
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
            logger.info("end  : StateEmbedGate3D() %f msec", round(dt*1000))

    def allocate_registers(self):
        """ allocate """
        self._preg = QuantumRegister(self._num_rbits, "plane")
        self._rreg = QuantumRegister(self._num_rbits, "row")
        self._creg = QuantumRegister(self._num_rbits, "col")
        self._qreg = QuantumRegister(name="q", bits = self._creg[:] + self._rreg[:] + self._preg[:])
        self.add_param(self._preg, self._rreg, self._creg)
        self._work = QuantumRegister(self._num_bits-1, "w")
        self.add_local(self._work)

    def build_circuit(self):
        """ build """
        norms = self.build_norm_tree()
        phases = self.build_phase_tree()
        n = self._num_bits

        # the top bit is treated differently from the rest,
        # so we cannot use the build_structure_for_bit method here.
        bit = n - 1

        qc = self.circuit

        s0 = norms[0][0]
        s1 = norms[1][0]
        theta = 2*math.atan2(s1,s0)

        avg0 = phases[0][0]
        avg1 = phases[1][0]
        phi = avg1 - avg0

        global_phase = (avg0 + avg1)/2
        if global_phase != 0.0:
            qc.x(self._qreg[0])
            qc.p(global_phase, self._qreg[0])
            qc.x(self._qreg[0])

        if theta == math.pi:
            qc.x(self._qreg[bit])
        elif theta == math.pi/2:
            qc.h(self._qreg[bit])
        elif theta != 0.0:
            qc.ry(theta, self._qreg[bit])
        if phi != 0.0:
            qc.rz(phi, self._qreg[bit])
        if bit >= 1:
            qc.cx(self._qreg[bit], self._work[bit-1], ctrl_state=0)
            self.build_structure_for_bit(bit-1, norms[0][1], phases[0][1])
            qc.x(self._work[bit-1])
            self.build_structure_for_bit(bit-1, norms[1][1], phases[1][1])
            qc.cx(self._qreg[bit], self._work[bit-1])
    
    def build_norm_tree(self):
        norm0 = [(abs(x),) for x in self._data]
        while len(norm0) >= 4:
            norm1 = []
            for j in range(len(norm0)//2):
                s0 = norm0[2*j][0]
                s1 = norm0[2*j+1][0]
                s = math.sqrt(s0*s0 + s1*s1)
                norm1.append((s, (norm0[2*j], norm0[2*j+1])))
            norm0 = norm1
        return norm0

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
    
    def build_structure_for_bit(self, bit: int, norms, phases):
        qc = self.circuit

        s0 = norms[0][0]
        s1 = norms[1][0]
        theta = 2*math.atan2(s1,s0)

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
            qc.crz(phi, self._work[bit], self._qreg[bit])
        if bit >= 1:
            qc.ccx(self._work[bit], self._qreg[bit], self._work[bit-1], ctrl_state="01")
            self.build_structure_for_bit(bit-1, norms[0][1], phases[0][1])
            qc.cx(self._work[bit], self._work[bit-1])
            self.build_structure_for_bit(bit-1, norms[1][1], phases[1][1])
            qc.ccx(self._work[bit], self._qreg[bit], self._work[bit-1])
        
    def bind(self, plane: QuantumRegister, row: QuantumRegister, col: QuantumRegister)-> Binding:
        """ bind """
        return Binding(self, {"plane": plane, "row": row, "col": col})

