""" state preparation gates (unary iteration using ancilla qubits)
"""

from typing import List
import math
import cmath
import time
import logging

from qiskit import QuantumRegister
from crsq_heap.heap import Frame, Binding

logger = logging.getLogger(__name__)
LOG_TIME_THRESH=1

class StateEmbedGate2(Frame):
    """ State embedding gate
    """
    def __init__(self, data: List[float] | List[complex], build=True):
        super().__init__()
        logger.info("start: StateEmbedGate()")
        t1 = time.time()
        num_bits = math.ceil(math.log2(len(data)))
        if 2**num_bits != len(data):
            raise ValueError("data length must be a power of 2")
        self._num_bits = num_bits
        self._label = f"emb({num_bits})"
        self._data = data
        self._qreg: QuantumRegister
        self._work: QuantumRegister
        self.allocate_registers()
        if build:
            self.build_circuit()
        t2 = time.time()
        dt = t2 - t1
        if dt > LOG_TIME_THRESH:
            logger.info("end  : StateEmbedGate() %f msec", round(dt*1000))

    def allocate_registers(self):
        """ allocate """
        self._qreg = QuantumRegister(self._num_bits, "q")
        self.add_param(self._qreg)
        self._work = QuantumRegister(self._num_bits-1, "w")
        self.add_local(self._work)

    def build_circuit(self):
        """ build """
        norms = self.build_norm_tree()
        phases = self.build_phase_tree()
        n = self._num_bits
        qc = self.circuit

        s0 = norms[0][0]
        s1 = norms[1][0]
        theta = 2*math.atan2(s1,s0)

        avg0 = phases[0][0]
        avg1 = phases[1][0]
        phi = avg1 - avg0

        bit = n - 1
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
            self.build_structure_for_bit(bit-1, norms[0][1], phases[1][1])
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
            self.build_structure_for_bit(bit-1)
            qc.cx(self._work[bit], self._work[bit-1])
            self.build_structure_for_bit(bit-1)
            qc.ccx(self._work[bit], self._qreg[bit], self._work[bit-1])
        

    def bind(self, q: QuantumRegister)-> Binding:
        """ bind """
        return Binding(self, {"q": q})
