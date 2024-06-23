""" QFT application block
"""

import time
from typing import List
import logging
from qiskit import QuantumRegister
from qiskit.circuit import library as circuit_lib


from crsq_heap import heap
from crsq.blocks import wave_function

logger=logging.getLogger(__name__)
LOG_TIME_THRESH=1


class QFTOnWaveFunctionsBlock(heap.Frame):
    """ QFT application block"""

    def __init__(self, wfr_spec: wave_function.WaveFunctionRegisterSpec,
                 on_electrons:bool = False, on_nucleus:bool = False,
                 inverse: bool = False, build: bool = True):
        super().__init__()
        t1 = time.time()
        self._wfr_spec = wfr_spec
        self._on_electrons = on_electrons
        self._on_nucleus = on_nucleus
        self._eregs: List[List[List[QuantumRegister]]] = []
        self._nregs: List[List[List[QuantumRegister]]] = []
        self._inverse = inverse
        if inverse:
            dag = "\u2020"
        else:
            dag = ""
        self._label = f"nQFT{dag}"
        self.allocate_registers()
        if build:
            self.build_circuit()
        dt = time.time() - t1
        if dt > LOG_TIME_THRESH:
            logger.info("QFTOnWaveFunctionsBlock took %d msec", round(dt*1000))

    def allocate_registers(self):
        """ allocate registers """
        if self._on_electrons:
            self._eregs = self._wfr_spec.allocate_elec_registers()
            self.add_param(('eregs', self._eregs))
        if self._on_nucleus:
            self._nregs = self._wfr_spec.allocate_nucl_registers()
            self.add_param(('nregs', self._nregs))

    def build_circuit(self):
        """ build circuit """
        wfr_spec = self._wfr_spec
        qc = self.circuit

        num_bits = wfr_spec.num_coordinate_bits
        if self._on_electrons:
            for elec in self._eregs:
                # the MSB of the electron index is for the spin,
                # and should be excluded from the QFT.
                for d in range(wfr_spec.dimension):
                    qc.append(circuit_lib.QFT(num_bits, inverse=self._inverse), elec[d][:])
        if self._on_nucleus:
            for nuc in self._nregs:
                for d in range(wfr_spec.dimension):
                    qc.append(circuit_lib.QFT(num_bits, inverse=self._inverse), nuc[d][:])

    def bind(self,
             eregs: List[List[List[QuantumRegister]]]|None = None,
             nregs: List[List[List[QuantumRegister]]]|None = None) -> heap.Binding:
        """ produce binding"""
        regs = {}
        if eregs:
            regs["eregs"] = eregs
        if nregs:
            regs["nregs"] = nregs
        return heap.Binding(self, regs)
