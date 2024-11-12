""" Hamiltonian calculation by QROM lookup, polynomial interpolation, and a Newton-Raphson iteration.

"""

import math
from qiskit.circuit import QuantumRegister, QuantumCircuit

from crsq_heap import heap
from crsq.blocks import wave_function
import crsq_arithmetic as ari
import crsq_arithmetic.ast as ast

class InverseSquareRoot(heap.Frame):
    """ A block to calculate 1/sqrt(x) by QROM lookup, polynomial interpolation, and a Newton-Raphson iteration.
    """

    def __init__(self, wfr_spec: wave_function.WaveFunctionRegisterSpec, allocate=True, build=True):
        self._wfr_spec = wfr_spec
        if allocate:
            self.allocate_registers()
            if build:
                self.build_circuit()

    def allocate_registers(self):
        nkey = self._wfr_spec.num_coordinate_bits
        nval = self._wfr_spec.num_coordinate_bits
        self._qx = QuantumRegister(nval*2+2, 'x')
        self._qy = QuantumRegister(nval+1, 'y')
        self.add_param(self._qx, self._qy)
        self._qk = QuantumRegister(nkey, 'k')
        self._qdarray = [QuantumRegister(nval, f'a{i}') for i in range(1, 5)]
        self.add_local(self._qk, self._qdarray)
        self._qtmp = self.allocate_temp_bits(nkey)
    
    def build_circuit(self):
        n = self._wfr_spec.num_coordinate_bits
        self._build_qrom()
        scope = ast.new_scope(self)
        astx = scope.register(self._qx, fraction_bits=0, signed=False)
        asta0 = scope.register(self._qy, fraction_bits=n, signed=False)
        asta1 = scope.register(self._qdarray[0], fraction_bits=n, signed=False)
        asta2 = scope.register(self._qdarray[1], fraction_bits=n, signed=False)
        asta3 = scope.register(self._qdarray[2], fraction_bits=n, signed=False)
        asta4 = scope.register(self._qdarray[3], fraction_bits=0, signed=False)
        astx -= asta4
        prod34 = astx * asta3
        asta2 -= prod34
        prod24 = astx * asta2
        asta1 -= prod24
        prod14 = astx * asta1
        asta0 -= prod14
    
    def _build_qrom(self):
        data = self._make_qrom_data()
        qc = self.circuit
        data_regs = [self._qy] + self._qdarray
        ari.vsqrom2(qc, self._qk, data_regs, [], self._qtmp, data)

    def bind(self, x, y):
        return heap.Binding(self, {
            "x": x,
            "y": y
        })


    """ coefficients for the polynomial interpolation
    """
    _coeff_a = [
        0.99994132489119882162,
        0.49609891915903542303,
        0.33261112772430493331,
        0.14876762006038398086
    ]

    def _make_qrom_data(self):
        rows = []
        wfr_spec = self._wfr_spec
        nkey = wfr_spec.num_coordinate_bits
        scale = 1 << (num_bits -1)
        sqrt2 = math.sqrt(2)
        num_bits = wfr_spec.num_coordinate_bits
        for k in range(nkey):
            a0 = int(scale * (2 ** (-k/2)) * InverseSquareRoot._coeff_a[0])
            a1 = int(scale * (2 ** (-3*k/2)) * InverseSquareRoot._coeff_a[1])
            a2 = int(scale * (2 ** (-5*k/2)) * InverseSquareRoot._coeff_a[2])
            a3 = int(scale * (2 ** (-7*k/2)) * InverseSquareRoot._coeff_a[3])
            a4 = 2 ** k
            row = [a0, a1, a2, a3, a4]
            rows.append(row)
        return rows
