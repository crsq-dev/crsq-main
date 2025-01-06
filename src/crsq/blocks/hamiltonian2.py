""" Hamiltonian calculation by QROM lookup, polynomial interpolation, and a Newton-Raphson iteration.

"""

import math
from qiskit.circuit import QuantumRegister, QuantumCircuit

from crsq_heap import heap
from crsq.blocks import wave_function
import crsq_arithmetic as ari
import crsq_arithmetic.ast as ast


class InverseSquareRoot(heap.Frame):
    """A block to calculate 1/sqrt(x) by QROM lookup, polynomial interpolation, and a Newton-Raphson iteration."""

    def __init__(
        self,
        wfr_spec: wave_function.WaveFunctionRegisterSpec,
        allocate=True,
        build=True,
    ):
        super().__init__(label="ISQRT")
        self._wfr_spec = wfr_spec
        if allocate:
            self.allocate_registers()
            if build:
                self.build_circuit()

    def allocate_registers(self):
        nkey = self._wfr_spec.num_coordinate_bits
        nval = self._wfr_spec.num_coordinate_bits
        # input: x = dx**2 + dy**2 + dz**2
        self._qx = QuantumRegister(nval * 2 + 2, "x")
        # output: y = 1/sqrt(x)
        self._qy = QuantumRegister(nval + 1, "y")
        self.add_param(self._qx, self._qy)
        self._qdarray = [QuantumRegister(nval * 2 + 2, f"a{i}") for i in range(5)]
        self.add_local(*self._qdarray)

    def build_circuit(self):
        n = self._wfr_spec.num_coordinate_bits
        self._build_qrom()
        # calculate y0 = a0 - (x-a4)*(a1-(x-a4)*(a2-a3*(x-a4)))
        # = a0 - c*(a1-c*(a2-a3*c))
        # = a0 - c*(a1-c*(a2-d))
        # = a0 - c*(a1-e*c)
        # = a0 - c*(a1-f)
        # = a0 - g*c
        # = a0 - h
        # = i
        scope = ast.new_scope(self)
        astx = scope.register(self._qx, fraction_bits=2 * n + 2, signed=False)
        asty: ast.QuantumValue = scope.register(
            self._qy, fraction_bits=n + 1, signed=False
        )
        asta0 = scope.register(self._qdarray[0], fraction_bits=2 * n + 2, signed=False)
        asta1 = scope.register(self._qdarray[1], fraction_bits=2 * n + 2, signed=False)
        asta2 = scope.register(self._qdarray[2], fraction_bits=2 * n + 2, signed=False)
        asta3 = scope.register(self._qdarray[3], fraction_bits=2 * n + 2, signed=False)
        asta4 = scope.register(self._qdarray[4], fraction_bits=2 * n + 2, signed=False)
        astx -= asta4
        astc = astx.adjust_precision(n * 2 + 2, n * 2 + 2)
        asta3 *= astc
        astd = asta3.adjust_precision(n * 2 + 2, n * 2 + 2)
        asta2 -= astd
        aste = asta2.adjust_precision(n * 2 + 2, n * 2 + 2)
        aste *= astc
        astf = aste.adjust_precision(n * 2 + 2, n * 2 + 2)
        asta1 -= astf
        astg = asta1.adjust_precision(n * 2 + 2, n * 2 + 2)
        astg *= astc
        asth = astg.adjust_precision(n * 2 + 2, n * 2 + 2)
        asta0 -= asth
        asta0 = asta0.adjust_precision(n * 2 + 2, n * 2 + 2)

        # Newton-Raphson iteration
        # y1 = y0*(3-y0^2*x)/2
        #    = y0*(3-a*x)/2
        #    = y0*(3-b)/2
        #    = y0*c/2
        #    = d/2
        y0 = asta0.adjust_precision(n + 1, n + 1)
        ya = scope.square(y0)
        ya = ya.adjust_precision(n + 1, n + 1)
        yastx = astx.adjust_precision(n + 1, n + 1)
        yb = ya * yastx
        yb = yb.adjust_precision(n + 1, n + 1)
        y3 = asty
        y3 += scope.constant(3 * (2**-4), n + 1, n + 1)
        y3 = y3.adjust_precision(n + 1, n + 1)
        y3 -= yb
        yc = y3.adjust_precision(n + 1, n + 1)
        yd = y0 * yc
        # y1 = yd.mult_pow2(-1)     # y1 = yd/2
        y1 = yd.adjust_precision(n + 1, n + 1)

        scope.build_circuit()

        scope.build_inverse_circuit()
        self._build_qrom(inverse=True)

    def _build_qrom(self, inverse=False):
        nkey = self._wfr_spec.num_coordinate_bits
        data = self._make_qrom_data()
        qc = self.circuit

        data_regs = self._qdarray
        data_reg_bits = []
        data_reg_num_bits = []
        for data_reg in data_regs:
            data_reg_bits += data_reg[:]
            data_reg_num_bits.append(data_reg.size)

        qrom_tmp = self.allocate_temp_bits(nkey * 2 + 2)
        qrom_gate = ari.vsqrom2_gate(nkey * 2 + 2, data_reg_num_bits, data)
        if inverse:
            qrom_gate = qrom_gate.inverse()
            qrom_gate.label = "vsqrom2\u2020"
        qc.append(qrom_gate, self._qx[:] + data_reg_bits + qrom_tmp)
        self.free_temp_bits(qrom_tmp)

    def bind(self, x, y):
        return heap.Binding(self, {"x": x, "y": y})

    """ coefficients for the polynomial interpolation for the range [1,3/2]
    """
    _coeff_a = [
        0.99994132489119882162,
        0.49609891915903542303,
        0.33261112772430493331,
        0.14876762006038398086,
    ]

    """ coefficients for the range [3/2, 2]
    """
    _coeff_b = [
        0.81648515205385221995,
        0.27136515484240234115,
        0.12756148214815175348,
        0.044753028579153842218,
    ]

    def _make_qrom_data(self):
        rows = []
        wfr_spec = self._wfr_spec
        dq = wfr_spec.delta_q
        nkey = wfr_spec.num_coordinate_bits * 2 + 2
        num_bits = nkey
        scale = 1 << (num_bits - 1)
        sqrt2 = math.sqrt(2)
        # rows[0] is special. f(x) = 1/sqrt(dq)
        a0 = 1.0 / math.sqrt(dq)
        a4 = 0
        row = [int(scale * a0), 0, 0, 0, a4]
        rows.append(row)
        # rows[1] to rows[3] are special. f(k) = 1/sqrt(k)
        for k in range(1, 4):
            a0 = 1.0 / math.sqrt(k)
            a4 = k
            row = [int(scale * a0), 0, 0, 0, a4]
            rows.append(row)
        # rows[4] and beyond follow the formula.
        for k in range(nkey - 2):
            a0 = int(scale * (2 ** (-k / 2)) * InverseSquareRoot._coeff_a[0])
            a1 = int(scale * (2 ** (-3 * k / 2)) * InverseSquareRoot._coeff_a[1])
            a2 = int(scale * (2 ** (-5 * k / 2)) * InverseSquareRoot._coeff_a[2])
            a3 = int(scale * (2 ** (-7 * k / 2)) * InverseSquareRoot._coeff_a[3])
            a4 = int(2 ** (k + 2))
            row = [a0, a1, a2, a3, a4]
            rows.append(row)
            b0 = int(scale * (2 ** (-k / 2)) * InverseSquareRoot._coeff_b[0])
            b1 = int(scale * (2 ** (-3 * k / 2)) * InverseSquareRoot._coeff_b[1])
            b2 = int(scale * (2 ** (-5 * k / 2)) * InverseSquareRoot._coeff_b[2])
            b3 = int(scale * (2 ** (-7 * k / 2)) * InverseSquareRoot._coeff_b[3])
            b4 = int(3 / 2 * (2 ** (k + 2)))
            row = [b0, b1, b2, b3, b4]
            rows.append(row)
        return rows
