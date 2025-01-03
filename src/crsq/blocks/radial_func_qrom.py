""" Coulomb potential term implemented with QROM
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

logger = logging.getLogger(__name__)
LOG_TIME_THRESH = 1


class RadialFuncQrom(Frame):
    """2D- Radial function implemented as QROM

    Args:
        n: bits per dimension
        dq: grid spacing
        rfunc: function of the form rfunc(r: float) -> float
        use_symmetry: use x-axis or y-axis symmetry of the function f(x,y) = f(-x,y) , f(x,y) = f(x, -y)
        use_transpose: use transpositional symmetry of the function f(x,y) = f(y,x)
    """

    def __init__(
        self,
        n: int,
        dq: float,
        rfunc: callable,
        use_symmetry=True,
        use_transpose=True,
        build=True,
        verbose=False,
    ):
        super().__init__(label="RadialFuncQROM")
        logger.info("start: RadialFuncQrom()")
        t1 = time.time()
        self._n = n
        self._use_symmetry = use_symmetry
        self._use_transpose = use_transpose
        if use_transpose and not use_symmetry:
            raise ValueError("use_transpose requires use_symmetry")
        logger.info("use_symmetry: %s, use_transpose: %s", use_symmetry, use_transpose)
        self._verbose = verbose
        self._dq = dq  # grid spacing
        self._rfunc = rfunc
        self._prepare_data()
        self.allocate_registers()
        if build:
            self.build_circuit()
        t2 = time.time()
        dt = t2 - t1
        if dt > LOG_TIME_THRESH:
            logger.info("end  : RadialFuncQrom() %f msec", round(dt * 1000))

    def _prepare_data(self):
        n = self._n
        M = 2**n
        HM = M // 2
        if self._use_symmetry:
            if self._use_transpose:
                self._prepare_data_cond(lambda i, j: j <= HM and i <= j)
            else:
                self._prepare_data_cond(lambda i, j: i <= HM and j <= HM)
        else:
            self._prepare_data_cond(lambda i, j: True)

    def _prepare_data_cond(self, condition_func):
        n = self._n
        M = 2**n
        self._data = np.ndarray((M, M), dtype=float)
        self._has_data_y = np.ndarray((n, M, M), dtype=int)
        self._has_data_x = np.ndarray((n, M, M), dtype=int)
        for i in range(M):
            si = (i + M // 2) % M - (M // 2)
            y = si * self._dq
            for j in range(M):
                if condition_func(i, j):
                    sj = (j + M // 2) % M - (M // 2)
                    x = sj * self._dq
                    r = math.sqrt(x * x + y * y)
                    if r == 0:
                        r = self._dq / 2
                    psi = self._rfunc(r)
                    if abs(psi) > math.pi:
                        logger.warning("x=%f, y=%f, r=%f, psi=%f", x, y, r, psi)
                    self._data[i, j] = psi
                    self._has_data_y[0, i, j] = 1
                else:
                    self._has_data_y[0, i, j] = 0
        self._make_has_data_tables()

    def _make_has_data_tables(self):
        n = self._n
        M = 2**n
        w = M
        for k in range(0, n - 1):
            hw = w // 2
            for i in range(0, hw):
                for j in range(0, w):
                    self._has_data_x[k, i, j] = (
                        self._has_data_y[k, 2 * i, j]
                        or self._has_data_y[k, 2 * i + 1, j]
                    )
            for i in range(0, hw):
                for j in range(0, hw):
                    self._has_data_y[k + 1, i, j] = (
                        self._has_data_x[k, i, 2 * j]
                        or self._has_data_x[k, i, 2 * j + 1]
                    )
            w = hw
        k = n - 1
        for i in range(0, hw):
            for j in range(0, w):
                self._has_data_x[k, i, j] = (
                    self._has_data_y[k, 2 * i, j] or self._has_data_y[k, 2 * i + 1, j]
                )
        if self._verbose:
            self._print_tables()

    def _print_tables(self):
        n = self._n
        M = 2**n
        w = M
        hw = w // 2
        for k in range(n):
            print(f"has_data_y[{k}]")
            for i in range(w):
                print(self._has_data_y[k, i, :w])
            print(f"has_data_x[{k}]")
            for i in range(hw):
                print(self._has_data_x[k, i, :w])
            w = hw
            hw = w // 2

    def allocate_registers(self):
        """allocate"""
        n = self._n
        self._x = QuantumRegister(n, "x")
        self._y = QuantumRegister(n, "y")
        self.add_param(self._x, self._y)

        # work for qrom
        self._wx = QuantumRegister(n, "wx")
        self._wy = QuantumRegister(n, "wy")
        self.add_local(self._wx, self._wy)

        qubits = []
        for i in range(n - 1, -1, -1):
            qubits += [self._x[i], self._wx[i], self._y[i], self._wy[i]]

        if self._use_symmetry:
            # sign
            self._sx = QuantumRegister(1, "sx")
            self._sy = QuantumRegister(1, "sy")
            # carry for abs
            self._cr = QuantumRegister(n - 1, "cr")

            self.add_local(self._sx, self._sy, self._cr)
            qubits += self._sx[:] + self._sy[:] + self._cr[:]

            if self._use_transpose:
                # compare result
                self._cz = QuantumRegister(1, "cz")
                self.add_local(self._cz)
                qubits += self._cz[:]

        bit_index = [self.circuit.qubits.index(qubit) for qubit in qubits]
        self._regs = bit_index

    @property
    def regs(self):
        return self._regs

    def build_circuit(self):
        """build"""
        qc = self.circuit
        n = self._n

        if self._use_symmetry:
            qc.append(ari.absolute_gate(n), self._x[:] + self._sx[:] + self._cr[:])
            qc.append(ari.absolute_gate(n), self._y[:] + self._sy[:] + self._cr[:])
            if self._use_transpose:
                qc.append(
                    ari.cdk_comparator_gate(n),
                    self._y[:] + self._x[:] + self._cz[:] + self._cr[0:1],
                )
                for i in range(n):
                    qc.cswap(self._cz[0], self._x[i], self._y[i])
        qc.barrier()
        self._build_area_x_top()
        qc.barrier()

        if self._use_symmetry:
            if self._use_transpose:
                for i in range(n - 1, -1, -1):
                    qc.cswap(self._cz[0], self._x[i], self._y[i])

                cmp_dag_gate = ari.cdk_comparator_gate(n).inverse()
                cmp_dag_gate.label = f"cmp\u2020({n})"
                qc.append(
                    cmp_dag_gate, self._y[:] + self._x[:] + self._cz[:] + self._cr[0:1]
                )

            absy_dag_gate = ari.absolute_gate(n).inverse()
            absy_dag_gate.label = f"abs\u2020({n})"
            qc.append(absy_dag_gate, self._y[:] + self._sy[:] + self._cr[:])

            absx_dag_gate = ari.absolute_gate(n).inverse()
            absx_dag_gate.label = f"abs\u2020({n})"
            qc.append(absx_dag_gate, self._x[:] + self._sx[:] + self._cr[:])

    def _build_area_x_top(self):
        qc = self.circuit
        n = self._n
        k = n - 1
        xi = 0
        yi = 0
        wxk = None
        wyk = None

        if self._has_data_x[k, yi, xi]:
            if self._has_data_x[k, yi, xi + 1]:
                # both x low and x high have data
                # focus on lower half.
                wxk = self._wx[k]
                qc.cx(self._x[k], wxk, ctrl_state="0")
                # do the lower half
                self._build_area_y(xi, yi * 2, k, wxk, wyk)
                # middle.
                # switch focus to upper half
                qc.x(wxk)
                # do the upper half
                self._build_area_y(xi + 1, yi * 2, k, wxk, wyk)
                # reset focus
                qc.cx(self._x[k], wxk, ctrl_state="1")
            else:
                # only x low has data
                # focus on lower half.
                wxk = self._wx[k]
                qc.x(wxk)
                # do the lower half
                self._build_area_y(xi, yi * 2, k, wxk, wyk)
                # reset focus
                qc.x(wxk)
        else:
            if self._has_data_x[k, yi, xi + 1]:
                # only x high has data.
                # set focus.
                wxk = self._wx[k]
                qc.x(wxk)
                # do the upper half
                self._build_area_y(xi + 1, yi * 2, k, wxk, wyk)
                # reset focus
                qc.x(wxk)
            else:
                # x low and x high both have no data.
                # does not come here.
                print("Error: no data for x low half")

    def _build_area_x(self, xi, yi, k, wxk: QuantumRegister, wyk: QuantumRegister):

        qc = self.circuit

        # In this method we inspect the data occupancy along the x axis
        # and see if the first half and second half is empty or not.
        # If both halves have data, the standard sequence is followed.
        # If only half is occupied, a shortcut sequence is followed.
        # For cases where both halves are empty, this function will
        # not be called.

        if self._has_data_x[k, yi, xi]:
            if self._has_data_x[k, yi, xi + 1]:
                # both x low and x high have data
                # focus on lower half.
                wxk = self._wx[k]
                qc.ccx(wyk, self._x[k], wxk, ctrl_state="01")
                if k == 0:
                    # optimized bottom func
                    self._build_area_y_bottom_dual(xi, yi*2, wxk, wyk)
                else:
                    # do the lower half
                    self._build_area_y(xi, yi * 2, k, wxk, wyk)
                    # middle.
                    # switch focus to upper half
                    qc.cx(wyk, wxk)
                    # do the upper half
                    self._build_area_y(xi + 1, yi * 2, k, wxk, wyk)
                # reset focus
                qc.ccx(wyk, self._x[k], wxk, ctrl_state="11")
            else:
                # only x low has data
                # focus on lower half.
                wxk = wyk
                # do the lower half
                self._build_area_y(xi, yi * 2, k, wxk, wyk)
                # reset focus
        else:
            if self._has_data_x[k, yi, xi + 1]:
                # only x high has data.
                # set focus.
                wxk = wyk
                # do the upper half
                self._build_area_y(xi + 1, yi * 2, k, wxk, wyk)
                # reset focus
            else:
                # x low and x high both have no data.
                # does not come here.
                print("Error: no data for x low half")

    def _build_area_y(self, xi, yi, k, wxk: QuantumRegister, wyk: QuantumRegister):
        qc = self.circuit
        if k == 0:
            self._build_area_y_bottom(xi, yi, wxk, wyk)
            return

        if self._has_data_y[k, yi, xi]:
            if self._has_data_y[k, yi + 1, xi]:
                # both y low and y high have data
                wyk = self._wy[k]
                qc.ccx(wxk, self._y[k], wyk, ctrl_state="01")
                self._build_area_x(xi * 2, yi, k - 1, wxk, wyk)
                qc.cx(wxk, wyk)
                self._build_area_x(xi * 2, yi + 1, k - 1, wxk, wyk)
                qc.ccx(wxk, self._y[k], wyk, ctrl_state="11")
            else:
                # only y low has data
                wyk = wxk
                self._build_area_x(xi * 2, yi, k - 1, wxk, wyk)
        else:
            if self._has_data_y[k, yi + 1, xi]:
                # only y high has data
                wyk = wxk
                self._build_area_x(xi * 2, yi + 1, k - 1, wxk, wyk)
            else:
                # y low and y high both have no data.
                # does not come here.
                print("Error: no data for x low half")

    def _build_area_y_bottom(self, xi, yi, wxk: QuantumRegister, wyk: QuantumRegister):
        qc = self.circuit

        v0 = self._data[yi, xi]
        v1 = self._data[yi + 1, xi]

        if v0 != 0.0:
            qc.x(self._y[0])
            qc.cp(v0, wxk, self._y[0])
            qc.x(self._y[0])

        if v1 != 0.0:
            qc.cp(v1, wxk, self._y[0])


    def _build_area_y_bottom_dual(self, xi, yi, wxk: QuantumRegister, wyk: QuantumRegister):
        qc = self.circuit
        v0 = self._data[yi, xi]
        v1 = self._data[yi + 1, xi]
        v2 = self._data[yi, xi + 1]
        v3 = self._data[yi + 1, xi + 1]

        # later bit comes first
        if v1 != 0.0:
            qc.cp(v1, wxk, self._y[0])

        if v0 != 0.0:
            if v2 != 0.0:
                # has 0, has 2
                qc.x(self._y[0])
                qc.cp(v0, wxk, self._y[0])
                # cancel qc.x(self._y[0])

                qc.cx(wyk, wxk)

                # cancel qc.x(self._y[0])
                qc.cp(v2, wxk, self._y[0])
                qc.x(self._y[0])
            else:
                # has 0, no 2
                qc.x(self._y[0])
                qc.cp(v0, wxk, self._y[0])
                qc.x(self._y[0])

                qc.cx(wyk, wxk)
        else:
            # no 0
            if v2 != 0.0:
                #  no 0, has 2
                qc.cx(wyk, wxk)

                qc.x(self._y[0])
                qc.cp(v2, wxk, self._y[0])
                qc.x(self._y[0])
            else:
                #  no 0, no 2
                pass
        if v3 != 0.0:
            qc.cp(v3, wxk, self._y[0])

    def bind(self, x: QuantumRegister, y: QuantumRegister):
        return Binding(self, {"x": x, "y": y})


class RadialFuncQromTestBoard(Frame):
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
        self.add_param(self._x, self._y)

    def build_circuit(self):
        qc = self.circuit
        qc.h(self._x)
        qc.h(self._y)
        self._rfq = RadialFuncQrom(
            self._n,
            self._dq,
            self._rfunc,
            use_symmetry=self._use_symmetry,
            use_transpose=self._use_transpose,
            verbose=self._verbose,
        )
        self.invoke(self._rfq.bind(x=self._x, y=self._y), invoke_as_instruction=True)

    @property
    def regs(self):
        return self._rfq.regs
