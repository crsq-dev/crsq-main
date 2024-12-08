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
LOG_TIME_THRESH=1

class RadialFuncQrom(Frame):
    """ 2D- Radial function implemented as QROM

    Args:
        n: bits per dimension
        rfunc: function of the form rfunc(r: float) -> float
    """
    def __init__(self, n: int, dq: float, rfunc: callable, build = True, verbose=False):
        super().__init__(label="RadialFuncQROM")
        logger.info("start: RadialFuncQrom()")
        t1 = time.time()
        self._n = n
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
            logger.info("end  : RadialFuncQrom() %f msec", round(dt*1000))
    
    def _prepare_data(self):
        # for a n bit x, abs(x) results as values 0 to 2^(n-1)
        # The values required is for 0 to 2^(n-1), which is 2^(n-1)+1 values.
        # We prepare a table sized 2^n, and use the range [0,0] to [2^(n-1), 2^(n-1)].
        n = self._n
        M = 2**n
        HM = M //2
        self._data = np.ndarray((M, M), dtype=float)
        self._has_data_y = np.ndarray((n, M, M), dtype=int)
        self._has_data_x = np.ndarray((n, M, M), dtype=int)
        for j in range(M):
            y = j * self._dq
            for i in range(M):
                if j <= i and i < HM + 1:
                    x = i * self._dq
                    r = math.sqrt(x*x + y*y)
                    psi = self._rfunc(r + self._dq/2)
                    if abs(psi) > math.pi:
                        logger.warning("x=%f, y=%f, r=%f, psi=%f", x, y, r, psi) 
                    self._data[i, j] = psi
                    self._has_data_y[0, i, j] = psi != 0.0
                else:
                    self._has_data_y[0, i, j] = 0
        w = M
        for k in range(0, n-1):
            for j in range(0, w//2):
                for i in range(0, w):
                    self._has_data_x[k, i, j] = self._has_data_y[k, i, 2*j] or self._has_data_y[k, i, 2*j+1]
            for j in range(0, w//2):
                for i in range(0, w//2):
                    self._has_data_y[k+1, i, j] = self._has_data_x[k, 2*i, j] or self._has_data_x[k, 2*i+1, j]
            w = w//2
        k = n-1
        for j in range(0, w//2):
            for i in range(0, w):
                self._has_data_x[k, i, j] = self._has_data_y[k, i, 2*j] or self._has_data_y[k, i, 2*j+1]
        w = M
        if self._verbose:
            self._print_tables()

    def _print_tables(self):
        n = self._n
        M = 2**n
        w = M
        for k in range(n):
            print(f"has_data_y[{k}]")
            for j in range(w):
                print(self._has_data_y[k, :w, j])
            print(f"has_data_x[{k}]")
            for j in range(w//2):
                print(self._has_data_x[k, :w, j])
            w = w//2


    def allocate_registers(self):
        """ allocate """
        n = self._n
        self._x = QuantumRegister(n, "x")
        self._y = QuantumRegister(n, "y")
        self.add_param(self._x, self._y)

        # work for qrom
        self._wx = QuantumRegister(n, "wx")
        self._wy = QuantumRegister(n, "wy")

        # sign
        self._sx = QuantumRegister(1, "sx")
        self._sy = QuantumRegister(1, "sy")
        # compare result
        self._cz = QuantumRegister(1, "cz")
        # carry for abs
        self._cr = QuantumRegister(n-1, "cr")

        self.add_local(self._wx, self._wy, self._sx, self._sy, self._cz, self._cr)

        regs = []
        for i in range(n-1, -1, -1):
            regs += [self._x[i], self._wx[i], self._y[i], self._wy[i]]
        regs += [self._sx[0], self._sy[0], self._cz[0]] + self._cr[:]
        reg_index = [self.circuit.qubits.index(reg) for reg in regs]
        self._regs = reg_index
    
    @property
    def regs(self):
        return self._regs

    def build_circuit(self):
        """ build """
        qc = self.circuit
        n = self._n
        qc.append(ari.absolute_gate(n), self._x[:] + self._sx[:] + self._cr[:])
        qc.append(ari.absolute_gate(n), self._y[:] + self._sy[:] + self._cr[:])
        qc.append(ari.cdk_comparator_gate(n), self._y[:] + self._x[:] + self._cz[:] + self._cr[0:1])
        for i in range(n):
            qc.cswap(self._cz[0], self._x[i], self._y[i])
        qc.barrier()

        xi = 0
        yi = 0
        k = n - 1
        self._build_area(xi, yi, k, None, None)

        qc.barrier()
        for i in range(n-1,-1,-1):
            qc.cswap(self._cz[0], self._x[i], self._y[i])

        cmp_dag_gate = ari.cdk_comparator_gate(n).inverse()
        cmp_dag_gate.label = f"cmp\u2020({n})"
        qc.append(cmp_dag_gate, self._y[:] + self._x[:] + self._cz[:] + self._cr[0:1])

        absy_dag_gate = ari.absolute_gate(n).inverse()
        absy_dag_gate.label = f"abs\u2020({n})"
        qc.append(absy_dag_gate, self._y[:] + self._sy[:] + self._cr[:])

        absx_dag_gate = ari.absolute_gate(n).inverse()
        absx_dag_gate.label = f"abs\u2020({n})"
        qc.append(absx_dag_gate, self._x[:] + self._sx[:] + self._cr[:])

    def _build_area(self, xi, yi, k, wxk:QuantumRegister, wyk:QuantumRegister):

        qc = self.circuit
        n = self._n

        # In this method we inspect the data occupancy along the x axis
        # and see if the first half and second half is empty or not.
        # If both halves have data, the standard sequence is followed.
        # If only half is occupied, a shortcut sequence is followed.
        # For cases where both halves are empty, this function will
        # not be called.

        if self._has_data_x[k, xi*2, yi]:
            if self._has_data_x[k, xi*2+1, yi]:
                # both x low and x high have data
                # focus on lower half.
                if k == n - 1:
                    qc.cx(self._x[k], self._wx[k], ctrl_state="0")
                    wxk = self._wx[k]
                else:
                    qc.ccx(wyk, self._x[k], self._wx[k], ctrl_state="01")
                    wxk = self._wx[k]
                # do the lower half
                self._build_area_x_low(xi,yi,k,wxk,wyk)
                # middle.
                # switch focus to upper half
                if k == n - 1:
                    qc.x(self._wx[k])
                else:
                    qc.cx(wyk, self._wx[k])
                # do the upper half
                self._build_area_x_high(xi,yi,k,wxk,wyk)
                # reset focus
                if k == n - 1:
                    qc.cx(self._x[k], self._wx[k], ctrl_state="1")
                else:
                    qc.ccx(wyk, self._x[k], self._wx[k], ctrl_state="11")
            else:
                # only x low has data
                # focus on lower half.
                if k == n - 1:
                    qc.x(self._wx[k])
                    wxk = self._wx[k]
                else:
                    wxk = wyk
                # do the lower half
                self._build_area_x_low(xi,yi,k,wxk,wyk)
                # reset focus
                if k == n - 1:
                    qc.x(self._wx[k])
                else:
                    pass
        else:
            if self._has_data_x[k, xi*2+1, yi]:
                # only x high has data.
                # set focus.
                if k == n - 1:
                    qc.x(self._wx[k])
                    wxk = self._wx[k]
                else:
                    wxk = wyk
                # do the upper half
                self._build_area_x_high(xi,yi,k,wxk,wyk)
                # reset focus
                if k == n - 1:
                    qc.x(self._wx[k])
                else:
                    pass
            else:
                # x low and x high both have no data.
                # does not come here.
                print("Error: no data for x low half")

    
    def _build_area_x_low(self, xi, yi, k, wxk:QuantumRegister, wyk:QuantumRegister):
        qc = self.circuit
        if k > 0:
            if self._has_data_y[k, xi*2, yi*2]:
                if self._has_data_y[k, xi*2, yi*2+1]:
                    # both y low and y high have data
                    qc.ccx(wxk, self._y[k], self._wy[k], ctrl_state='01')
                    wyk = self._wy[k]
                    self._build_area(xi*2, yi*2, k-1, wxk, wyk)
                    qc.cx(wxk, self._wy[k])
                    self._build_area(xi*2, yi*2+1, k-1, wxk, wyk)
                    qc.ccx(wxk, self._y[k], self._wy[k], ctrl_state='11')
                else:
                    # only y low has data
                    wyk = wxk
                    self._build_area(xi*2, yi*2, k-1, wxk, wyk)
            else:
                if self._has_data_y[k, xi*2, yi*2+1]:
                    # only y high has data
                    wyk = wxk
                    self._build_area(xi*2, yi*2+1, k-1, wxk, wyk)
                else:
                    # y low and y high both have no data.
                    # does not come here.
                    print("Error: no data for x low half")
        else:
            # later bit comes first
            v = self._data[xi*2, yi*2+1]
            if v != 0.0:
                qc.cp(v, wxk, self._y[0])

            # earlier bit comes second
            v = self._data[xi*2, yi*2]
            # cancels with high half:
            if self._has_data_x[k, xi*2+1, yi]:
                qc.x(self._y[0])
                if v != 0.0:
                    qc.cp(v, wxk, self._y[0])
                    pass
            else:
                if v != 0.0:
                    qc.x(self._y[0])
                    qc.cp(v, wxk, self._y[0])
                    qc.x(self._y[0])

    def _build_area_x_high(self, xi, yi, k, wxk:QuantumRegister, wyk:QuantumRegister):
        qc = self.circuit
        if k > 0:
            if self._has_data_y[k, xi*2+1, yi*2]:
                if self._has_data_y[k, xi*2+1, yi*2+1]:
                    # both y low and y high have data
                    qc.ccx(wxk, self._y[k], self._wy[k], ctrl_state='01')
                    wyk = self._wy[k]
                    self._build_area(xi*2+1, yi*2, k-1, wxk, wyk)
                    qc.cx(wxk, self._wy[k])
                    self._build_area(xi*2+1, yi*2+1, k-1, wxk, wyk)
                    qc.ccx(wxk, self._y[k], self._wy[k], ctrl_state='11')
                else:
                    # only y low has data
                    wyk = wxk
                    self._build_area(xi*2+1, yi*2, k-1, wxk, wyk)
            else:
                if self._has_data_y[k, xi*2+1, yi*2+1]:
                    # only y high has data
                    wyk = wxk
                    self._build_area(xi*2+1, yi*2+1, k-1)
                else:
                    # y low and y high both have no data.
                    # does not come here.
                    print("Error: no data for x high half")
        else:
            v = self._data[xi*2+1, yi*2]
            # cancels with low half
            if self._has_data_x[k, xi*2, yi]:
                if v != 0.0:
                    qc.cp(v, wxk, self._y[0])
                    pass
                qc.x(self._y[0])
            else:
                if v != 0.0:
                    qc.x(self._y[0])
                    qc.cp(v, wxk, self._y[0])
                    qc.x(self._y[0])

            v = self._data[xi*2+1, yi*2+1]
            if v != 0.0:
                qc.cp(v, self._wx[0], self._y[0])

    def bind(self, x:QuantumRegister, y:QuantumRegister):
        return Binding(self, {"x": x, "y": y})


class RadialFuncQromTestBoard(Frame):
    def __init__(self, n: int, dq: float, rfunc: callable, verbose=True):
        super().__init__(label="RFQTest")
        self._n = n
        self._dq = dq
        self._rfunc = rfunc
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
        self._rfq = RadialFuncQrom(self._n, self._dq, self._rfunc, verbose=self._verbose)
        self.invoke(self._rfq.bind(x=self._x, y=self._y))

    @property
    def regs(self):
        return self._rfq.regs
