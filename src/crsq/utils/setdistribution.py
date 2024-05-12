""" Set distribution
"""

import math
import cmath

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate

def SetdGate(N:int, c: np.array, csq = None) -> Gate:
    """ create a gate that sets c[k]/sum(c**2)) as amplitude for reg[k]
    """
    if N < 1:
        raise ValueError("N must be >= 1.")
    if 2**N != len(c):
        raise ValueError("2**N must match array size.")
    # make array of squares
    if csq is None:
        csq = [(ck.conjugate()*ck).real for ck in c]
    half = len(c) // 2
    c1 = c[:half]
    c2 = c[half:]
    csq1 = csq[:half]
    csq2 = csq[half:]
    s1 = np.sqrt(np.sum(csq1))
    s2 = np.sqrt(np.sum(csq2))
    if s1 == 0 and s2 == 0:
        raise ValueError("All elements of c[] cannot be zero.")
    qc = QuantumCircuit(N)
    target = N - 1
    if s2 == 0:
        pass # Ry(0)
    elif s1 == s2: # Ry(pi/2)
        qc.h(target)
    elif s1 == 0:
        qc.x(target) # Ry(pi)
    else:
        th = 2*math.atan2(s2, s1)
        qc.ry(th, target)

    if target > 0:
        if np.array_equal(c1, c2):
            regs = list(range(N-1))
            g = SetdGate(N-1, c1, csq1)
            qc.append(g, regs)
        else:
            regs = [target] + list(range(N-1))
            if s1 > 0:
                g1 = SetdGate(N-1, c1, csq1).control(1, ctrl_state = 0)
                qc.append(g1, regs)
            if s2 > 0:
                g2 = SetdGate(N-1, c2, csq2).control(1, ctrl_state = 1)
                qc.append(g2, regs)
    else:
        if c1[0] != 0:
            th1 = cmath.phase(c1[0])
            if th1 != 0:
                # rotate |0>
                qc.x(target)
                qc.p(th1, target)
                qc.x(target)
        if c2[0] != 0:
            th2 = cmath.phase(c2[0])
            if th2 != 0:
                # rotate |1>
                qc.p(th2, target)
    g = qc.to_gate(label=f"emb({N})")
    return g

def setdist(qc: QuantumCircuit, qr: QuantumRegister, c: np.array):
    """ Set c[k]/sum(c**2) as amplitudes to qr[k]
    """
    N = qr.size
    qc.append(SetdGate(N, c), [*qr])
