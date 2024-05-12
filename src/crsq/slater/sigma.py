""" Construct a circuit to create a superposition of integers 0 to N
"""
import numpy as np
from qiskit.circuit import Gate, QuantumCircuit, QuantumRegister, Qubit
from qiskit.circuit.library import HGate, RYGate

def _find_msb(N: int) -> tuple[int, int, int]:
    """ find the MSB. return (n,t,num_bits) where 2**n = t and t is the msb
        and num_bits is the number of bits required to hold N-1

        N   n   t   num_bits
        1   0   1   1
        2   1   2   1
        3   1   2   2
        4   2   4   2
        5   2   4   3
        6   2   4   3
        7   2   4   3
        8   3   8   3
    """
    n = 0
    t = 1
    while t*2 <= N:
        n += 1
        t *= 2
    assert int(2**n) >= t
    if N == 1:
        num_bits = 1
    elif N == t:
        num_bits = n
    else:
        num_bits = n + 1

    return (n, t, num_bits)

def _fill_h(qc: QuantumCircuit, qr: QuantumRegister, n: int):
    """ put H on the lower n bits of qr"""
    for k in range(n):
        qc.h(qr[k])

def _fill_vch(qc: QuantumCircuit, qr: QuantumRegister, n: int, cbits: list[int]):
    """ put variable length controlled H gates on the lower n bits of qr.
        The control bits are given as a list of bit indices (cbits).
    """
    for k in range(n):
        num_cbits = len(cbits)
        ctrl_state = '1' * num_cbits
        qbits = cbits + [qr[k]]
        qc.append(HGate().control(num_cbits, ctrl_state=ctrl_state), qbits[:])

def _fill_nch(qc: QuantumCircuit, qr: QuantumRegister, n: int, ncbit: int):
    """ create negative controlled H gates on the lower n bits of qr.
    """
    for k in range(n):
        qc.ch(qr[ncbit], qr[k], ctrl_state=0)  # A negative controlled Hadamard.

def _fill_vcnch(qc: QuantumCircuit, qr: QuantumRegister,
                n: int, ncbit: int, cbits: list[Qubit]):
    """ create variable length negative controlled H gates on the lower n bits of qr.
        The control bits consists of positive set cbits and one negative bit ncbit
    """
    for k in range(n):
        qbits = cbits + [qr[ncbit], qr[k]]
        num_cbits = len(cbits)
        ctrl_state = '0' + '1' * num_cbits
        qc.append(HGate().control(num_cbits+1, ctrl_state=ctrl_state),
                  qbits[:])
        # qc.ch(cbit, k, ctrl_state=0)  # A negative controlled Hadamard.

def _build_nth_bit_gate(qc: QuantumCircuit, qr: QuantumRegister, n: int,
                        t: int, N: int):
    if N == t:
        _fill_h(qc, qr, n)
    else:
        a: float = np.sqrt(t/N)
        th = 2*np.arccos(a)
        # print(f"n={n} a={a} th={a}")
        qc.ry(th, qr[n])
        _fill_nch(qc, qr, n, n)

def _build_controlled_nth_bit_gate(qc: QuantumCircuit, qr: QuantumRegister,
                                   n: int, t: int, N: int,
                                   cbits: list[Qubit]):
    if N == t:
        _fill_vch(qc, qr, n, cbits)
    else:
        a: float = np.sqrt(t/N)
        th = 2*np.arccos(a)
        # print(f"n={n} a={a} th={a}")
        cbit_len = len(cbits)
        ctrl_state = '1' * cbit_len
        qbits = cbits + [qr[n]]
        qc.append(RYGate(th).control(cbit_len, ctrl_state=ctrl_state), qbits[:])
        # qc.cry(th, cbit, n)
        _fill_vcnch(qc, qr, n, n, cbits)

def _nth_bit_gate(n: int, t: int, N: int, num_bits: int) -> Gate:
    assert 2**n == t
    assert N >= t
    assert N < 2**(n+1)
    qr = QuantumRegister(num_bits)
    qc = QuantumCircuit(qr)
    _build_nth_bit_gate(qc, qr, n, t, N)
    g = qc.to_gate(label=f"Σ({N-1})")
    return g

def _build_lower_gates(qc: QuantumCircuit, qr: QuantumRegister,
                       n: int, t: int, N: int, cbits: list[Qubit],
                       use_custom_gates):
    """ using the first n bits in qr, build a circuit to produce
        sum[|i> for i in range(N)]/sqrt(N)
    """
    N1 = N - t
    if N1 == 1:
        return
    n1, t1, num_bits1 = _find_msb(N1)
    # the control bit will be placed at the LSB but we want it at the MSB.
    if use_custom_gates:
        qc.append(_nth_bit_gate(n1, t1, N1, num_bits1).control(
            len(cbits), label=f"c|{N1}>"), cbits + qr[:num_bits1])
    else:
        _build_controlled_nth_bit_gate(qc, qr, n1, t1, N1, cbits)
    if N1 > t1:
        _build_lower_gates(qc, qr, n1, t1, N1, [num_bits1-1] + cbits, use_custom_gates)

def build_sums(qc: QuantumCircuit, qr: QuantumRegister, N: int, use_custom_gates=True):
    """ Build a circuit to set up a sum{k} superposition state
    """
    if N < 1:
        raise ValueError("N must be > 0")
    if N == 1:
        return
    n, t, num_bits = _find_msb(N)
    if num_bits > qr.size:
        raise ValueError("N is too big")
    # top bit gate runs unconditionally
    if use_custom_gates:
        qc.append(_nth_bit_gate(n, t, N, num_bits), qr[:num_bits])
    else:
        _build_nth_bit_gate(qc, qr, n, t, N)
    # the remaining bits are conditional
    if n > 1 and N > t:
        _build_lower_gates(qc, qr, n, t, N, [qr[num_bits - 1]], use_custom_gates)
