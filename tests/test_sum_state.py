""" test for sum module """
import math
from qiskit.circuit import QuantumCircuit, QuantumRegister
import crsq.slater.sigma as ss
import crsq_arithmetic.test_tools as test_tools


def test_find_msb():
    """ _find_msb """
    assert ss._find_msb(1) == (0, 1, 1)
    assert ss._find_msb(2) == (1, 2, 1)
    assert ss._find_msb(3) == (1, 2, 2)
    assert ss._find_msb(4) == (2, 4, 2)
    assert ss._find_msb(5) == (2, 4, 3)
    assert ss._find_msb(6) == (2, 4, 3)
    assert ss._find_msb(7) == (2, 4, 3)
    for N in range(1, 16):
        (n, t, num_bits) = ss._find_msb(N)
        assert 2**n == t
        assert N >= t
        assert 2*N > t
        assert 2**(num_bits)-1 >= N-1


def _log2(x):
    s = 0
    t = 1
    while x > t:
        s += 1
        t *= 2
    return s


def test_build_sums():
    for N in range(0+1,32+1):
        nbits =  _log2(N-1) + 1
        expected = {}
        p = 1/math.sqrt(N)
        for k in range(N):
            key = bin(k+(1<<nbits))[-nbits:]
            expected[key] = p
        qr1 = QuantumRegister(nbits)
        qc1 = QuantumCircuit(qr1)
        ss.build_sums(qc1, qr1, N, False)
        qc1.save_statevector()
        test_tools.run_circuit_and_check(qc1, expected)
        qr2 = QuantumRegister(nbits)
        qc2 = QuantumCircuit(qr2)
        ss.build_sums(qc2, qr2, N, True)
        qc2.save_statevector()
        test_tools.run_circuit_and_check(qc2, expected)


if __name__ == '__main__':
    test_find_msb()
    test_build_sums()
