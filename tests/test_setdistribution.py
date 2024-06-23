""" test set_distribution
"""

import pytest
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
import crsq.utils.setdistribution as dist
import crsq_arithmetic.test_tools as test_tools


def do_setv(n, c: list) -> QuantumCircuit:
    """ Test driver for setv """
    qr = QuantumRegister(n, "R")
    qc = QuantumCircuit(qr)
    vals = np.array(c)
    dist.setdist(qc, qr, vals)
    qc.save_statevector()
    expected = {}
    for k, ck in enumerate(c):
        key = bin((1<<n) + k)[-n:]
        expected[key] = ck
    test_tools.run_circuit_and_check(qc, expected)
    return qc

def test_bad_N():
    """ array length does not match 2**N """
    c = [1, 1]
    with pytest.raises(ValueError) as e:
        do_setv(0, c)
    assert str(e.value) == "N must be >= 1."

def test_bad_length():
    """ array length does not match 2**N """
    c = [1, 1, 1]
    with pytest.raises(ValueError) as e:
        do_setv(1, c)
    assert str(e.value) == "2**N must match array size."

def test_1b_all_0():
    """ All zero for c[k] causes an exception. """
    c = [0, 0]
    with pytest.raises(ValueError) as e:
        do_setv(1, c)
    assert str(e.value) == "All elements of c[] cannot be zero."

def test_1b_all_1():
    """ All one for c[k] results in H on all bits. """
    p = 1 / np.sqrt(2)
    c = [p, p]
    do_setv(1, c)


def test_1b_10():
    """ All weight goes to |0>. """
    c = [1, 0]
    do_setv(1, c)


def test_1b_01():
    """ All weight goes to |1>. """
    c = [0, 1]
    do_setv(1, c)


def test_1b_arbitrary1():
    """ Test with 1 qubit """
    vals = [np.sqrt(1/3)*1j, np.sqrt(2/3)]
    do_setv(1, vals)


def test_1b_arbitrary2():
    """ Test with 1 qubit """
    vals = [np.sqrt(1/3), np.sqrt(2/3)*1j]
    do_setv(1, vals)

def test_2b_all_0():
    """ All zero for c[k] causes an exception. """
    c = [0, 0, 0, 0]
    with pytest.raises(ValueError) as e:
        do_setv(2, c)
    assert str(e.value) == "All elements of c[] cannot be zero."

def test_2b_all_1():
    """ All one for c[k] results in H on all bits. """
    p = 1/np.sqrt(4)
    c = [p, p, p, p]
    do_setv(2, c)

def test_2b_1100():
    """ Weight is only on first half. """
    p = 1/np.sqrt(2)
    c = [p, p, 0, 0]
    do_setv(2, c)

def test_2b_0011():
    """ Weight is only on second half. """
    p = 1/np.sqrt(2)
    c = [0, 0, p, p]
    do_setv(2, c)

def test_2b_1001():
    """ Weight is on both halves and is asymmetric."""
    p = 1/np.sqrt(2)
    c = [p, 0, 0, p]
    do_setv(2, c)

if __name__ == '__main__':
    test_bad_N()
    test_bad_length()
    test_1b_all_0()
    test_1b_all_1()
    test_1b_10()
    test_1b_01()
    test_1b_arbitrary1()
    test_1b_arbitrary2()
    test_2b_all_0()
    test_2b_all_1()
    test_2b_1100()
    test_2b_0011()
    test_2b_1001()
