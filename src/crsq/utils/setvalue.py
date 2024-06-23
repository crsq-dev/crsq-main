"""
    Set a value on a register.
"""

from qiskit import QuantumCircuit, QuantumRegister


def set_binary_value(circ: QuantumCircuit, qr: QuantumRegister, val: int):
    """ Decompose an integer value into bits and set them
        to a quantum register.

        :param circ: The circuit for the register.
        :param qr: The target register.
        :param val: Value to set to the register.
    """
    n = qr.size
    for k in range(n):
        mask = 1 << k
        if val & mask:
            circ.x(qr[k])

def set_unary_value(circ: QuantumCircuit, qr: QuantumRegister, val: int, padding=0):
    """ set at most one bit to represent val in unary.
        for val == 0, all bits will be zero.
        for val == 1, the LSB will be 1.
        for val == 2, the second LSB will be 1, preceding bits will be prefix.

        :param circ: The circuit for the register.
        :param qr: The target register.
        :param val: Value to set to the register.
    """
    assert val >= 0
    assert val <= qr.size
    if val == 0:
        return
    circ.x(qr[val-1])
    if padding == 0:
        return
    for k in range(val-1):
        circ.x(qr[k])
