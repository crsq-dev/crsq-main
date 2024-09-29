""" test gates in qspace.py
"""

from qiskit import QuantumRegister
from crsq_heap import heap
from crsq_arithmetic import test_tools
from crsq.utils import setvalue
from crsq.blocks import wave_function
from crsq.blocks.time_evolution import qspace

def do_one_r(K: int, L: int, k: int, l: int):
    """ test R"""
    wfr_spec = wave_function.WaveFunctionRegisterSpec(1, 3, 2.0, 1, 1, 0)
    q_spec = qspace.QSpaceTEVSpec(wfr_spec, K, L)
    rb = qspace.RBlock(q_spec)
    frame = heap.Frame()
    areg = QuantumRegister(q_spec.k, "a")
    breg = QuantumRegister(q_spec.num_l_bits, "b")
    frame.add_local(areg, breg)
    qc = frame.circuit
    setvalue.set_unary_value(qc, areg, k, padding=1)
    setvalue.set_binary_value(qc, breg, l)
    frame.invoke(rb.bind(a=areg, b=breg))
    frame.circuit.save_statevector()
    a_bits = "0"*(K-k)+"1"*k
    b_bits = bin(l)[2:].zfill(q_spec.num_l_bits)
    expected: dict[str,complex] = {}
    if k == 0 and l == 0:
        p = -1.0
    else:
        p = 1.0
    expected[b_bits + a_bits] = p
    test_tools.run_circuit_and_check(qc, expected)

def test_r():
    """ Test R block
    """
    K = 2
    L = 3
    for k in range(K):
        for l in range(L):
            do_one_r(K,L,k,l)

if __name__ == '__main__':
    test_r()
