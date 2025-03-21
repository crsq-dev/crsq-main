"""utility funcs for circuits"""

import re
from qiskit import QuantumCircuit


def decompose_circuit_to_privmities(
    circuit: QuantumCircuit, gates_to_decompose: list[str]
) -> QuantumCircuit:
    """decompose circuit to primitives
    Args:
        circuit: QuantumCircuit
        gates_to_decompose: gate names to decompose
    Returns:
        QuantumCircuit: decomposed circuit
    """
    gate_pattern = "|".join(gates_to_decompose)
    while True:
        ops = circuit.count_ops()
        cnames = []
        for op in ops:
            # print(f"op: [{op}]")
            m = re.search(gate_pattern, op)
            if m:
                cnames.append(op)
        if len(cnames) == 0:
            return circuit
        # print(f"decompose with {cnames}")
        qc = circuit.decompose(cnames)
        circuit = qc


def merge_controled_gates(ops):
    """merge controlled gates cx and ccx with suffixes like _o0 or _o1 into non suffixed ones.
    Args:
        ops: dict of operations
    Returns:
        dict: merged operations
    """
    odict = {}
    for op in ops:
        m = re.search(r"([a-z]+)_(o\d+)", op)
        if m:
            op0 = m.group(1)
            if op0 not in odict:
                odict[op0] = ops[op]
            else:
                odict[op0] += ops[op]
        else:
            odict[op] = ops[op]
    return odict
