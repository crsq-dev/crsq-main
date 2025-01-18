""" Sparse statevector class
    A vector representation by an array of keys and an array of values.
"""

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import cupy as cp
import numpy as np
import numpy.typing as npt
import math

class SparseStatevector:
    """ Sparse statevector class

        To convert from a qiskit Statevector, use sv_to_sparse(sv).
    """
    def __init__(self, num_bits: int, keys: npt.NDArray[np.int64], values: npt.NDArray[np.complex128]):
        self.num_bits: int = num_bits
        self.keys: npt.NDArray[np.int64] = keys
        self.values: npt.NDArray[np.complex128] = values

def sv_to_sparse(sv: Statevector, eps=1.0e-8) -> SparseStatevector:
    """ Convert a Statevector to a SparseStateVector
    """
    keys = []
    values = []
    for i, v in enumerate(sv.data):
        if math.isnan(v.real) or math.isnan(v.imag):
            raise ValueError(f"NaN detected at index {i} in sv_to_sparse")
        if abs(v) > eps:
            keys.append(i)
            values.append(v)
    return SparseStatevector(sv.num_qubits, np.array(keys, dtype=np.int64), np.array(values, dtype=np.complex128))

def sparse_to_sv(ssv: SparseStatevector) -> Statevector:
    """ Convert a SparseStateVector to a Statevector
    """
    data = np.zeros(2**ssv.num_bits, dtype=cp.complex128)
    data[ssv.keys] = ssv.values
    return Statevector(data)

def save_to_file(path: str, ssv: SparseStatevector):
    """ Save components of a statevector to a file
    """
    if not isinstance(ssv, SparseStatevector):
        raise ValueError("ssv must be a SparseStatevector")
    with open(path, "w", encoding="utf-8") as f:
        d = ssv.num_bits
        f.write(f"{d}\n")
        n = len(ssv.keys)
        f.write(f"{n}\n")
        for j in range(len(ssv.keys)):
            i = ssv.keys[j]
            z: np.complex128 = ssv.values[j]
            key = bin((1<<d) + i)[-d:]
            f.write(f"{key},{z.real},{z.imag}\n")


def read_from_file(path: str) -> SparseStatevector:
    """ Read a statevector from a file created by save_to_file.
    """
    with open(path, "r", encoding="utf-8") as f:
        d = int(f.readline())
        n = int(f.readline())
        keys = np.zeros(n, dtype=np.int64)
        values = np.zeros(n, dtype=np.complex128)
        for i, line in enumerate(f):
            cols = line.split(',')
            key = cols[0]
            re = float(cols[1])
            im = float(cols[2])
            k = int(key, base=2)
            z = re + im * 1j
            keys[i] = k
            values[i] = z
        ssv = SparseStatevector(n, keys, values)
    return ssv


def extract_dist2_sub(ssv: SparseStatevector, xfrom: int, xto: int,
                      yfrom: int, yto: int,
                      eps:float = 1.0e-10) -> np.ndarray:
    """ Extract distribution indexed by a given bit range spec.
        returns data[x, y]
    """
    x_bitcount = xto - xfrom
    num_xdata = 2**x_bitcount
    x_bitmask = (num_xdata - 1) << xfrom
    y_bitcount = yto - yfrom
    num_ydata = 2**y_bitcount
    y_bitmask = (num_ydata - 1) << yfrom
    dists = np.zeros((num_xdata, num_ydata), dtype=np.complex128)
    for i, k in enumerate(ssv.keys):
        z = ssv.values[i]
        x_index = (k & x_bitmask) >> xfrom
        y_index = (k & y_bitmask) >> yfrom
        dists[x_index, y_index] = dists[x_index, y_index] + z
        # if abs(z) > 0.001:
        #     print(f"dists[{x_index}][{y_index}]+={z} => {dists[x_index,y_index]}")
    return dists

def extract_dist2d(qc: QuantumCircuit, ssv: SparseStatevector, xreg: str, yreg: str, eps=1.0e-12) -> np.ndarray:
    """ make a 2-d array indexed by xreg, yreg
    """
    reg_map = {}
    acc = 0
    for reg in qc.qregs:
        reg_map[reg.name] = (acc, acc + reg.size)
        acc += reg.size
    xb = reg_map[xreg]
    yb = reg_map[yreg]
    return extract_dist2_sub(ssv, xb[0], xb[1], yb[0], yb[1], eps)
