""" Statevector utilities
"""

import re
import math
import cmath
import numpy as np
from qiskit.circuit import Qubit, QuantumRegister, QuantumCircuit
from qiskit.quantum_info import Statevector
import crsq.utils.amplitudes as amp
import crsq_arithmetic.ast as ast
import crsq_heap.heap as heap

def save_to_file(path: str, sv: Statevector, eps=1.0e-10):
    """ Save components of a statevector to a file
    """
    with open(path, "w", encoding="utf-8") as f:
        d = int(math.log2(sv.dim))
        f.write(f"{d}\n")
        for i, z in enumerate(sv.data):
            if abs(z) > eps:
                key = bin((1<<d) + i)[-d:]
                f.write(f"{key},{z.real},{z.imag}\n")

def read_from_file(path: str) -> Statevector:
    """ Read a statevector from a file created by save_to_file.
    """
    with open(path, "r", encoding="utf-8") as f:
        d = int(f.readline())
        data = np.zeros(2**d, dtype=np.complex128)
        for line in f:
            cols = line.split(',')
            key = cols[0]
            re = float(cols[1])
            im = float(cols[2])
            k = int(key, base=2)
            z = re + im * 1j
            data[k] = z
        sv = Statevector(data)
    return sv


def extract_grouped_dist(sv: Statevector, group_from: int, group_to: int,
                 data_from: int, data_to: int, eps:float = 1.0e-10) -> np.ndarray:
    """ Extract distribution indexed by a given bit range spec.
    """
    group_bitcount = group_to - group_from
    data_bitcount = data_to - data_from
    num_groups = 2**group_bitcount
    num_data = 2**data_bitcount
    groups_bitmask = (num_groups - 1) << group_from
    data_bitmask = (num_data - 1) << data_from
    dists = np.zeros((num_groups, num_data), dtype=np.complex128)
    for k, z in enumerate(sv.data):
        group_index = (k & groups_bitmask) >> group_from
        data_index = (k & data_bitmask) >> data_from
        dists[group_index, data_index] += z
        if z > 0.1:
            print(f"dists[{group_index},{data_index}]+={z}")
    return dists


def extract_grouped_dist2(qc: QuantumCircuit, sv: Statevector, group_reg: str, data_reg: str, eps=1.0e-12) -> np.ndarray:
    """ make a 2-d array indexed by group_reg, data_reg
    """
    reg_map = {}
    acc = 0
    for reg in qc.qregs:
        reg_map[reg.name] = (acc, acc + reg.size)
        acc += reg.size
    greg = reg_map[group_reg]
    dreg = reg_map[data_reg]
    return extract_grouped_dist(sv, greg[0], greg[1], dreg[0], dreg[1], eps)


def extract_dist_sub(sv: Statevector, data_from: int, data_to: int,
                     eps:float = 1.0e-10) -> np.ndarray:
    """ Extract distribution indexed by a given bit range spec.
    """
    data_bitcount = data_to - data_from
    num_data = 2**data_bitcount
    data_bitmask = (num_data - 1) << data_from
    dists = np.zeros((num_data), dtype=np.complex128)
    for k, z in enumerate(sv.data):
        data_index = (k & data_bitmask) >> data_from
        dists[data_index] += z
        if z > 0.1:
            print(f"dists[{data_index}]+={z}")
    return dists

def extract_dist2_sub(sv: Statevector, xfrom: int, xto: int,
                      yfrom: int, yto: int,
                      eps:float = 1.0e-10) -> np.ndarray:
    """ Extract distribution indexed by a given bit range spec.
    """
    x_bitcount = xto - xfrom
    num_xdata = 2**x_bitcount
    x_bitmask = (num_xdata - 1) << xfrom
    y_bitcount = yto - yfrom
    num_ydata = 2**y_bitcount
    y_bitmask = (num_ydata - 1) << yfrom
    dists = np.zeros((num_xdata,num_ydata), dtype=np.complex128)
    for k, z in enumerate(sv.data):
        x_index = (k & x_bitmask) >> xfrom
        y_index = (k & y_bitmask) >> yfrom
        dists[x_index, y_index] = dists[x_index, y_index] + z
        if abs(z) > 0.001:
            print(f"dists[{x_index}][{y_index}]+={z} => {dists[x_index,y_index]}")
    return dists

def extract_dist(qc: QuantumCircuit, sv: Statevector, data_reg: str, eps=1.0e-12) -> np.ndarray:
    """ make a 2-d array indexed by group_reg, data_reg
    """
    reg_map = {}
    acc = 0
    for reg in qc.qregs:
        reg_map[reg.name] = (acc, acc + reg.size)
        acc += reg.size
    dreg = reg_map[data_reg]
    return extract_dist_sub(sv, dreg[0], dreg[1], eps)

def extract_dist2d(qc: QuantumCircuit, sv: Statevector, xreg: str, yreg: str, eps=1.0e-12) -> np.ndarray:
    """ make a 2-d array indexed by group_reg, data_reg
    """
    reg_map = {}
    acc = 0
    for reg in qc.qregs:
        reg_map[reg.name] = (acc, acc + reg.size)
        acc += reg.size
    xb = reg_map[xreg]
    yb = reg_map[yreg]
    return extract_dist2_sub(sv, xb[0], xb[1], yb[0], yb[1], eps)

def _extract_int(x: int, bits: tuple[int], signed: bool, frac_bits=0) -> int:
    s: int = 0
    weight = 1
    for k in bits:
        mask = 1 << k
        if x & mask != 0:
            s += weight
        weight <<= 1
    if signed:
        m = len(bits)
        HM = 1 << (m-1)
        if s >= HM:
            s -= (HM << 1)
    if frac_bits > 0:
        s *= 2**(-frac_bits)
    return s

def _find_bit_pos_in_circuit(qc: QuantumCircuit | heap.Frame, b: Qubit) -> int:
    if isinstance(qc, heap.Frame):
        frame = qc
        qc = frame._circuit
    else:
        frame = None
    pos = None
    for i, bit in enumerate(qc.qubits):
        if bit == b:
            pos = i
            break
    if pos is None:
        raise ValueError(f"bit {b} not found in circuit.")
    if frame is None or frame.parent is None:
        return pos
    parent_reg_set = frame.parent
    qubit_in_parent = frame._args_in_parent[pos]
    return _find_bit_pos_in_circuit(parent_reg_set, qubit_in_parent)


def _get_bits_from_register(
        c: QuantumCircuit | heap.Frame,
        bit_list_spec: tuple[int] | QuantumRegister | ast.QuantumValue):
    if isinstance(bit_list_spec, tuple):
        return bit_list_spec
    elif isinstance(bit_list_spec, QuantumRegister):
        bit_list = list(bit_list_spec[:])
        return tuple(_find_bit_pos_in_circuit(c, b) for b in bit_list)
    elif isinstance(bit_list_spec, ast.QuantumValue):
        bit_list = list(bit_list_spec.register[:])
        return tuple(_find_bit_pos_in_circuit(c, b) for b in bit_list)


def _get_signed_flag_from_bit_spec(bit_spec: tuple[int] | QuantumRegister) -> bool:
    if isinstance(bit_spec, ast.QuantumValue):
        qv: ast.QuantumValue = bit_spec
        return qv.signed
    else:
        return False


def extract_arithmetic_result_1d_scatter(
        qc: QuantumCircuit | heap.Frame, sv: Statevector,
        in_bit_spec: tuple[int] | QuantumRegister | ast.QuantumValue,
        out_bit_spec: tuple[int] | QuantumRegister | ast.QuantumValue,
        eps=1.0e-10, print_states: bool=False):
    """
        get array mapping in value to out value from a statevector
        :param qc: the quantum circuit
        :param sv: the statevector
        :param in_bits: LSB first bit position list for the input value
        :param out_bits: LSB first bit position list for the output value
    """
    n = sv.num_qubits
    bx = _get_bits_from_register(qc, in_bit_spec)
    by = _get_bits_from_register(qc, out_bit_spec)
    xsigned = _get_signed_flag_from_bit_spec(in_bit_spec)
    ysigned = _get_signed_flag_from_bit_spec(out_bit_spec)
    if isinstance(in_bit_spec, ast.QuantumValue):
        x_frac_bits = in_bit_spec.fraction_bits
    else:
        x_frac_bits = 0
    if isinstance(out_bit_spec, ast.QuantumValue):
        y_frac_bits = out_bit_spec.fraction_bits
    else:
        y_frac_bits = 0
    nbx = len(bx)
    nby = len(by)
    x = []
    y = []
    for k, z in enumerate(sv.data):
        if abs(z) > eps:
            in_index = _extract_int(k, bx, xsigned)
            out_index = _extract_int(k, by, ysigned)
            in_val = in_index * (2**-x_frac_bits)
            out_val = out_index * (2**-y_frac_bits)
            x.append(in_val)
            y.append(out_val)
            if print_states:
                print(bin((1<<n) + k)[-n:],
                    bin((1<<nbx)+in_index)[-nbx:],
                    bin((1<<nby)+out_index)[-nby:])
    return (x, y)


def extract_arithmetic_result_1d_scatter2(
        rs: heap.Frame, sv: Statevector,
        in_bit_spec: ast.QuantumValue,
        out_bit_spec: ast.QuantumValue,
        eps=1.0e-10):
    """
        get array mapping in value to out value from a statevector
        :param qc: the quantum circuit
        :param sv: the statevector
        :param in_bits: LSB first bit position list for the input value
        :param out_bits: LSB first bit position list for the output value
    """
    n = sv.num_qubits
    bx = _get_bits_from_register(rs, in_bit_spec)
    by = _get_bits_from_register(rs, out_bit_spec)
    xsigned = _get_signed_flag_from_bit_spec(in_bit_spec)
    ysigned = _get_signed_flag_from_bit_spec(out_bit_spec)
    x_frac_bits = in_bit_spec.fraction_bits
    y_frac_bits = out_bit_spec.fraction_bits
    nbx = len(bx)
    nby = len(by)
    x = []
    y = []
    for k, z in enumerate(sv.data):
        if abs(z) > eps:
            in_index = _extract_int(k, bx, xsigned, x_frac_bits)
            out_index = _extract_int(k, by, ysigned, y_frac_bits)
            in_val = in_index * (2**-x_frac_bits)
            out_val = out_index * (2**-y_frac_bits)
            x.append(in_val)
            y.append(out_val)
            print(bin((1<<n) + k)[-n:],
                  bin((1<<nbx)+in_index)[-nbx:], bin((1<<nby)+out_index)[-nby:])
    return (x, y)

def extract_arithmetic_result_2d_bars(
        rs: heap.Frame, sv: Statevector,
        a_bit_spec: ast.QuantumValue,
        b_bit_spec: ast.QuantumValue,
        c_bit_spec: ast.QuantumValue,
        eps=1.0e-10, print_states: bool=False):
    """ extract (a,b) -> c style data for a bar3d plot

        :param rs: the RegisterSet where the state vector belongs
        :param sv: the state vector to analyze
        :param a_bit_spec: the AST node for a
        :param b_bit_spec: the AST node for b
        :param c_bit_spec: the AST node for c
        :return: (x, y, z, dz)
    """
    n = sv.num_qubits
    ba = _get_bits_from_register(rs, a_bit_spec)
    bb = _get_bits_from_register(rs, b_bit_spec)
    bc = _get_bits_from_register(rs, c_bit_spec)
    asigned = _get_signed_flag_from_bit_spec(a_bit_spec)
    bsigned = _get_signed_flag_from_bit_spec(b_bit_spec)
    csigned = _get_signed_flag_from_bit_spec(c_bit_spec)
    nba = len(ba)
    nbb = len(bb)
    nbc = len(bc)
    if csigned:
        bottom = -(1 << nbc)/2
    else:
        bottom = 0
    a = []
    b = []
    c = []
    dz = []
    for k, z in enumerate(sv.data):
        if abs(z) > eps:
            a_index = _extract_int(k, ba, asigned)
            b_index = _extract_int(k, bb, bsigned)
            c_index = _extract_int(k, bc, csigned)
            a.append(a_index)
            b.append(b_index)
            c.append(bottom)
            dz.append(c_index-bottom)
            if print_states:
                print(bin((1<<n) + k)[-n:], bin((1<<nba)+a_index)[-nba:],
                    bin((1<<nbb)+b_index)[-nbb:], bin((1<<nbc)+c_index)[-nbc:])
    ra = np.array(a)
    rb = np.array(b)
    rc = np.array(c)
    rdz = np.array(dz)
    return (ra, rb, rc, rdz)


def dump_statevector(sv: Statevector, qc: QuantumCircuit, eps:float = 1e-12, global_phase=0):
    """ Dump a state vector
    """
    gp = cmath.exp(global_phase*1J)
    fr = []
    to = []
    wid = []
    lab = []
    pos = 0
    qregs = []
    tmp = []
    for r in qc.qregs:
        if re.match(r'tmp\d', r.name):
            tmp.append(r)
        elif r.size > 0:
            qregs.append(r)
    if len(tmp) > 0:
        tmp_bits = len(tmp)
        qregs.append(QuantumRegister(tmp_bits, "tmp"))

    n = len(qregs)
    for r in qregs:
        s = r.size
        fr.append(pos)
        to.append(pos+s)
        pos += s
        w = max(s, len(r.name))
        wid.append(w)
        pad = " "*(w-len(r.name))
        lab.append(pad + r.name)
    total_bits = pos
    title = ",".join(reversed(lab)) + ",amplitude"
    print(title)
    for i, z in enumerate(sv.data):
        if abs(z) < eps:
            continue
        z *= gp
        cols = []
        allbits = bin((1<<total_bits) + i)[-total_bits:]
        for j in range(n):
            bits = allbits[total_bits-to[j]:total_bits-fr[j]]
            w = wid[j]
            nb = to[j] - fr[j]
            pad = " "*(w-nb)
            cols.append(pad + bits)
        spaced_bits = ",".join(reversed(cols))
        print(spaced_bits + f", {amp.format_in_polar(z)}")

def decode_state_key(qc: QuantumCircuit, key: int) -> dict[str,int]:
    """ Get a dict of { "register_name": int_value } from
        a state vector key integer.
        ex.
        qc: qregs[ QR("a",3), QR("b",2)]  # Lower 3 bits is a, higher 2 bits is b
        key: 0b10111
        result: { "a": 0b111, "b", 0b10 }
    """
    result = {}
    for qreg in qc.qregs:
        reg_size = qreg.size
        reg_mask = (1 << reg_size) - 1
        reg_val = key & reg_mask
        result[qreg.name] = reg_val
        key >>= reg_size
    return result

def extract_wave_functions(qc: QuantumCircuit,
                           grouping_regs: list[QuantumRegister],
                           wave_regs: list[list[QuantumRegister]],
                           state_vector: Statevector,
                           eps = 1.0e-6,
                           sv_has_spin = False
                           ) -> dict[int, np.ndarray[float]] :
    """ Returns
        { "011010": np.array[wave_function_index, pos_index]}
    """
    # construct bit address map
    of = open("extract.txt","w", encoding="utf-8")
    num_particles = len(wave_regs)
    num_bits = wave_regs[0][0].size
    num_positions = 1 << num_bits
    reg_start_pos = {}
    pos = 0
    total_bits = len(qc.qubits)
    for reg in qc.qregs:
        reg_start_pos[reg.name] = pos
        pos += reg.size
        print("reg ", reg.name, " start_pos: ", reg_start_pos[reg.name])
    group_reg_pos = []
    for reg in grouping_regs:
        print("group reg: ", reg.name)
        n = reg.size
        group_reg_pos.append(
            ((reg_start_pos[reg.name]),
             (1 << n) - 1))
    wave_reg_pos = []
    if sv_has_spin:
        spatial_dims = len(wave_regs[0])-1
    else:
        spatial_dims = len(wave_regs[0])
    for elec in wave_regs:
        for dim in elec[:spatial_dims]:
            reg = dim
            print("wave_function_reg: ", reg.name)
            n = reg.size
            wave_reg_pos.append((
                reg_start_pos[reg.name],
                (1 << n) - 1
                ))

    m = {}
    for state_key, p in enumerate(state_vector.data):
        # if abs(p) < eps:
        #     continue
        group_key_list = []
        for pos, mask in group_reg_pos:
            step_key_val = (state_key >> pos) & mask
            group_key_list.append(bin(step_key_val + mask + 1)[3:])
        group_key = " ".join(group_key_list)
        of.write(f"group_key: {group_key}, state_key = " +
                 bin(state_key + (1 << total_bits))[-total_bits:] +
                 f" p = {p}\n")
        if group_key not in m:
            m[group_key] = np.zeros((num_particles, num_positions))
        dat = m[group_key]
        positions = []
        for i, (pos, mask) in enumerate(wave_reg_pos):
            x = (state_key >> pos) & mask
            # dat[i][x] += (p * p.conjugate()).real
            dat[i][x] += p.real
            positions.append(x)
        of.write(f"{positions} = {p}\n")
    of.close()
    # for _key, data in m.items():
    #     for orb in data:
    #         for v in orb:
    #             v = math.sqrt(v)
    return m
