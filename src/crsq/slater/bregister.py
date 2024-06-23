""" permutation B-register
"""
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit
from qiskit.circuit.library import XGate, CXGate, MCXGate, SwapGate, RZGate
from crsq_heap.heap import Frame

from crsq.slater.sigma import build_sums
from crsq.slater.aregister import ARegister, ARegisterFrame

def _min_bits_to_hold(v):
    """ get minimum number of bits to hold value v """
    n = 1  # number of digits
    t = 2  # 2 to the power of n
    while v > t-1:
        t *= 2
        n += 1
    return n

def _make_cond_bits_for_permutation(
        top_reg: int, value: int) -> tuple[int, str]:
    """ make bit string to be used in a controlled gate
        to check if the condition bits match the given value
        top_reg = 8, value =2 -> (3, "010")
    """
    assert top_reg > 1
    num_cond_bits = _min_bits_to_hold(top_reg)
    cond_bits_str = bin((1 << num_cond_bits) + value)[-num_cond_bits:]
    return num_cond_bits, cond_bits_str

def _change_value_from_known_state_conditionally(
        qc: QuantumCircuit, ancilla: Qubit, qr: QuantumRegister,
        from_value: int, to_value: int):
    """ Add gates to qc so that register qr, guaranteed to hold
        value 'from_value' will be changed to 'to_value'.
        Only the bits need to be flipped will be flipped
    """
    assert from_value >= 0 and to_value >= 0
    all_bits = from_value | to_value
    i = 0
    mask = 1
    while mask <= all_bits:
        if (from_value ^ to_value) & mask > 0:
            # need to flip this bit.
            qc.cx(ancilla, qr[i])
        i += 1
        mask *= 2

def _build_set_value_to_reg_0_to_2(
        qc: QuantumCircuit, should_swap_aregs: bool, btgt: QuantumRegister,
        atop: list[QuantumRegister] | None, atgt: list[QuantumRegister] | None):
    """ special case for _build_set_value_to_reg from 0 to 2 """
    if should_swap_aregs:
        assert atop is not None and atgt is not None
        for d, _t in enumerate(atgt):
            for i in range(atgt[d].size):
                qc.cswap(btgt[0], atop[d][i], atgt[d][i], ctrl_state=0)
    # negative controlled not
    qc.append(CXGate(ctrl_state=0), [btgt[0], btgt[1]])

def _build_set_value_to_reg_1_to_2(
        qc: QuantumCircuit, should_swap_aregs: bool, btgt: QuantumRegister,
        atop: list[QuantumRegister] | None, atgt: list[QuantumRegister] | None):
    """ special case for _build_set_value_to_reg from 1 to 2 """
    if should_swap_aregs:
        assert atop is not None and atgt is not None
        for d, _t in enumerate(atgt):
            for i in range(atgt[d].size):
                qc.cswap(btgt[0], atop[d][i], atgt[d][i])
    qc.swap(btgt[0], btgt[1])
    return

def _build_set_value_to_reg(
        qc: QuantumCircuit, should_swap_aregs: bool, ancilla: Qubit,
        btgt: QuantumRegister,
        atop: list[QuantumRegister] | None, atgt: list[QuantumRegister] | None,
        num_cond_bits: int, from_bits_str: str,
        from_value: int, to_value: int):
    """ emit the gates to change the content of a register from
        from_value to to_value conditionally when the content is
        from_value.
    """
    assert num_cond_bits >= 2
    if num_cond_bits == 2 and to_value == 2:
        # special short-cut cases:
        if from_value == 0:
            _build_set_value_to_reg_0_to_2(qc, should_swap_aregs, btgt, atop, atgt)
            return
        if from_value == 1:
            _build_set_value_to_reg_1_to_2(qc, should_swap_aregs, btgt, atop, atgt)
            return
    # general cases will use the ancilla bit.
    # set the ancilla bit if the cond bits match from_value
    bits = btgt[:num_cond_bits] + [ancilla]
    qc.append(XGate().control(num_cond_bits, ctrl_state=from_bits_str), bits)
    # change bk from from_value to to_value when ancilla is set.
    _change_value_from_known_state_conditionally(
        qc, ancilla, btgt, from_value, to_value)
    # swap areg_top and areg when ancilla is set.
    if should_swap_aregs:
        assert atop is not None and atgt is not None
        for d, _t in enumerate(atgt):
            for i in range(atgt[d].size):
                qc.cswap(ancilla, atop[d][i], atgt[d][i])
    # reset the ancilla to its original state.
    to_bits_str = bin((1 << num_cond_bits) + to_value)[-num_cond_bits:]
    qc.append(XGate().control(num_cond_bits, ctrl_state=to_bits_str), bits)

def _build_set_value_to_reg_0_to_2_conditionally(
        qc: QuantumCircuit, should_swap_aregs: bool,
        btop: QuantumRegister, btgt: QuantumRegister,
        atop: QuantumRegister | None, atgt: QuantumRegister | None,
        num_cond_bits: int, from_bits_str: str):
    """ special case for _build_set_value_to_reg_conditionally
        from 0 to 2
    """
    top_ctrl_bits = btop[:num_cond_bits]
    if should_swap_aregs:
        assert atop is not None and atgt is not None
        for d, _t in enumerate(atgt):
            for i in range(atgt[d].size):
                qc.append(SwapGate().control(
                        num_cond_bits+1, ctrl_state="0"+from_bits_str),
                        top_ctrl_bits + [btgt[0], atop[d][i], atgt[d][i]])
    # negative controlled not
    qc.append(MCXGate(
              num_cond_bits+1, ctrl_state="0"+from_bits_str),
              top_ctrl_bits + [btgt[0], btgt[1]])

def _build_set_value_to_reg_1_to_2_conditionally(
        qc: QuantumCircuit, should_swap_aregs: bool,
        btop: QuantumRegister, btgt: QuantumRegister,
        atop: QuantumRegister | None, atgt: QuantumRegister | None,
        num_cond_bits: int, from_bits_str: str):
    """ special case for _build_set_value_to_reg_conditionally
        from 1 to 2
    """
    top_ctrl_bits = btop[:num_cond_bits]
    if should_swap_aregs:
        assert atop is not None and atgt is not None
        for d, _t in enumerate(atgt):
            for i in range(atgt[d].size):
                qc.append(SwapGate().control(
                        num_cond_bits+1, ctrl_state="1"+from_bits_str),
                        top_ctrl_bits + [btgt[0], atop[d][i], atgt[d][i]])
    # qc.swap(qtgt[0], qtgt[1])
    qc.append(SwapGate().control(
              num_cond_bits, ctrl_state=from_bits_str),
              top_ctrl_bits + [btgt[0], btgt[1]])

def _build_set_value_to_reg_conditionally(
        qc: QuantumCircuit, should_swap_aregs: bool, ancilla: Qubit,
        btop: QuantumRegister, btgt: QuantumRegister,
        atop: list[QuantumRegister] | None, atgt: list[QuantumRegister] | None,
        num_cond_bits: int, from_bits_str: str,
        from_value: int, to_value: int):
    """ if atgt is None, only permutation will be done,
        and swapping will be omitted.
    """
    assert num_cond_bits >= 2
    top_ctrl_bits = btop[:num_cond_bits]
    if num_cond_bits == 2 and to_value == 2:
        # special short-cut cases:
        if from_value == 0:
            _build_set_value_to_reg_0_to_2_conditionally(
                qc, should_swap_aregs, btop, btgt, atop, atgt, num_cond_bits, from_bits_str)
            return
        if from_value == 1:
            _build_set_value_to_reg_1_to_2_conditionally(
                qc, should_swap_aregs, btop, btgt, atop, atgt, num_cond_bits, from_bits_str)
            return
    # general cases will use the ancilla bit.:
    bits = top_ctrl_bits + btgt[:num_cond_bits] + [ancilla]
    qc.append(XGate().control(num_cond_bits*2,
                              ctrl_state=from_bits_str*2), bits)
    if should_swap_aregs:
        assert atop is not None and atgt is not None
        for d, _t in enumerate(atgt):
            for i in range(atgt[d].size):
                qc.cswap(ancilla, atop[d][i], atgt[d][i])
    _change_value_from_known_state_conditionally(
        qc, ancilla, btgt, from_value, to_value)
    to_bits_str = bin((1 << num_cond_bits) + to_value)[-num_cond_bits:]
    qc.append(XGate().control(num_cond_bits*2,
                              ctrl_state=to_bits_str+from_bits_str), bits)


class BRegisterFrame(Frame):
    """ The Frame for BRegister

        :param areg_frame: [optional] frame for ARegister.
    """
    def __init__(self, areg_frame: ARegisterFrame|None = None, label="Shuffle"):
        super().__init__(label=label)
        if areg_frame is not None:
            self.areg_frame: ARegisterFrame = areg_frame
            areg_frame.attach_to_parent_frame(self)
        self.bregs: list[QuantumRegister] = []
        self.ancilla: QuantumRegister = None

    def set_bregs(self, bregs: list[QuantumRegister]):
        """ Store the BRegister QuantumRegisters
        """
        self.bregs = bregs
        self.add_param(*bregs)

    def set_ancilla(self, ancilla: QuantumRegister):
        """ Store the ancilla register
        """
        self.ancilla = ancilla
        self.add_param(ancilla)


class BRegister:
    """ The B register
    """
    def __init__(self, num_orbit_index_bits: int, N: int, dimension: int):
        """ Just set the size of the B register.
        """
        assert 2**num_orbit_index_bits >= N
        self.frame: BRegisterFrame
        self.num_orbital_index_bits: int = num_orbit_index_bits
        self.N: int = N
        self.dimension = dimension
        self.sums_are_ready: bool = False
        self.use_custom_gates = True
        self.areg: ARegister | None = None
        self.state_index = 0
        self.should_swap_aregs: bool = True
        self.should_save_statevector: bool = False

    def set_use_custom_gates(self, use_custom_gates: bool):
        """ Instruct to use custom gates for sub-structures
            so that the circuit diagram will become compact.
            Set this to False to see inside the sub-structures
            for debugging or explaining.
        """
        self.use_custom_gates = use_custom_gates

    def set_should_swap_aregs(self, should_swap_aregs: bool):
        """ request to perform swapping of the areg bits.

            Set this to false to do permutation only without
            the swapping.
        """
        self.should_swap_aregs = should_swap_aregs

    def set_areg(self, areg: ARegister):
        """ Set the A register.

            Call this before allocate_registers.
            This can be skipped when you do not need to swap
            ARegister registers.
        """
        self.areg = areg

    def allocate_registers(self) -> BRegisterFrame:
        """ Allocate the value registers and the ancilla bits register """
        if self.should_swap_aregs and self.areg is None:
            raise ValueError("should_do_swap is true but areg is None.")
        if not self.should_swap_aregs and self.areg is not None:
            raise ValueError("areg is set but should_do_swap is true.")

        areg_frame = None
        if self.should_swap_aregs:
            areg_frame = ARegisterFrame()
        frame = BRegisterFrame(areg_frame)
        self.allocate_registers_on_frame(frame)
        self.frame = frame
        return frame

    def allocate_registers_on_frame(self, frame: BRegisterFrame):
        """ Allocate registers on the register set
        """
        self.add_ab_registers_upto_k(frame, self.N, self.N)

    def add_ab_registers_upto_k(self, frame: BRegisterFrame, aregs_num: int, bregs_num: int):
        """ Allocate registers on the register set
        """
        if self.should_swap_aregs:
            assert self.areg is not None
            assert frame.areg_frame is not None
            self.areg.add_a_registers_upto(frame.areg_frame, aregs_num)

        if self.N >= 3:
            # ancilla will be used.
            ancilla = QuantumRegister(1, "shuffle")
            frame.set_ancilla(ancilla)

        if self.N >= 2:
            bregs = []
            for i in range(bregs_num):
                bregs.append(QuantumRegister(self.num_orbital_index_bits, f"b{i}"))
            frame.set_bregs(bregs)

    def build_sums(self, frame: BRegisterFrame):
        """ Build the gates to compute the factorials. """
        qc = frame.circuit
        if self.N > 1:
            for k in range(self.N):
                build_sums(qc, frame.bregs[k], k+1, self.use_custom_gates)
        self.sums_are_ready = True

    def build_permutations(self, frame: BRegisterFrame):
        """ build the gates to compute permutations on all registers """
        assert self.sums_are_ready
        if self.N > 1:
            for k in range(1, self.N):
                self._shuffle_registers_upto(frame, k)

    def _shuffle_registers_upto(self, frame: BRegisterFrame, k: int):
        """ shuffle all registers lower than k so that registers 0..k
            will hold a set of permutations.
        """
        for j in range(k-1, -1, -1):
            self._shuffle_register_pair(frame, k, j)

    def _shuffle_register_pair(self, frame: BRegisterFrame, top_reg: int, from_value: int):
        """ build gates to check the b_reg[top_reg] value if is from_value,
            and when it is, alter its content to top_reg.
        """
        # special easy case
        if top_reg == 1:
            assert from_value == 0
            self._shuffle_register_pair_1_0(frame)
            return
        assert top_reg > 1
        # value str to check if reg (< top_reg) has content = value
        num_cond_bits, cond_bits_str = _make_cond_bits_for_permutation(
            top_reg, from_value)
        if self.use_custom_gates:
            self._shuffle_register_pair_with_custom_gates(
                frame, top_reg, num_cond_bits, cond_bits_str, from_value)
        else:
            self._shuffle_register_pair_with_std_gates(
                frame, top_reg, num_cond_bits, cond_bits_str, from_value)

    def _shuffle_register_pair_1_0(self, frame: BRegisterFrame):
        """ Special easy case to shuffle b1 with b0"""
        qc = frame.circuit
        # negative controlled not
        top_reg = 1
        bregs = frame.bregs
        ctrl_bit = bregs[top_reg][0]
        qc.cx(ctrl_bit, bregs[0][0], ctrl_state=0)
        if self.should_swap_aregs:
            assert frame.areg_frame is not None
            aregs = frame.areg_frame.aregs
            atgt = aregs[0]
            atop = aregs[top_reg]
            qc.x(ctrl_bit)
            qc.z(ctrl_bit)
            qc.x(ctrl_bit)
            for d, _t in enumerate(atgt):
                for i in range(atgt[d].size):
                    qc.cswap(ctrl_bit, atop[d][i], atgt[d][i], ctrl_state=0)

    def _shuffle_register_pair_with_custom_gates(
            self, frame: BRegisterFrame, top_reg: int, num_cond_bits: int,
            cond_bits_str: str, from_value: int):
        """ Build and apply a controlled custom gate to set values"""
        qc = frame.circuit
        bregs = frame.bregs
        cond_bits = bregs[top_reg][:num_cond_bits]
        target_bits = []
        for bk in bregs[:top_reg]:
            #  target_bits.append(reg[:]) does not work.
            target_bits += bk[:]
        areg_bits = []
        if self.should_swap_aregs:
            assert frame.areg_frame is not None
            aregs = frame.areg_frame.aregs
            for i in range(top_reg+1):
                for d in range(self.dimension + 1):
                    ak = aregs[i][d]
                    areg_bits += ak[:]
        sub_frame = self._make_frame_to_shuffle_regs(
            top_reg, num_cond_bits, cond_bits_str, from_value)
        set_gate = sub_frame.circuit.to_gate(label=sub_frame._label)
        controlled_set_gate = set_gate.control(
            num_cond_bits, ctrl_state=cond_bits_str)
        temp_bits = frame.allocate_temp_bits(sub_frame.opaque_bit_count)
        all_bits = cond_bits + sub_frame.param_bits_list() + temp_bits
        qc.append(controlled_set_gate, all_bits)
        frame.free_temp_bits(temp_bits)

    def _shuffle_register_pair_with_std_gates(
            self, frame: BRegisterFrame, top_reg: int, num_cond_bits: int,
            cond_bits_str: str, from_value: int):
        qc = frame.circuit
        bregs = frame.bregs
        btop = bregs[top_reg]
        atop = None
        atgt = None
        for i in range(top_reg-1, -1, -1):
            btgt = bregs[i]
            if self.should_swap_aregs:
                assert frame.areg_frame is not None
                aregs = frame.areg_frame.aregs
                atop = aregs[top_reg]
                atgt = aregs[i]
            ancilla = frame.ancilla
            _build_set_value_to_reg_conditionally(
                qc, self.should_swap_aregs, ancilla[0],
                btop, btgt, atop, atgt,
                num_cond_bits, cond_bits_str, from_value, top_reg)
        if self.should_swap_aregs:
            assert frame.areg_frame is not None
            aregs = frame.areg_frame.aregs
            atgt = aregs[0]
            rybits = btop[:num_cond_bits] + [atgt[0][0]]
            qc.append(RZGate(2*np.pi).control(
                num_cond_bits, ctrl_state=cond_bits_str), rybits)

    def _make_frame_to_shuffle_regs(
            self, top_reg: int, num_cond_bits: int,
            cond_bits_str: str, from_value: int) -> BRegisterFrame:
        assert top_reg > 1
        frame: BRegisterFrame = self._allocate_shuffle_gate_registers(top_reg, from_value)
        self._emit_shuffle_circuit(
            frame, top_reg, num_cond_bits, cond_bits_str, from_value)
        return frame

    def _allocate_shuffle_gate_registers(
            self, top_reg: int, from_value: int) -> BRegisterFrame:
        areg_frame = None
        if self.should_swap_aregs:
            areg_frame = ARegisterFrame()
        breg_frame = BRegisterFrame(areg_frame, label=f"Sh({from_value},{top_reg})")
        self.add_ab_registers_upto_k(breg_frame, top_reg+1, top_reg) # +1?
        return breg_frame

    def _emit_shuffle_circuit(
            self, frame: BRegisterFrame,
            top_reg: int, num_cond_bits: int, cond_bits_str: str, from_value: int):
        qc = frame.circuit
        bregs = frame.bregs
        for i in range(top_reg-1, -1, -1):
            btgt = bregs[i]
            if self.should_swap_aregs:
                assert frame.areg_frame is not None
                aregs = frame.areg_frame.aregs
                atop = aregs[top_reg]
                atgt = aregs[i]
            else:
                atop = None
                atgt = None
            ancilla = frame.ancilla
            _build_set_value_to_reg(
                qc, self.should_swap_aregs,
                ancilla[0], btgt, atop, atgt,
                num_cond_bits, cond_bits_str, from_value, top_reg)
        if self.should_swap_aregs:
            assert frame.areg_frame is not None
            aregs = frame.areg_frame.aregs
            atgt = aregs[0][0]
            assert atgt is not None
            qc.z(atgt[0])
            qc.x(atgt[0])
            qc.z(atgt[0])
            qc.x(atgt[0])
