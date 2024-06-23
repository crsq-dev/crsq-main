""" Antisymmetrization block
"""

from typing import List
import math
import logging

from qiskit import QuantumRegister
from qiskit.circuit.library import XGate, ZGate
from crsq_heap import heap
from crsq.blocks import wave_function
from crsq import slater

logger = logging.getLogger(__name__)

class AntisymmetrizationSpec:
    """ parameters for antisymmetrization
        :param method: 1 : conventional, 2: unary encoding, 3: binary
    """
    def __init__(self, wfr_spec: wave_function.WaveFunctionRegisterSpec, method: int):
        self._wfr_spec = wfr_spec
        self.method = method
        self._should_do_anti_symmetrization = True
        if method == 0:
            self._should_do_anti_symmetrization = False
        if method == 1:
            self._init_for_conventional()
        elif method == 2:
            self._init_for_unary_coding()
        elif method == 3:
            self._init_for_binary_coding()
        else:
            raise ValueError("Unknown method number")
        logger.info("AntisymmetrizationSpec: method = %d", method)

    def _init_for_conventional(self):
        wfr_spec = self._wfr_spec
        Ne = wfr_spec.num_electrons
        oi = wfr_spec.num_orbital_index_bits
        if Ne >= 2:
            # list of (size, name) for registers
            self._sigma_register_specs = [(oi, f"b{k}") for k in range(Ne)]
        else:
            self._sigma_register_specs = []

        if Ne >= 3:
            self._ancilla_register_size = 1
        else:
            self._ancilla_register_size = 0

    def _unary_bit_num_for(self, eta):
        if eta == 2:
            return 1
        return eta

    def _init_for_unary_coding(self):
        wfr_spec = self._wfr_spec
        Ne = wfr_spec.num_electrons
        self._sigma_register_specs = [
            (self._unary_bit_num_for(k), f"au{k-1}") for k in range(2, Ne+1)]
        self._ancilla_register_size = 0

    def _init_for_binary_coding(self):
        wfr_spec = self._wfr_spec
        Ne = wfr_spec.num_electrons
        self._sigma_register_specs = [
            (math.ceil(math.log2(k)), f"ab{k-1}") for k in range(2, Ne+1)]
        if Ne > 1:
            self._ancilla_register_size = 1
        else:
            self._ancilla_register_size = 0

    @property
    def should_do_anti_symmetrization(self)->bool:
        """ should do anti-symmetrization """
        return self._should_do_anti_symmetrization

    @property
    def wfr_spec(self) -> wave_function.WaveFunctionRegisterSpec:
        """ wfr spec"""
        return self._wfr_spec

    def allocate_sigma_regs(self) -> List[QuantumRegister]:
        """ allocate sigma regs """
        regs = [QuantumRegister(s, n) for s, n in self._sigma_register_specs]
        return regs

    def allocate_ancilla_register(self) -> QuantumRegister | None:
        """ allocate ancilla registers"""
        if self._ancilla_register_size > 0:
            return QuantumRegister(self._ancilla_register_size, "shuffle")
        else:
            return None

class ABRegisterPermutationBlock(heap.Frame):
    """ Gate to Antisymmetrize wave function set with conventional method
    """
    def __init__(self, wfr_spec: wave_function.WaveFunctionRegisterSpec, build=True):
        super().__init__(label=" S_ab")
        self._wfr_spec = wfr_spec
        self._eregs: List[List[QuantumRegister]] = []
        self._slater_areg = slater.ARegister(
            wfr_spec.num_coordinate_bits,
            wfr_spec.num_electrons,
            wfr_spec.dimension,
            wfr_spec.use_spin
        )
        self._slater_breg = slater.BRegister(
            wfr_spec.num_orbital_index_bits,
            wfr_spec.num_electrons,
            wfr_spec.dimension
        )
        self._slater_breg.set_areg(self._slater_areg)
        self._slater_areg_frame = slater.ARegisterFrame()
        self._slater_breg_frame = slater.BRegisterFrame(
            self._slater_areg_frame)
        self.allocate_registers()
        if build:
            self.build_circuit()

    def allocate_registers(self):
        """ allocate registers """
        self._eregs = self._wfr_spec.allocate_elec_registers()
        self.add_param(('eregs', self._eregs))
        self._slater_breg.allocate_registers_on_frame(self._slater_breg_frame)
        self.add_param(('sigmas', self._slater_breg_frame.bregs))
        self.add_param(self._slater_breg_frame.ancilla)

    def build_circuit(self):
        """ build circuit """
        self._slater_breg.build_sums(self._slater_breg_frame)
        self._slater_breg.build_permutations(self._slater_breg_frame)
        self.apply(self._slater_breg_frame)

    def bind(self,
             sigmas: List[QuantumRegister],
             ancilla: QuantumRegister,
             eregs: List[List[QuantumRegister]]) -> heap.Binding:
        """ create binding
        """
        return heap.Binding(self, {
            "sigmas": sigmas,
            "shuffle": ancilla,
            "eregs": eregs
        })


class UnaryCodedSequenceBlock(heap.Frame):
    """ Gate for |LC_k> with unary coding

        Produces |lc> = 1/sqrt(k)(|0001> + |0010> + ... + |1000>)

    """
    def __init__(self, k: int, build=True):
        super().__init__(label=f" ku_Σ({k})")
        assert k >= 2
        self._k = k
        self._areg: QuantumRegister
        self.allocate_registers()
        if build:
            self.build_circuit()

    def allocate_registers(self):
        """ allocate the registers"""
        k = self._k
        if k == 2:
            num_bits = 1
        else:
            num_bits = k
        self._areg = QuantumRegister(num_bits, f"au{k-1}")
        self.add_param(self._areg)

    def build_circuit(self):
        """ build the circuit"""
        qc = self.circuit
        k = self._k
        if k == 2:
            qc.ry(math.pi/2, 0)
            return
        s1 = math.sqrt(1/k)
        s2 = math.sqrt((k-1)/k)
        th=2*math.atan2(s2, s1)
        qc.x(0)
        qc.ry(th, 1)
        qc.cx(1,0)
        for i in range(2, k):
            s1 = math.sqrt(1/(k-i+1))
            s2 = math.sqrt((k-i)/(k-i+1))
            th=2*math.atan2(s2, s1)
            qc.cry(th, i-1, i)
            qc.cx(i, i-1)

    def bind(self, areg: QuantumRegister) -> heap.Binding:
        """ create binding """
        return heap.Binding(self, {
            self._areg.name: areg
            })


class UnaryCodedShuffleBlock(heap.Frame):
    """ Gate for σ (unary)"""
    def __init__(self, wfr_spec: wave_function.WaveFunctionRegisterSpec, eta: int, build=True):
        super().__init__(label=f" σu({eta})")
        self._wfr_spec = wfr_spec
        self._eta = eta
        self._areg: QuantumRegister
        self._eregs: List[List[QuantumRegister]]
        self.allocate_registers()
        if build:
            self.build_circuit()

    def allocate_registers(self):
        """allocate"""
        wfr_spec = self._wfr_spec
        eta = self._eta
        if eta == 2:
            areg_size = 1
        else:
            areg_size = eta
        self._areg = QuantumRegister(areg_size, f"au{eta-1}")
        self._eregs = wfr_spec.allocate_elec_registers_upto(eta)
        self.add_param(
            ("eregs", self._eregs),
            ("areg", self._areg)
            )

    def build_circuit(self):
        """ build """
        eta = self._eta
        for k in range(eta-1):
            self._controlled_shuffle_unary(eta, k)
        qc = self.circuit
        lcreg = self._areg
        flip_bit = lcreg[lcreg.size-1]
        qc.z(flip_bit)

    def _controlled_shuffle_unary(self, eta: int, k: int):
        qc = self.circuit
        ctrl_bit = self._areg[k]
        eregs = self._eregs
        if eta == 2:
            ctrl_state = 0
        else:
            ctrl_state = 1
        # loop includes spin
        for d, _s in enumerate(eregs[k]):
            for j, _t in enumerate(eregs[k][d]):
                qc.cswap(ctrl_bit, eregs[k][d][j], eregs[eta-1][d][j], ctrl_state=ctrl_state)

    def bind(self, areg: QuantumRegister, eregs: List[List[QuantumRegister]]) -> heap.Binding:
        """ produce binding """
        return heap.Binding(self, {
            "eregs": eregs,
            "areg": areg
        })


class UnaryCodedPermutationBlock(heap.Frame):
    """ build Hartree-Fock state using unary-coded ancilla registers """
    def __init__(self, asy_spec: AntisymmetrizationSpec, build=True):
        super().__init__(label=f" S_u")
        self._asy_spec = asy_spec
        self._wfr_spec = asy_spec.wfr_spec
        self._aregs: List[QuantumRegister]
        self._eregs: List[QuantumRegister]
        self.allocate_registers()
        if build:
            self.build_circuit()

    def allocate_registers(self):
        """ allocate """
        wfr_spec = self._wfr_spec
        self._eregs = wfr_spec.allocate_elec_registers()
        self._aregs = self._asy_spec.allocate_sigma_regs()
        self.add_param(
            ("eregs", self._eregs),
            ("aregs", self._aregs)
            )

    def build_circuit(self):
        """ build"""
        wfr_spec = self._wfr_spec
        Ne = wfr_spec.num_electrons
        aregs = self._aregs
        eregs = self._eregs
        # negate phase of register 0 when
        # number of swap gates is odd.
        num_swap_gates = Ne -1
        if num_swap_gates % 2 == 1:
            qc = self.circuit
            qc.x(aregs[0][0])
            qc.z(aregs[0][0])
            qc.x(aregs[0][0])
        for eta in range(2, Ne+1):
            lcb = UnaryCodedSequenceBlock(eta)
            self.invoke(lcb.bind(areg=aregs[eta-2]))
        for eta in range(2, Ne+1):
            shuff = UnaryCodedShuffleBlock(wfr_spec, eta)
            self.invoke(shuff.bind(areg=aregs[eta-2], eregs=eregs[0:eta]))

    def bind(self, aregs: List[QuantumRegister], eregs: List[QuantumRegister]) -> heap.Binding:
        """ produce binding """
        return heap.Binding(self, {
            "aregs": aregs,
            "eregs": eregs
        })


class BinaryCodedSequenceBlock(heap.Frame):
    """ Gate for |LC_k> with binary coding

        Produces |lc_4> = 1/sqrt(4)(|00> + |01> + |10> + |11>))
    """
    def __init__(self, num_bits, k: int, build=True):
        super().__init__(label=f"  kb_Σ({num_bits},{k})")
        assert k >= 2
        assert (1 << num_bits) >= k
        self._k = k
        self._areg: QuantumRegister
        self._num_bits = num_bits
        self.allocate_registers()
        if build:
            self.build_circuit()

    def allocate_registers(self):
        """ allocate """
        self._areg = QuantumRegister(self._num_bits, f"ab{self._k-1}")
        self.add_param(self._areg)

    def build_circuit(self):
        """ build """
        N = self._k
        qc = self.circuit
        areg = self._areg
        num_bits = self._num_bits
        lower_bits = num_bits - 1
        lower_N = 1 << lower_bits
        while N <= lower_N:
            num_bits -= 1
            lower_bits -= 1
            lower_N = 1 << lower_bits
        if (1 << num_bits) == N:
            qc.h(areg[0:num_bits])
            return
        lower_reg = QuantumRegister(bits=areg[0:lower_bits])
        upper_N = N - lower_N
        theta = 2*math.atan2(math.sqrt(upper_N), math.sqrt(lower_N))
        top_bit = areg[num_bits - 1]
        qc.ry(theta, top_bit)
        qc.ch(top_bit, lower_reg, ctrl_state=0)
        if lower_bits == 1:
            return
        if upper_N == 1:
            return
        sub_block = BinaryCodedSequenceBlock(lower_bits, upper_N)
        self.invoke_with_control(sub_block.bind(areg=lower_reg), [top_bit], "1")

    def bind(self, areg: QuantumRegister) -> heap.Binding:
        """ bind """
        return heap.Binding(self, {
            self._areg.name: areg
            })


class BinaryCodedShuffleBlock(heap.Frame):
    """ Gate for σ (binary)"""
    def __init__(self, wfr_spec: wave_function.WaveFunctionRegisterSpec, eta: int, build=True):
        super().__init__(label=f" σb({eta})")
        self._wfr_spec = wfr_spec
        self._eta = eta
        self._areg: QuantumRegister
        self._eregs: List[List[QuantumRegister]]
        self._num_bits = math.ceil(math.log2(eta))
        self._swap_ancilla_reg: QuantumRegister
        self.allocate_registers()
        if build:
            self.build_circuit()

    def allocate_registers(self):
        """ allocate """
        wfr_spec = self._wfr_spec
        eta = self._eta
        num_bits = self._num_bits
        self._areg = QuantumRegister(num_bits, f"ab{eta-1}")
        self._swap_ancilla_reg = QuantumRegister(1, "swap")
        self._eregs = wfr_spec.allocate_elec_registers_upto(eta)
        # Ordered to make the diagram look nice:
        self.add_param(
            ("eregs", self._eregs),
            self._swap_ancilla_reg,
            ("areg", self._areg))

    def build_circuit(self):
        """ build """
        eta = self._eta
        for k in range(eta-1):
            self._controlled_shuffle_binary(k)
        self._negate_eta_when_eta_is_k()

    def _controlled_shuffle_binary(self, k):
        """ when |lcreg> == k swap(ereg[k], ereg[eta-1])
        """
        eta = self._eta
        qc = self.circuit
        num_ctrl_bits = self._num_bits
        offset = 1 << num_ctrl_bits
        ctrl_state = bin(offset + k)[-num_ctrl_bits:]
        ctrl_bits = self._areg[:]
        eregs = self._eregs
        ancilla = self._swap_ancilla_reg
        cx_gate = XGate().control(num_ctrl_bits, ctrl_state=ctrl_state)
        cx_bits = ctrl_bits + ancilla[:]
        # flip swap ancilla
        qc.append(cx_gate, cx_bits)
        # loop includes spin.
        for d, _s in enumerate(eregs[k]):
            for j, _t in enumerate(eregs[k][d]):
                qc.cswap(ancilla[0], eregs[k][d][j], eregs[eta-1][d][j])
        # flip swap ancilla back
        qc.append(cx_gate, cx_bits)

    def _negate_eta_when_eta_is_k(self):
        """ when |lcreg> == k negate(lcreg[0])
        """
        qc = self.circuit
        lcreg = self._areg
        key = self._eta - 1
        num_bits = self._num_bits
        num_ctrl_bits = num_bits - 1

        msb = lcreg[num_bits -1]
        if num_ctrl_bits == 0:
            qc.z(msb)
        else:
            key_lower = key & ((1 << num_ctrl_bits) - 1)
            offset = 1 << num_ctrl_bits
            offset = 1 << num_ctrl_bits
            ctrl_state = bin(offset + key_lower)[-num_ctrl_bits:]
            ctrl_bits = lcreg[:num_ctrl_bits]
            ctrl_z_gate = ZGate().control(num_ctrl_bits, ctrl_state=ctrl_state)
            ctrl_z_bits = ctrl_bits + [msb]
            qc.append(ctrl_z_gate, ctrl_z_bits)

    def bind(self,
             areg: QuantumRegister,
             swap: QuantumRegister,
             eregs: list[QuantumRegister]) -> heap.Binding:
        """ produce binding """
        return heap.Binding(self, {
            "areg": areg,
            "swap": swap,
            "eregs": eregs
        })


class BinaryCodedPermutationBlock(heap.Frame):
    """ build Hartree-Fock state using binary-coded ancilla registers """
    def __init__(self, asy_spec: AntisymmetrizationSpec, build=True):
        super().__init__(label=f" S_b")
        self._asy_spec = asy_spec
        self._wfr_spec = asy_spec.wfr_spec
        self._eregs: List[QuantumRegister]
        self._ancilla_reg: QuantumRegister
        self._aregs: List[QuantumRegister]
        self.allocate_registers()
        if build:
            self.build_circuit()

    def allocate_registers(self):
        """ allocate """
        asy_spec = self._asy_spec
        wfr_spec = self._wfr_spec
        self._eregs = wfr_spec.allocate_elec_registers()
        self._aregs = asy_spec.allocate_sigma_regs()
        self._ancilla_reg = asy_spec.allocate_ancilla_register()
        self.add_param(
            ("eregs", self._eregs),
            self._ancilla_reg,
            ("aregs", self._aregs)
            )

    def build_circuit(self):
        """build circuit"""
        wfr_spec = self._wfr_spec
        Ne = wfr_spec.num_electrons
        aregs = self._aregs
        ancilla_reg = self._ancilla_reg
        eregs = self._eregs
        # negate phase of register 0 when
        # number of swap gates is odd.
        num_swap_gates = Ne -1
        if num_swap_gates % 2 == 1:
            qc = self.circuit
            qc.x(aregs[0][0])
            qc.z(aregs[0][0])
            qc.x(aregs[0][0])
        for eta in range(2, Ne+1):
            num_bits = math.ceil(math.log2(eta))
            lcb = BinaryCodedSequenceBlock(num_bits, eta)
            self.invoke(lcb.bind(areg=aregs[eta-2]))
        for eta in range(2, Ne+1):
            shuff = BinaryCodedShuffleBlock(wfr_spec, eta)
            self.invoke(shuff.bind(areg=aregs[eta-2],
                                 swap=ancilla_reg,
                                 eregs=eregs[0:eta]))

    def bind(self,
             eregs: List[QuantumRegister],
             swap: QuantumRegister, 
             aregs: List[QuantumRegister]) -> heap.Binding:
        """ produce binding """
        return heap.Binding(self, {
            "eregs": eregs,
            "shuffle": swap,
            "aregs": aregs
        })
