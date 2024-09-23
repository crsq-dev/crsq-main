""" q-space based time evolution
"""


from contextlib import contextmanager
import math
import logging
import time
from typing import List
from qiskit import QuantumRegister
from qiskit.circuit.library import ZGate
from crsq_heap import heap
import crsq_arithmetic as ari
from crsq.blocks import (
    embed, wave_function,
    energy_initialization,
    antisymmetrization,
    hamiltonian
)
from . import spec


logger = logging.getLogger(__name__)
LOG_TIME_THRESH=1

@contextmanager
def check_time(label:str):
    """ context manager to check time.
        :param label: label used for logs.
    """
    logger.info("%s start", label)
    t1 = time.time()
    yield
    dt = time.time() - t1
    if dt > LOG_TIME_THRESH:
        logger.info("%s took %d msec", label, round(dt*1000))
    else:
        logger.info("%s end", label)

class QSpaceTEVSpec:
    """ q-space time evolution parameters
    """

    def __init__(self, wfr_spec: wave_function.WaveFunctionRegisterSpec,
                  k: int, alpha_l: list[float], t: float, r: int):
        """
            :param k: number of terms in the taylor expansion
            :param alpha_l: coefficients for the unitary expansion of H
            :param t: time of one step of time evolution
            :param r: number to divide one step of time evolution
        """
        self._wfr_spec = wfr_spec
        self._k = k
        self._alpha_l = alpha_l
        self._l = len(alpha_l)
        self._num_l_bits = int(math.ceil(math.log2(self._l)))
        self._t = t
        self._r = r

    @property
    def wfr_spec(self) -> wave_function.WaveFunctionRegisterSpec:
        """ wave function register spec """
        return self._wfr_spec

    @property
    def k(self) -> int:
        """ number of terms in the taylor expansion """
        return self._k

    @property
    def alpha_l(self) -> list[float]:
        """ list of alpha values """
        return self._alpha_l

    @property
    def l(self) -> int:
        """ number of terms in the unitary expansion of H """
        return self._l

    @property
    def num_l_bits(self) -> int:
        """ number of bits to represent l """
        return self._num_l_bits

    @property
    def t(self) -> float:
        """ time of one step of time evolution """
        return self._t

    @property
    def r(self) -> int:
        """ number to divide one step of time evolution """
        return self._r

class RBlock(heap.Frame):
    """ The "R" block
        Negates the phase when all bits in a and b are 0.

    """
    def __init__(self, q_spec: QSpaceTEVSpec, label: str="R", allocate=True, build=True):
        super().__init__(label=label)
        self.q_spec = q_spec
        if allocate:
            self.allocate_registers()
            if build:
                self.build_circuit()

    def allocate_registers(self):
        """ allocate registers """
        q_spec = self.q_spec
        self._a_reg = QuantumRegister(q_spec.k, "a")
        self._b_reg = QuantumRegister(q_spec.num_l_bits, "b")
        self.add_param(self._a_reg, self._b_reg)

    def build_circuit(self):
        """ build the circuit """
        target_bit = self._b_reg[-1]
        ctrl_bits = self._a_reg[:] + self._b_reg[:-1]
        qc = self.circuit
        zgate = ZGate()
        ctrl_str = "0"*len(ctrl_bits)
        qc.x(target_bit)
        qc.append(zgate.control(len(ctrl_bits), ctrl_state=ctrl_str), ctrl_bits + [target_bit])
        qc.x(target_bit)

    def bind(self, a, b):
        """ bind registers.

            :param a: a register. holds value for k.
            :param b: b register.
        """
        return heap.Binding(self, {
            "a": a,
            "b": b
        })


class KBlock(heap.Frame):
    """ Prepare values on areg based on the value K, t and r"""
    def __init__(self, q_spec: QSpaceTEVSpec, label: str="K", allocate=True, build=True):
        super().__init__(label=label)
        self.q_spec = q_spec
        t = q_spec.t
        r = q_spec.r
        K = q_spec.k
        A = sum((t/r)**k/math.factorial(k) for k in range(K+1))
        self._A = A
        self._a = []
        self._b = []
        self._th = []
        ak = math.sqrt(1.0/self._A)
        self._a.append(ak)
        bprod=1
        for k in range(K):
            bk = math.sqrt(1-ak*ak)
            self._b.append(bk)
            bprod *= bk
            ak1 = math.sqrt((t/r)**(k+1)/math.factorial(k+1)/A)/bprod
            self._a.append(ak1)
            ak = ak1
        bk = math.sqrt((t/r)**(K)/math.factorial(K)/A)/bprod
        self._b.append(bk)

        if allocate:
            self.allocate_registers()
            if build:
                self.build_circuit()

    @property
    def A(self):
        """ A value """
        return self._A

    @property
    def ak(self):
        """ a values """
        return self._a

    @property
    def bk(self):
        """ b values """
        return self._b

    def allocate_registers(self):
        """ allocate registers """
        q_spec = self.q_spec
        self._a_reg = QuantumRegister(q_spec.k, "a")
        self.add_param(self._a_reg)

    def build_circuit(self):
        """ build the circuit """
        qc = self.circuit
        qspec = self.q_spec
        areg = self._a_reg
        for k in range(qspec.k):
            th = 2*math.atan2(self._b[k], self._a[k])
            if k == 0:
                qc.ry(th, areg[k])
            else:
                qc.cry(th, areg[k-1], areg[k])

    def bind(self, a):
        """ bind registers.

            :param a: a register. holds value for k.
        """
        return heap.Binding(self, {"a": a })


class LogLBlock(heap.Frame):
    """ Prepare values on breg based on the value alpha_l"""
    def __init__(self, q_spec: QSpaceTEVSpec, label: str="logL", allocate=True, build=True):
        super().__init__(label=label)
        self.q_spec = q_spec
        if allocate:
            self.allocate_registers()
            if build:
                self.build_circuit()

    def allocate_registers(self):
        """ allocate registers """
        q_spec = self.q_spec
        self._b_reg = QuantumRegister(q_spec.num_l_bits, "b")
        self.add_param(self._b_reg)

    def build_circuit(self):
        """ build the circuit """
        qspec = self.q_spec
        breg = self._b_reg
        emb = embed.AmplitudeEmbedGate(qspec.alpha_l)
        self.invoke(emb.bind(q=breg))

    def bind(self, b):
        """ bind registers.

            :param b: b register. holds value for l.
        """
        return heap.Binding(self, {"b": b })


class BParallelBlock(heap.Frame):
    """ The B block for parallel x-y-z version.
    """
    def __init__(self, q_spec: QSpaceTEVSpec, label: str="B_par", allocate=True, build=True):
        super().__init__(label=label)
        self.q_spec = q_spec
        if allocate:
            self.allocate_registers()
            if build:
                self.build_circuit()

    def allocate_registers(self):
        """ allocate registers """
        q_spec = self.q_spec

        self._a_reg = QuantumRegister(q_spec.k, "a")
        self._b_reg = QuantumRegister(q_spec.num_l_bits, "b")
        self.add_param(self._a_reg, self._b_reg)

    def build_circuit(self):
        """ build the circuit """
        kblock = KBlock(self.q_spec)
        loglblock = LogLBlock(self.q_spec)
        self.invoke(kblock.bind(a=self._a_reg))
        self.invoke(loglblock.bind(b=self._b_reg))

    def bind(self, a, b):
        """ bind registers.

            :param a: a register. holds value for k.
            :param b: b register. holds value for l.
        """
        return heap.Binding(self, {
            "a": a,
            "b": b
        })


class BSerialBlock(heap.Frame):
    """ The B block for serial x-y-z version.
    """
    def __init__(self, q_spec: QSpaceTEVSpec, label: str="B_ser", allocate=True, build=True):
        super().__init__(label=label)
        self.q_spec = q_spec
        if allocate:
            self.allocate_registers()
            if build:
                self.build_circuit()

    def allocate_registers(self):
        """ allocate registers """
        q_spec = self.q_spec
        wfr_spec = q_spec.wfr_spec

        self._a_reg = QuantumRegister(q_spec.k, "a")
        self._b2_reg = QuantumRegister(q_spec.num_l_bits, "b2")
        self.add_param(self._a_reg, self._b2_reg)

        self._b1_regs = []
        for dim in range(wfr_spec.dimension):
            label="xyz"[dim]
            b1reg = QuantumRegister(2, f"b1{label}")
            self._b1_regs.append(b1reg)

        self.add_param(("b1_regs", self._b1_regs))

    def build_circuit(self):
        """ build the circuit """
        kblock = KBlock(self.q_spec)
        loglblock = LogLBlock(self.q_spec)
        self.invoke(kblock.bind(a=self._a_reg))
        self.invoke(loglblock.bind(b=self._b2_reg))
        wfr_spec = self.q_spec.wfr_spec
        qc = self.circuit
        for dim in range(wfr_spec.dimension):
            qc.h(self._b1_regs[dim])

    def bind(self, a, b2, b1_regs):
        """ bind registers.

            :param a: a register. holds value for k.
            :param b2: b2 register. holds value for l.
        """
        return heap.Binding(self, {
            "a": a,
            "b2": b2,
            "b1_regs": b1_regs
        })


class HodBlock(heap.Frame):
    """ The H_od block.
        H expansion by unitaries.
    """
    def __init__(self, q_spec: QSpaceTEVSpec, label: str="H_od", allocate=True, build=True):
        super().__init__(label=label)
        self.q_spec = q_spec
        if q_spec.num_l_bits != 2:
            raise ValueError("num_l_bits must be 2")
        if allocate:
            self.allocate_registers()
            if build:
                self.build_circuit()

    def allocate_registers(self):
        """ allocate registers """
        q_spec = self.q_spec
        wfr_spec = q_spec.wfr_spec
        self._b_reg = QuantumRegister(q_spec.num_l_bits, "b")
        self._i_reg = QuantumRegister(wfr_spec.num_coordinate_bits, "i")
        self.add_param(self._b_reg, self._i_reg)

    def build_circuit(self):
        """ build the circuit """
        qspec = self.q_spec
        wfr_spec = qspec.wfr_spec
        qc = self.circuit
        c_bits = self.allocate_temp_bits(wfr_spec.num_coordinate_bits-1)
        h0_gate = ari.scoadder_gate(wfr_spec.num_coordinate_bits, -1, label="Adder -1")
        ch0_gate = h0_gate.control(2, ctrl_state="00")
        target_bits = self._b_reg[:] + self._i_reg[:] + c_bits[:]
        qc.append(ch0_gate, target_bits)
        h1_gate = ari.scoadder_gate(wfr_spec.num_coordinate_bits, 1, label="Adder +1")
        ch1_gate = h1_gate.control(2, ctrl_state="01")
        qc.append(ch1_gate, target_bits)
        self.free_temp_bits(c_bits)

    def bind(self, b, i):
        """ bind registers.

            :param b: b register. holds value for l.
            :param i: i register. holds value for coordinate.
        """
        return heap.Binding(self, {
            "b": b,
            "i": i
        })

class HodxyzBlock(heap.Frame):
    """ H_od block for x-y-z
    """
    def __init__(self, q_spec: QSpaceTEVSpec, label: str="H_od_xyz", allocate=True, build=True):
        super().__init__(label=label)
        self.q_spec = q_spec
        if q_spec.num_l_bits != 2:
            raise ValueError("num_l_bits must be 2")
        if allocate:
            self.allocate_registers()
            if build:
                self.build_circuit()

    def allocate_registers(self):
        """ allocate registers """
        q_spec = self.q_spec
        wfr_spec = q_spec.wfr_spec
        self._b2_reg = QuantumRegister(q_spec.num_l_bits, "b2")
        self.add_param(self._b2_reg)
        self._b1_regs = []
        self._i_regs = []
        for dim in range(wfr_spec.dimension):
            label="xyz"[dim]
            b1reg = QuantumRegister(2, f"b1{label}")
            i_reg = QuantumRegister(wfr_spec.num_coordinate_bits, f"i{label}")
            self._b1_regs.append(b1reg)
            self._i_regs.append(i_reg)
        self.add_param(("b1_regs", self._b1_regs))
        self.add_param(("i_regs", self._i_regs))

    def build_circuit(self):
        """ build the circuit """
        qspec = self.q_spec
        wfr_spec = qspec.wfr_spec
        hod_block = HodBlock(qspec)
        for dim in range(wfr_spec.dimension):
            label="xyz"[dim]
            self.invoke_with_control(
                hod_block.bind(b=self._b1_regs[dim], i=self._i_regs[dim]),
                self._b2_reg[:], bin(dim)[2:].zfill(2),
                label=f"H_{label}")

    def bind(self, b2, b1_regs, i_regs):
        """ bind registers.

            :param b2: b2 register. holds value for l.
            :param b1_regs: b1 registers.
            :param i_regs: i registers.
        """
        return heap.Binding(self, {
            "b2": b2,
            "b1_regs": b1_regs,
            "i_regs": i_regs
        })

class WParallelBlock(heap.Frame):
    """ The "W" block for oblivious amplitude amplification, parallel x-y-z
        version.
    """
    def __init__(self, q_spec: QSpaceTEVSpec, label: str="W_par", allocate=True, build=True):
        super().__init__(label=label)
        self.q_spec = q_spec
        if allocate:
            self.allocate_registers()
            if build:
                self.build_circuit()

    def allocate_registers(self):
        """ allocate registers """
        q_spec = self.q_spec
        wfr_spec = q_spec.wfr_spec

        self._a_reg = QuantumRegister(q_spec.k, "a")
        self._b_reg = QuantumRegister(q_spec.num_l_bits, "b")
        self._i_reg = QuantumRegister(wfr_spec.num_coordinate_bits, f"i")
        self.add_param(self._a_reg, self._b_reg, self._i_reg)

    def build_circuit(self):
        """ build the circuit """
        qc = self.circuit
        q_spec = self.q_spec
        bblock = BParallelBlock(q_spec)
        self.invoke(bblock.bind(a=self._a_reg, b=self._b_reg))
        hod_gate = HodBlock(q_spec)
        for k in range(q_spec.k):
            qc.s(self._a_reg[k])
            self.invoke_with_control(hod_gate.bind(b=self._b_reg, i=self._i_reg),
                                     self._a_reg[k:k+1], "1")
        self.invoke(bblock.bind(a=self._a_reg, b=self._b_reg), inverse=True, label="B_par\u2020")

    def bind(self, a, b, i):
        """ bind registers.

            :param a: a register. holds value for k.
            :param b: b register. holds value for l.
            :param i: i register. holds value for coordinate.
        """
        return heap.Binding(self, {
            "a": a,
            "b": b,
            "i": i
        })

class WSerialBlock(heap.Frame):
    """ The "W" block for oblivious amplitude amplification, serial x-y-z
        version.
    """
    def __init__(self, q_spec: QSpaceTEVSpec, label: str="W_ser", allocate=True, build=True):
        super().__init__(label=label)
        self.q_spec = q_spec
        if allocate:
            self.allocate_registers()
            if build:
                self.build_circuit()

    def allocate_registers(self):
        """ allocate registers
        """
        q_spec = self.q_spec
        wfr_spec = q_spec.wfr_spec

        self._a_reg = QuantumRegister(q_spec.k, "a")
        self._b2_reg = QuantumRegister(q_spec.num_l_bits, "b2")
        self.add_param(self._a_reg, self._b2_reg)

        self._b1_regs = []
        self._i_regs = []
        for dim in range(wfr_spec.dimension):
            axis="xyz"[dim]
            b1reg = QuantumRegister(2, f"b1{axis}")
            ireg = QuantumRegister(wfr_spec.num_coordinate_bits, f"i{axis}")
            self._b1_regs.append(b1reg)
            self._i_regs.append(ireg)

        self.add_param(("b1_regs", self._b1_regs))
        self.add_param(("i_regs", self._i_regs))

    def build_circuit(self):
        """ build the circuit
        """
        qc = self.circuit
        q_spec = self.q_spec
        bser_block = BSerialBlock(q_spec)
        self.invoke(bser_block.bind(a=self._a_reg, b2=self._b2_reg, b1_regs=self._b1_regs))
        hxyz_gate = HodxyzBlock(q_spec)
        for k in range(q_spec.k):
            logger.info("q_spec: k=%d", k)
            qc.s(self._a_reg[k])
            self.invoke_with_control(
                hxyz_gate.bind(b2=self._b2_reg, b1_regs=self._b1_regs, i_regs=self._i_regs),
                self._a_reg[k:k+1], "1")
        self.invoke(bser_block.bind(a=self._a_reg, b2=self._b2_reg, b1_regs=self._b1_regs),
                    inverse=True, label="B_ser\u2020")

    def bind(self, a, b2, b1_regs, i_regs):
        """ bind registers.

            :param a: a register. holds value for k.
            :param b2: b2 register. holds value for l.
            :param b1_regs: b1 registers.
            :param i_regs: i registers.
        """
        return heap.Binding(self, {
            "a": a,
            "b2": b2,
            "b1_regs": b1_regs,
            "i_regs": i_regs
        })

class AParallelBlock(heap.Frame):
    """ The A block (oblivious amplitude amplification) for parallel x-y-z
        version.
    """
    def __init__(self, q_spec: QSpaceTEVSpec, label: str="A_par", allocate=True, build=True):
        super().__init__(label=label)
        self.q_spec = q_spec
        if allocate:
            self.allocate_registers()
            if build:
                self.build_circuit()

    def allocate_registers(self):
        """ allocate registers
        """
        q_spec = self.q_spec
        wfr_spec = q_spec.wfr_spec

        self._a_reg = QuantumRegister(q_spec.k, "a")
        self._b_reg = QuantumRegister(q_spec.num_l_bits, "b")
        self._i_reg = QuantumRegister(wfr_spec.num_coordinate_bits, f"i")
        self.add_param(self._a_reg, self._b_reg, self._i_reg)

    def build_circuit(self):
        """ build the circuit
        """
        qc = self.circuit
        q_spec = self.q_spec
        # negating the overall amplitude is omitted.

        wpar_block = WParallelBlock(q_spec)
        self.invoke(wpar_block.bind(a=self._a_reg, b=self._b_reg, i=self._i_reg))
        rblock = RBlock(q_spec)
        self.invoke(rblock.bind(a=self._a_reg, b=self._b_reg))
        self.invoke(wpar_block.bind(a=self._a_reg, b=self._b_reg, i=self._i_reg),
                    inverse=True, label="W_par\u2020")
        self.invoke(rblock.bind(a=self._a_reg, b=self._b_reg))
        self.invoke(wpar_block.bind(a=self._a_reg, b=self._b_reg, i=self._i_reg))

    def bind(self, a, b, i):
        """ bind registers.

            :param a: a register. holds value for k.
            :param b: b register. holds value for l.
            :param i: i register. holds value for coordinate.
        """
        return heap.Binding(self, {
            "a": a,
            "b": b,
            "i": i
        })


class ASerialBlock(heap.Frame):
    """ The "W" block for oblivious amplitude amplification, serial x-y-z
        version.
    """
    def __init__(self, q_spec: QSpaceTEVSpec, label: str="A_ser", allocate=True, build=True):
        super().__init__(label=label)
        self.q_spec = q_spec
        if allocate:
            self.allocate_registers()
            if build:
                self.build_circuit()

    def allocate_registers(self):
        """ allocate registers
        """
        q_spec = self.q_spec
        wfr_spec = q_spec.wfr_spec

        self._i_regs = []
        self._b1_regs = []
        for dim in range(wfr_spec.dimension):
            axis="xyz"[dim]
            ireg = QuantumRegister(wfr_spec.num_coordinate_bits, f"i{axis}")
            self._i_regs.append(ireg)
            b1reg = QuantumRegister(2, f"b1{axis}")
            self._b1_regs.append(b1reg)
        self.add_param(("i_regs", self._i_regs))
        self.add_local(("b1_regs", self._b1_regs))

        self._a_reg = QuantumRegister(q_spec.k, "a")
        self._b2_reg = QuantumRegister(q_spec.num_l_bits, "b2")
        self.add_local(self._a_reg, self._b2_reg)


    def build_circuit(self):
        """ build the circuit
        """
        qc = self.circuit
        q_spec = self.q_spec
        # negating the overall amplitude is omitted
        wser_block = WSerialBlock(q_spec)
        self.invoke(wser_block.bind(a=self._a_reg, b2=self._b2_reg,
                                    b1_regs=self._b1_regs, i_regs=self._i_regs))
        rblock = RBlock(q_spec)
        self.invoke(rblock.bind(a=self._a_reg, b=self._b2_reg))
        self.invoke(wser_block.bind(a=self._a_reg, b2=self._b2_reg,
                                    b1_regs=self._b1_regs, i_regs=self._i_regs),
                    inverse=True, label="W_ser\u2020")
        self.invoke(rblock.bind(a=self._a_reg, b=self._b2_reg))
        self.invoke(wser_block.bind(a=self._a_reg, b2=self._b2_reg,
                                    b1_regs=self._b1_regs, i_regs=self._i_regs))

    def bind(self, i_regs):
        """ bind registers.

            :param a: a register. holds value for k.
            :param b2: b2 register. holds value for l.
            :param b1_regs: b1 registers.
            :param i_regs: i registers.
        """
        return heap.Binding(self, {
            "i_regs": i_regs
        })

class QSpaceMethodBlock(heap.Frame):
    """ Schrödinger equation integrator in q-space
    """
    def __init__(self,
                 q_spec: QSpaceTEVSpec,
                 tev_spec: spec.TimeEvolutionSpec,
                 ene_spec: energy_initialization.EnergyConfigurationSpec,
                 asy_spec: antisymmetrization.AntisymmetrizationSpec,
                 label="QSpace", allocate=True, build=True, use_motion_block_gates=False):
        super().__init__(label=label)
        self._q_spec = q_spec
        self._tev_spec = tev_spec
        self._ene_spec = ene_spec
        self._asy_spec = asy_spec
        self._ham_spec = tev_spec.ham_spec
        self._disc_spec = tev_spec.disc_spec
        self._wfr_spec = self._ham_spec.wfr_spec
        self._use_motion_block_gates = use_motion_block_gates
        if allocate:
            self.allocate_registers()
            if build:
                with check_time("QSpaceMethodBlock.build_circuit"):
                    self.build_circuit()

    def allocate_registers(self):
        """ allocate """
        wfr_spec = self._wfr_spec
        ene_spec = self._ene_spec
        asy_spec = self._asy_spec
        self._e_index_regs = wfr_spec.allocate_elec_registers()
        self._n_index_regs = wfr_spec.allocate_nucl_registers()
        self.add_param(("eregs", self._e_index_regs),
                       ("nregs", self._n_index_regs))
        self._shuffle = asy_spec.allocate_ancilla_register()
        self.add_param(self._shuffle)
        self._slater_indices = asy_spec.allocate_sigma_regs()
        self.add_param(("slater_indices", self._slater_indices))

        if ene_spec.num_energy_configuration_bits > 0 :
            self._energy_configuration_reg = QuantumRegister(
                ene_spec.num_energy_configuration_bits, "p")
            self.add_param(
                self._energy_configuration_reg
                )

    def build_circuit(self):
        """ build the gates for time evolution.
            There are several variations for this.
        """
        qc = self.circuit
        tev_spec = self._tev_spec

        self._build_initialization_block()

        n_atom_it = tev_spec.num_atom_iterations
        n_elec_it = tev_spec.num_elec_per_atom_iterations
        with qc.for_loop(range(n_atom_it)):
            with qc.for_loop(range(n_elec_it)):
                if tev_spec.should_calculate_electron_motion:
                    self._build_electron_motion_block()
            if tev_spec.should_calculate_nucleus_motion:
                self._build_nuclei_motion_block()

    def _build_initialization_block(self):
        if self._ene_spec.num_energy_configurations > 1:
            self._initialize_with_general_state()
        else:
            self._initialize_with_sd_state()

    def _initialize_with_general_state(self):
        g_block = energy_initialization.GeneralStatePreparationBlock(
            self._ene_spec,
            self._asy_spec)
        with check_time("GeneralStatePreparationBlock.invoke"):
            self.invoke(
                g_block.bind(
                    eregs=self._e_index_regs,
                    nregs=self._n_index_regs,
                    bregs=self._slater_indices,
                    shuffle=self._shuffle,
                    p=self._energy_configuration_reg
                    ))

    def _initialize_with_sd_state(self):
        sd_block = energy_initialization.SlaterDeterminantPreparationBlock(
            self._ene_spec,
            self._asy_spec,
            0)
        logger.info("SlaterDeterminantPreparationBlock.num_qubits=%d", sd_block.circuit.num_qubits)
        with check_time("SlaterDeterminantPreparationBlock.invoke"):
            self.invoke(
                sd_block.bind(
                    eregs=self._e_index_regs,
                    nregs=self._n_index_regs,
                    bregs=self._slater_indices,
                    shuffle=self._shuffle
                )
            )
    
    def _build_electron_motion_block(self):
        if self._use_motion_block_gates:
            elec_motion_block = QspaceElectronMotionBlock(self._tev_spec, self._q_spec)
            logger.info("QspaceElectronMotionBlock.num_qubits = %d", elec_motion_block.circuit.num_qubits)
            with check_time("QspaceElectronMotionBlock.invoke"):
                self.invoke(elec_motion_block.bind(
                    eregs=self._e_index_regs,
                    nregs=self._n_index_regs
                ), invoke_as_instruction=True)
            return
        # don't use the motion block gates.
        elec_motion_block = QspaceElectronMotionBlock(self._tev_spec, self._q_spec, build=False)
        elec_motion_block.build_circuit_on(self)

    def _build_nuclei_motion_block(self):
        if self._use_motion_block_gates:
            nucl_motion_block = QspaceNucleusMotionBlock(self._tev_spec, self._q_spec)
            with check_time("NucleusMotionBlock.invoke"):
                self.invoke(nucl_motion_block.bind(
                    nregs=self._n_index_regs), invoke_as_instruction=True)
            return
        # don't use the motion block gates.
        nucl_motion_block = QspaceNucleusMotionBlock(self._tev_spec, self._q_spec, build=False)
        qc = self.circuit
        nucl_motion_block.build_circuit_on(self)

    def bind(self, eregs, nregs, shuffle, slater_indices, p):
        """ bind registers.

            :param eregs: electron registers.
            :param nregs: nucleus registers.
            :param shuffle: slater ancilla register.
            :param slater_indices: slater indices.
        """
        return heap.Binding(self, {
            "eregs": eregs,
            "nregs": nregs,
            "shuffle": shuffle,
            "slater_indices": slater_indices,
            "p": p
        })


class QspaceElectronMotionBlock(heap.Frame):
    """ H_ep, H_ek

        Electron motion by truncated Taylor expansion
    """
    def __init__(self,
                 tev_spec: spec.TimeEvolutionSpec,
                 q_spec: QSpaceTEVSpec,
                 label=" TEV_e(x)", allocate=True, build=True):
        super().__init__(label=label)
        t1 = time.time()
        self._tev_spec = tev_spec
        self._q_spec = q_spec
        self._ham_spec = tev_spec.ham_spec
        self._disc_spec = tev_spec.disc_spec
        self._wfr_spec = self._ham_spec.wfr_spec
        # registers
        self._e_index_regs: List[List[QuantumRegister]]
        self._n_index_regs: List[List[QuantumRegister]]
        # blocks
        # self._elec_potential_block: hamiltonian.ElectronPotentialBlock
        if allocate:
            self.allocate_registers()
            if build:
                self.build_circuit()
        dt = time.time() - t1
        if dt > LOG_TIME_THRESH:
            logger.info("ElectronMotionBlock() took %d msec", round(dt*1000))

    def allocate_registers(self):
        """ allocate registers """
        wfr_spec = self._wfr_spec
        self._e_index_regs = wfr_spec.allocate_elec_registers()
        self._n_index_regs = wfr_spec.allocate_nucl_registers()
        self.add_param(("eregs", self._e_index_regs),
                       ("nregs", self._n_index_regs))
    
    def build_circuit_on(self, other_frame: heap.Frame):
        stash_circuit = self._circuit
        stash_tmp_allocator = self._temp_allocator
        stash_ancilla_allocator = self._ancilla_allocator
        self._circuit = other_frame.circuit
        self._temp_allocator = other_frame._temp_allocator
        self._ancilla_allocator = other_frame._ancilla_allocator
        self.build_circuit()
        self._circuit = stash_circuit
        self._temp_allocator = stash_tmp_allocator
        self._ancilla_allocator = stash_ancilla_allocator

    def build_circuit(self):
        """ build the gates for time evolution.
            There are several variations for this.
        """
        wfr_spec = self._wfr_spec
        tev_spec = self._tev_spec
        if tev_spec.should_calculate_potential_term and wfr_spec.has_elec_potential_term:
            self._build_elec_potential_step()
        if tev_spec.should_calculate_kinetic_term:
            self._build_elec_kinetic_step()

    def _build_elec_potential_step(self):
        block = self.build_elec_potential_block()
        logger.info("ElectronPotentialBlock.num_qubits = %d", block.circuit.num_qubits)
        with check_time("ElectronPotentialBlock.invoke"):
            self.invoke(block.bind(eregs=self._e_index_regs, nregs=self._n_index_regs))

    def build_elec_potential_block(self, allocate=True, build=True):
        """ build a ElectronPotentialBlock instance."""
        block = hamiltonian.ElectronPotentialBlock(
            self._ham_spec, self._disc_spec, allocate=allocate, build=build)
        return block

    def _build_elec_kinetic_step(self):
        block = self.build_elec_kinetic_block()
        logger.info("ASerialBlock.num_qubits = %d", block.circuit.num_qubits)
        for elec_i in range(self._wfr_spec.num_electrons):
            with check_time("ASerialBlock.invoke"):
                self.invoke(block.bind(
                    i_regs=self._e_index_regs[elec_i]
                ))

    def build_elec_kinetic_block(self):
        block = ASerialBlock(self._q_spec)
        return block

    def bind(self, eregs, nregs):
        """ bind registers.

            :param eregs: electron registers.
            :param nregs: nucleus registers.
        """
        return heap.Binding(self, {
            "eregs": eregs,
            "nregs": nregs
        })

class QspaceNucleusMotionBlock(heap.Frame):
    """ H_np, QFT, H_nk, QFT\dagger """
    def __init__(self,
                 tev_spec: spec.TimeEvolutionSpec,
                 q_spec: QSpaceTEVSpec,
                 label=" TEV_n(x)", allocate=True, build=True):
        super().__init__(label=label)
        self._tev_spec = tev_spec
        self._q_spec = q_spec
        self._ham_spec = tev_spec.ham_spec
        self._disc_spec = tev_spec.disc_spec
        self._wfr_spec = self._ham_spec.wfr_spec
        # registers
        self._n_index_regs: List[List[QuantumRegister]]
        if allocate:
            self.allocate_registers()
            if build:
                with check_time("NucleusMotionBlock.build_circuit"):
                    self.build_circuit()

    def allocate_registers(self):
        """ allocate """
        wfr_spec = self._wfr_spec
        self._n_index_regs = wfr_spec.allocate_nucl_registers()
        self.add_param(("nregs", self._n_index_regs))

    def build_circuit_on(self, other_frame: heap.Frame):
        """ Build the instructions on another compatible quantum circuit."""
        stash_circuit = self._circuit
        stash_tmp_allocator = self._temp_allocator
        stash_ancilla_allocator = self._ancilla_allocator
        self._circuit = other_frame.circuit
        self._temp_allocator = other_frame._temp_allocator
        self._ancilla_allocator = other_frame._ancilla_allocator
        self.build_circuit()
        self._circuit = stash_circuit
        self._temp_allocator = stash_tmp_allocator
        self._ancilla_allocator = stash_ancilla_allocator

    def build_circuit(self):
        """ build the gates for time evolution.
            There are several variations for this.
        """
        if self._wfr_spec.num_moving_nuclei == 0:
            return
        tev_spec = self._tev_spec
        if tev_spec.should_calculate_potential_term:
            self._build_nuclei_potential_step()
        if tev_spec.should_calculate_kinetic_term:
            self._build_nuclei_kinetic_step()

    def _build_nuclei_potential_step(self):
        block = hamiltonian.NucleusPotentialBlock(self._ham_spec, self._disc_spec)
        with check_time("NucleusPotentialBlock.invoke"):
            self.invoke(block.bind(nregs=self._n_index_regs))

    def _build_nuclei_kinetic_step(self):
        block = self.build_nuclei_kinetic_block()
        logger.info("ASerialBlock.num_qubits = %d", block.circuit.num_qubits)
        for nucl_i in range(self._wfr_spec.num_moving_nuclei):
            with check_time("ASerialBlock.invoke"):
                self.invoke(block.bind(
                    i_regs=self._n_index_regs[nucl_i]
                ))

    def build_nuclei_kinetic_block(self):
        block = ASerialBlock(self._q_spec)
        return block

    def bind(self, nregs: List[List[QuantumRegister]]):
        """ bind arguments to the function """
        return heap.Binding(self, {
            "nregs": nregs
        })
