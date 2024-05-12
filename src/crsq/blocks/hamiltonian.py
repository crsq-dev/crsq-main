""" The Hamiltonian blocks implementation
"""

import math
from typing import List
import time
import logging
from contextlib import contextmanager

from qiskit import QuantumRegister
from qiskit.circuit.library import XGate
from crsq import heap, arithmetic as ari
from crsq.arithmetic import ast
from crsq.blocks import wave_function , discretization

logger=logging.getLogger(__name__)
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

class HamiltonianSpec:
    """ Hamiltonian spec parameters
    """
    def __init__(self,
                 wfr_spec: wave_function.WaveFunctionRegisterSpec,
                 nuclei_data: list[dict]):
        self._wfr_spec = wfr_spec
        # bit size for constant in V(x)
        self._should_apply_potential_to_phase = True
        self._should_revert_potential_ancilla_value = True
        self._should_mask_potential_singularity = False
        self._nuclei_data = nuclei_data
        if wfr_spec.num_moving_nuclei ==0:
            self._max_particle_charge = 1.0
            self._num_v_numerator_int_bits = 1
        else:
            self._max_particle_charge = max([ent["charge"] for ent in nuclei_data])
            # qA = self._max_particle_charge
            self._num_v_numerator_int_bits = 1
            # self._num_v_numerator_int_bits = 1+int(2*math.log2(qA))
        logger.info("HamiltonianSpec: max_particle_charge = %f", self._max_particle_charge)
        logger.info("HamiltonianSpec: num_v_numerator_int_bits = %d",
                    self._num_v_numerator_int_bits)

    def set_should_apply_potential_to_phase(self, flag):
        """ set flag to enable function"""
        self._should_apply_potential_to_phase = flag
        logger.info("HamiltonianSpec: should_apply_potential_to_phase = %d", flag)

    def set_should_revert_potential_ancilla_value(self, flag):
        """ set flag to enable function"""
        self._should_revert_potential_ancilla_value = flag
        logger.info("HamiltonianSpec: should_revert_potential_ancilla_value = %d", flag)

    def set_should_mask_potential_singularity(self, flag):
        """ set flag to mask potential division by zero """
        self._should_mask_potential_singularity = flag
        logger.info("HamiltonianSpec: should_mask_potential_singularity = %d", flag)

    @property
    def wfr_spec(self) -> wave_function.WaveFunctionRegisterSpec:
        """ get wfr_spec
        """
        return self._wfr_spec

    @property
    def num_v_numerator_int_bits(self) -> int:
        """ get numerator int bits
        """
        return self._num_v_numerator_int_bits

    @property
    def should_apply_potential_to_phase(self) -> bool:
        """ returns flag """
        return self._should_apply_potential_to_phase

    @property
    def should_revert_potential_ancilla_value(self) -> bool:
        """ returns flag """
        return self._should_revert_potential_ancilla_value

    @property
    def nuclei_data(self) -> list[dict]:
        """ returns data """
        return self._nuclei_data

    @property
    def should_mask_potential_singularity(self):
        """ returns flag """
        return self._should_mask_potential_singularity

class PotentialBlockBase(heap.Frame):
    """ Common base class for Potential block classes"""

    def __init__(self, ham_spec: HamiltonianSpec,
                 disc_spec: discretization.DiscretizationSpec, label):
        super().__init__(label=label)
        self._ham_spec = ham_spec
        self._wfr_spec = ham_spec.wfr_spec
        self._disc_spec = disc_spec
        self._diff_zero_ancilla_reg: QuantumRegister
        self._diff_stash_reg: QuantumRegister

    def _rotate_phase_by_register(self, reg: QuantumRegister, charge: float):
        n = self._wfr_spec.num_coordinate_bits
        m = self._ham_spec.num_v_numerator_int_bits
        numerator_frac_bits = n + 1 - m
        qc = self.circuit
        delta_t = self._disc_spec.delta_t
        delta_q = self._wfr_spec.delta_q
        for i in range(reg.size):
            digit_weight = 2**(i - numerator_frac_bits)
            eta = charge * delta_t / delta_q
            qc.p(-eta*digit_weight, reg[i])

    def _allocate_singularity_exchange_registers(self):
        if not self._ham_spec.should_mask_potential_singularity:
            return
        diff_is_0 = QuantumRegister(1, " diff_is_0")
        self._diff_zero_ancilla_reg = diff_is_0
        wfr_spec = self._wfr_spec
        ham_spec = self._ham_spec
        stash_reg_size = wfr_spec.num_coordinate_bits + ham_spec.num_v_numerator_int_bits
        self._diff_stash_reg = QuantumRegister(stash_reg_size, "diff_stash")
        self.add_local(self._diff_zero_ancilla_reg, self._diff_stash_reg)

    def _build_set_diff0_reg(self, dist_reg: QuantumRegister):
        """ emit a CNOT on diff_zero_ancilla_reg controlled by
            all bits in qregs = 0
        """
        num_ctrl_bits = dist_reg.size
        ctrl_str = "0" * num_ctrl_bits
        ctrl_bits = dist_reg[:]
        target_bits = ctrl_bits + self._diff_zero_ancilla_reg[:]
        qc = self.circuit
        qc.append(XGate().control(num_ctrl_bits,
                                  ctrl_state=ctrl_str),
                                  target_bits)

    def _build_stash_quotient_on_diff0(self, quot_reg: QuantumRegister, reverse=False):
        qc = self.circuit
        assert quot_reg.size == self._diff_stash_reg.size
        if not reverse:
            for k in range(quot_reg.size):
                qc.cswap(self._diff_zero_ancilla_reg[0], quot_reg[k], self._diff_stash_reg[k])
        else:
            for k in reversed(range(quot_reg.size)):
                qc.cswap(self._diff_zero_ancilla_reg[0], quot_reg[k], self._diff_stash_reg[k])

class ElectronPotentialBlock(PotentialBlockBase):
    """ electron-electron and electron-nucleus potential term
    """

    def __init__(self, ham_spec: HamiltonianSpec,
                 disc_spec: discretization.DiscretizationSpec, allocate=True, build=True):
        super().__init__(ham_spec, disc_spec, label='  Θ_ep')
        self._eregs: List[List[List[QuantumRegister]]] = []
        self._nregs: List[List[List[QuantumRegister]]] = []
        self._vx_const_numerator_reg: QuantumRegister
        if allocate:
            self.allocate_registers()
            if build:
                with check_time("ElectronPotentialBlock.build_circuit"):
                    self.build_circuits()

    def allocate_registers(self):
        """ allocate registers
        """
        self._eregs = self._wfr_spec.allocate_elec_registers()
        self._nregs = self._wfr_spec.allocate_nucl_registers()
        self.add_param(('eregs', self._eregs), ('nregs', self._nregs))
        n = self._wfr_spec.num_coordinate_bits
        self._vx_const_numerator_reg = QuantumRegister(n + 1, "one")
        self.add_local(self._vx_const_numerator_reg)
        self._allocate_singularity_exchange_registers()

    def build_circuits(self):
        """ build circuits
        """
        scope = ast.new_scope(self)
        wfr_spec = self._wfr_spec
        ham_spec = self._ham_spec

        n = wfr_spec.num_coordinate_bits
        m = ham_spec.num_v_numerator_int_bits
        numerator_frac_bits = n + 1 - m
        numerator_val = int(1.0 * (2**numerator_frac_bits))
        qc = self.circuit
        ari.set_value(qc, self._vx_const_numerator_reg, numerator_val)
        ast_numerator = scope.register(self._vx_const_numerator_reg, numerator_frac_bits)
        dim = wfr_spec.dimension

        # prepare AST registers for electron indices
        ast_eregs: list[list[ast.QuantumValue]] = []
        for ie in range(wfr_spec.num_electrons):
            dims: list = []
            for d in range(dim):
                dims.append(scope.register(self._eregs[ie][d], signed=True))
            ast_eregs.append(dims)

        # prepare AST registers for nucleus indices
        ast_nregs = []
        for ia in range(wfr_spec.num_moving_nuclei):
            dims = []
            for d in range(dim):
                dims.append(scope.register(self._nregs[ia][d], signed=True))
            ast_nregs.append(dims)

        # prepare AST constant nodes for stationary nucleus indices.
        nuclei_data = self._ham_spec.nuclei_data
        num_orbitals = wfr_spec.num_moving_nuclei
        for ia in range(wfr_spec.num_stationary_nuclei):
            ndata = nuclei_data[num_orbitals + ia]
            dims = []
            for d in range(dim):
                pos = ndata["pos"]  # tuple or float
                if isinstance(pos, tuple):
                    q_d = ndata["pos"][d]
                elif dim == 1:
                    q_d = ndata["pos"]
                else:
                    raise ValueError("pos is not tuple")
                dims.append(scope.constant(q_d, n, signed=True))
            ast_nregs.append(dims)

        self._build_elec_elec_potential_terms(
            scope, ast_numerator, ast_eregs)

        self._build_elec_nucl_potential_terms(
            scope, ast_numerator, ast_eregs, ast_nregs)

        scope.close()

        # reset constant values
        if ham_spec.should_revert_potential_ancilla_value:
            ari.set_value(self.circuit, self._vx_const_numerator_reg, numerator_val)

    def _build_elec_elec_potential_terms(
            self,
            scope: ast.Scope,
            ast_numerator: ast.QuantumValue,
            ast_eregs: list[list[ast.QuantumValue]]):
        """ build e-e potential terms
        """

        wfr_spec = self._wfr_spec
        ham_spec = self._ham_spec
        should_mask_singularity = ham_spec.should_mask_potential_singularity
        for ie in range(wfr_spec.num_electrons):
            for ih in range(ie + 1, wfr_spec.num_electrons):
                t1 = time.time()
                ast_dist: ast.QuantumValue
                if wfr_spec.dimension == 1:
                    ast_e = ast_eregs[ie][0]
                    ast_h = ast_eregs[ih][0]
                    ast_e -= ast_h  # diff
                    ast_dist =  scope.abs(ast_e)
                else:
                    ast_squares = []
                    for d in range(wfr_spec.dimension):
                        ast_e = ast_eregs[ie][d]
                        ast_h = ast_eregs[ih][d]
                        ast_e -= ast_h  # diff
                        ast_sq = scope.square(ast_e)
                        ast_squares.append(ast_sq)
                        if d > 0:
                            ast_squares[0] += ast_squares[d]  # sum
                    ast_dist = scope.square_root(ast_squares[0])

                if ast_dist.total_bits < ast_numerator.total_bits:
                    pad_denominator = True
                else:
                    pad_denominator = False
                if pad_denominator:
                    total_bits = ast_dist.total_bits
                    fraction_bits = ast_dist.fraction_bits
                    high_bits = self.allocate_ancilla_bits(1, "msb")
                    ast_dist2 = ast_dist.adjust_precision(total_bits+1, fraction_bits, new_high_bits=high_bits)
                    ast_quotient = ast_numerator / ast_dist2 # inv
                else:
                    ast_quotient = ast_numerator / ast_dist
                self.alias_regs['elec_elec_potential_quotient'] = ast_quotient.register
                scope.build_circuit()
                qq = -1.0*-1.0  # product of charges
                # apply ratio to phase
                if ham_spec.should_apply_potential_to_phase:
                    # hide 1/r when r == 0
                    if should_mask_singularity:
                        self._build_set_diff0_reg(ast_dist.register)
                        self._build_stash_quotient_on_diff0(ast_quotient.register)
                    self._rotate_phase_by_register(ast_quotient.register, qq)
                if ham_spec.should_revert_potential_ancilla_value:
                    # restore the hidden 1/r
                    if should_mask_singularity:
                        self._build_stash_quotient_on_diff0(ast_quotient.register, reverse=True)
                        self._build_set_diff0_reg(ast_dist.register)
                    scope.build_inverse_circuit()
                scope.clear_operations()
                if pad_denominator:
                    self.free_ancilla_bits(high_bits)
                dt = time.time() - t1
                if dt > LOG_TIME_THRESH:
                    logger.info("  Vee(%d,%d) done. %d msec", ie, ih, round(dt*1000))

    def _build_elec_nucl_potential_terms(
            self,
            scope: ast.Scope,
            ast_numerator: ast.QuantumValue,
            ast_eregs: list[list[ast.QuantumValue]],
            ast_nregs: list[list[ast.QuantumValue]]):

        wfr_spec = self._wfr_spec
        ham_spec = self._ham_spec
        nuclei_data = ham_spec.nuclei_data
        should_mask_singularity = ham_spec.should_mask_potential_singularity
        for ia in range(wfr_spec.num_nuclei):
            for ie in range(wfr_spec.num_electrons):
                t1 = time.time()
                qe = -1.0
                qA = nuclei_data[ia]['charge']
                ast_dist: ast.QuantumValue
                if wfr_spec.dimension == 1:
                    ast_e = ast_eregs[ie][0]
                    ast_n = ast_nregs[ia][0]
                    ast_e -= ast_n  # diff
                    ast_dist = scope.abs(ast_e)
                else:
                    ast_squares = []
                    for d in range(wfr_spec.dimension):
                        ast_e = ast_eregs[ie][d]
                        ast_n = ast_nregs[ia][d]
                        ast_e -= ast_n  # diff
                        ast_sq = scope.square(ast_e)
                        ast_squares.append(ast_sq)
                        if d > 0:
                            ast_squares[0] += ast_squares[d]  # sum
                    ast_dist = scope.square_root(ast_squares[0])

                if ast_dist.total_bits < ast_numerator.total_bits:
                    pad_denominator = True
                else:
                    pad_denominator = False
                if pad_denominator:
                    total_bits = ast_dist.total_bits
                    fraction_bits = ast_dist.fraction_bits
                    high_bits = self.allocate_ancilla_bits(1, "msb")
                    ast_dist2 = ast_dist.adjust_precision(total_bits+1, fraction_bits, new_high_bits=high_bits)
                    ast_quotient = ast_numerator / ast_dist2 # inv
                else:
                    ast_quotient = ast_numerator / ast_dist
                self.alias_regs['elec_nucl_potential_quotient'] = ast_quotient.register
                scope.build_circuit()
                qq = qe*qA  # product of charges
                # apply ratio to phase
                if ham_spec.should_apply_potential_to_phase:
                    # hide 1/r when r == 0
                    if should_mask_singularity:
                        self._build_set_diff0_reg(ast_dist.register)
                        self._build_stash_quotient_on_diff0(ast_quotient.register)
                    self._rotate_phase_by_register(ast_quotient.register, qq)
                if ham_spec.should_revert_potential_ancilla_value:
                    # restore the hidden 1/r
                    if should_mask_singularity:
                        self._build_stash_quotient_on_diff0(ast_quotient.register, reverse=True)
                        self._build_set_diff0_reg(ast_dist.register)
                    scope.build_inverse_circuit()
                scope.clear_operations()
                if pad_denominator:
                    self.free_ancilla_bits(high_bits)
                dt = time.time() - t1
                if dt > LOG_TIME_THRESH:
                    logger.info("  VeA(%d,%d) done. %d ms", ie, ia, round(dt*1000))

    def bind(self,
             eregs: List[List[List[QuantumRegister]]],
             nregs: List[List[List[QuantumRegister]]]
             ) -> heap.Binding:
        """ produce binding
        """
        return heap.Binding(self, {
            "eregs": eregs,
            "nregs": nregs
        })

class NucleusPotentialBlock(PotentialBlockBase):
    """ nucleus-nucleus potential term
    """

    def __init__(self, ham_spec: HamiltonianSpec,
                 disc_spec: discretization.DiscretizationSpec,
                 allocate=True, build=True):
        super().__init__(ham_spec, disc_spec, label='  Θ_np')
        self._nregs: List[List[List[QuantumRegister]]] = []
        self._vx_const_numerator_reg: QuantumRegister
        if allocate:
            self.allocate_registers()
            if build:
                with check_time("NucleusPotentialBlock.build_circuits"):
                    self.build_circuits()

    def allocate_registers(self):
        """ allocate registers
        """
        self._nregs = self._wfr_spec.allocate_nucl_registers()
        self.add_param(('nregs', self._nregs))
        n = self._wfr_spec.num_coordinate_bits
        self._vx_const_numerator_reg = QuantumRegister(n + 1, "one")
        self.add_local(self._vx_const_numerator_reg)
        self._allocate_singularity_exchange_registers()

    def build_circuits(self):
        """ build circuits
        """
        scope = ast.new_scope(self)
        wfr_spec = self._wfr_spec
        ham_spec = self._ham_spec
        dim = wfr_spec.dimension

        n = wfr_spec.num_coordinate_bits
        m = ham_spec.num_v_numerator_int_bits
        numerator_frac_bits = n + 1 - m
        numerator_val = int(1.0 * (2**numerator_frac_bits))
        qc = self.circuit
        ari.set_value(qc, self._vx_const_numerator_reg, numerator_val)
        ast_numerator = scope.register(self._vx_const_numerator_reg, numerator_frac_bits)

        # prepare AST registers for nucleus indices
        ast_nregs = []
        for ia in range(wfr_spec.num_moving_nuclei):
            dims: List = []
            for d in range(dim):
                dims.append(scope.register(self._nregs[ia][d], signed=True))
            ast_nregs.append(dims)

        # prepare AST constant nodes for stationary nucleus indices.
        nuclei_data = self._ham_spec.nuclei_data
        num_orbitals = wfr_spec.num_moving_nuclei
        for ia in range(wfr_spec.num_stationary_nuclei):
            ndata = nuclei_data[num_orbitals + ia]
            dims = []
            for d in range(dim):
                pos = ndata["pos"]
                if isinstance(pos, tuple):
                    q_d = pos[d]
                elif dim == 1:
                    q_d = pos
                else:
                    raise ValueError("pos is not tuple")
                dims.append(scope.constant(q_d, n))
            ast_nregs.append(dims)

        self._build_nucl_nucl_potential_terms(
            scope, ast_numerator, ast_nregs)

        scope.close()

    def _build_nucl_nucl_potential_terms(
            self,
            scope: ast.Scope,
            ast_numerator: ast.QuantumValue,
            ast_nregs: list[list[ast.QuantumValue]]):

        wfr_spec = self._wfr_spec
        ham_spec = self._ham_spec
        nuclei_data = ham_spec.nuclei_data
        should_mask_singularity = ham_spec.should_mask_potential_singularity

        for ia in range(wfr_spec.num_moving_nuclei):
            qA = nuclei_data[ia]['charge']
            for ib in range(ia+1, wfr_spec.num_moving_nuclei):
                t1 = time.time()
                qB = nuclei_data[ib]['charge']
                ast_dist: ast.QuantumValue
                if wfr_spec.dimension == 1:
                    ast_a = ast_nregs[ia][0]
                    ast_b = ast_nregs[ib][0]
                    ast_a -= ast_b  # diff
                    ast_dist =  scope.abs(ast_a)
                else:
                    ast_squares = []
                    for d in range(wfr_spec.dimension):
                        ast_a = ast_nregs[ia][d]
                        ast_b = ast_nregs[ib][d]
                        ast_a -= ast_b  # diff
                        ast_sq = scope.square(ast_a)
                        ast_squares.append(ast_sq)
                        if d > 0:
                            ast_squares[0] += ast_squares[d]  # sum
                    ast_dist = scope.square_root(ast_squares[0])

                if ast_dist.total_bits < ast_numerator.total_bits:
                    pad_denominator = True
                else:
                    pad_denominator = False
                if pad_denominator:
                    total_bits = ast_dist.total_bits
                    fraction_bits = ast_dist.fraction_bits
                    high_bits = self.allocate_ancilla_bits(1, "msb")
                    ast_dist2 = ast_dist.adjust_precision(total_bits+1, fraction_bits, new_high_bits=high_bits)
                    ast_quotient = ast_numerator / ast_dist2 # inv
                else:
                    ast_quotient = ast_numerator / ast_dist
                scope.build_circuit()
                qq = qA*qB  # product of charges
                # apply ratio to phase
                if ham_spec.should_apply_potential_to_phase:
                    if should_mask_singularity:
                        self._build_set_diff0_reg(ast_dist.register)
                        self._build_stash_quotient_on_diff0(ast_quotient.register)
                    self._rotate_phase_by_register(ast_quotient.register, qq)
                if ham_spec.should_revert_potential_ancilla_value:
                    # restore the hidden 1/r
                    if should_mask_singularity:
                        self._build_stash_quotient_on_diff0(ast_quotient.register, reverse=True)
                        self._build_set_diff0_reg(ast_dist.register)
                    scope.build_inverse_circuit()
                scope.clear_operations()
                if pad_denominator:
                    self.free_ancilla_bits(high_bits)
                dt = time.time() - t1
                if dt > LOG_TIME_THRESH:
                    logger.info("  VAA(%d,%d) done. %d msec", ia, ib, round(dt*1000))

    def bind(self,
             nregs: List[List[List[QuantumRegister]]]
             ) -> heap.Binding:
        """ produce binding
        """
        return heap.Binding(self, {
            "nregs": nregs
        })


class ElectronKineticBlock(heap.Frame):
    """ Electron Kinetic block"""

    def __init__(self,
                 wfr_spec: wave_function.WaveFunctionRegisterSpec,
                 disc_spec: discretization.DiscretizationSpec,
                 allocate=True,
                 build=True):
        super().__init__(label="  Θ_ek")
        self._wfr_spec = wfr_spec
        self._disc_spec = disc_spec
        self._eregs: List[List[List[QuantumRegister]]] = []
        if allocate:
            self.allocate_registers()
            if build:
                with check_time("ElectronKineticBlock.build_circuit"):
                    self.build_circuit()

    def allocate_registers(self):
        """ allocate registers """
        self._eregs = self._wfr_spec.allocate_elec_registers()
        self.add_param(('eregs', self._eregs))

    def build_circuit(self):
        """ build circuit """
        wfr_spec = self._wfr_spec
        disc_spec = self._disc_spec
        qc = self.circuit
        m_e = 1.0
        gamma = wfr_spec.delta_k * wfr_spec.delta_k * disc_spec.delta_t / (2 * m_e)
        m = wfr_spec.num_coordinate_bits
        for ie in range(wfr_spec.num_electrons):
            for d in range(wfr_spec.dimension):
                ereg = self._eregs[ie][d]
                # Σ_j p_j^2 * 2**(2j)
                for j in range(m):
                    qc.p(-gamma * 2**(2*j), ereg[j])
                for k in reversed(range(m-1)):
                    qc.cp(gamma * 2**(m+k), ereg[m-1], ereg[k])
                for j in reversed(range(1, m-1)):
                    for k in reversed(range(j)):
                        qc.cp(-gamma * 2**(j+k+1), ereg[j], ereg[k])

    def bind(self, eregs: List[List[List[QuantumRegister]]]):
        """ produce binding """
        return heap.Binding(self, {
            "eregs": eregs
        })

class NucleusKineticBlock(heap.Frame):
    """ Electron Kinetic block"""

    def __init__(self,
                 wfr_spec: wave_function.WaveFunctionRegisterSpec,
                 disc_spec: discretization.DiscretizationSpec,
                 ham_spec: HamiltonianSpec,
                 allocate = True,
                 build = True):
        super().__init__(label="  Θ_nk")
        self._wfr_spec = wfr_spec
        self._disc_spec = disc_spec
        self._ham_spec = ham_spec
        self._nregs: List[List[List[QuantumRegister]]] = []
        if allocate:
            self.allocate_registers()
            if build:
                with check_time("NucleusKineticBlock.build_circuit"):
                    self.build_circuit()

    def allocate_registers(self):
        """ allocate registers """
        self._nregs = self._wfr_spec.allocate_nucl_registers()
        self.add_param(('nregs', self._nregs))

    def build_circuit(self):
        """ build circuit """
        wfr_spec = self._wfr_spec
        disc_spec = self._disc_spec
        ham_spec = self._ham_spec
        qc = self.circuit
        m = wfr_spec.num_coordinate_bits
        # this loop is for nuclei orbitals only.
        # there is no need for stationary nuclei.
        for ia in range(wfr_spec.num_moving_nuclei):
            m_n = ham_spec.nuclei_data[ia]['mass']
            gamma = wfr_spec.delta_k * wfr_spec.delta_k * disc_spec.delta_t / (2 * m_n)
            for d in range(wfr_spec.dimension):
                nreg = self._nregs[ia][d]
                # Σ_j p_j^2 * 2**(2j)
                for j in range(m):
                    qc.p(-gamma * 2**(2*j), nreg[j])
                for k in reversed(range(m-1)):
                    qc.cp(gamma * 2**(m+k), nreg[m-1], nreg[k])
                for j in reversed(range(1, m-1)):
                    for k in reversed(range(j)):
                        qc.cp(-gamma * 2**(j+k+1), nreg[j], nreg[k])

    def bind(self, nregs: List[List[List[QuantumRegister]]]):
        """ produce binding """
        return heap.Binding(self, {
            "nregs": nregs
        })
