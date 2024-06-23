""" First quantization integrator
"""

import math
import time
import logging
from typing import Any
from contextlib import contextmanager

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT

import crsq.slater as slater
import crsq_arithmetic.ast as ast
import crsq_arithmetic as ari
import crsq_heap.heap as heap
from crsq.blocks import (
    wave_function,
    hamiltonian,
    discretization,
    time_evolution,
    energy_initialization,
    antisymmetrization,
    qft)

logger = logging.getLogger(__name__)
LOG_TIME_THRESH=1

@contextmanager
def check_time(label:str):
    """ context manager to check time.
        :param label: label used for logs.
    """
    t1 = time.time()
    yield
    dt = time.time() - t1
    if dt > LOG_TIME_THRESH:
        logger.info("%s took %d msec", label, round(dt*1000))


class IntegratorFrame(heap.Frame):
    """ Container for a quantum circuit and all relevant registers.
        This is used for both holding registers for the integrator,
        and holding registers for a custom gate that works as a
        submodule of the integrator
    """

    def __init__(self, label=''):
        super().__init__(label=label)
        # if breg_frame is None:
        #     areg_frame = slater.ARegisterFrame()
        #     breg_frame = slater.BRegisterFrame(areg_frame)

        # self.breg_frame = breg_frame

        self.e_index_regs: list[list[QuantumRegister]] = [] # coordinate index for electrons
        self.n_index_regs: list[list[QuantumRegister]] = [] # coordinate index for nuclei
        self.energy_configuration_reg: QuantumRegister = None
        self.energy_configuration_ancilla_reg: QuantumRegister = None
        self.vx_const_numerator_reg: QuantumRegister = None
        self.slater_bregs: list[QuantumRegister] = []
        self.slater_ancilla: QuantumRegister = None


    def set_elec_index_regs(self, regs: list[list[QuantumRegister]]):
        """ Set the electron index registers """
        self.e_index_regs = regs
        for n in regs:
            self.add_param(*n)

    def set_nuclei_index_regs(self, regs: list[list[QuantumRegister]]):
        """ Set the nuclei index registers

            :param regs: [[n0x, n0y, n0z], [n1x, n1y, n1z], ...]
        """
        self.n_index_regs = regs
        for n in regs:
            self.add_param(*n)

    def set_slater_regs(self, bregs, ancilla):
        """ store the slater determinant related registers"""
        self.slater_bregs = bregs
        self.add_param(*bregs)
        # the slater ancilla bit is used only when num_electrons >= 3
        self.slater_ancilla = ancilla
        if ancilla is not None:
            self.add_param(ancilla)

    def set_energy_configuration_regs(self,
                                      reg: QuantumRegister,
                                    #   ancilla_reg: QuantumRegister
                                      ):
        """ Set the energy configuration register """
        self.energy_configuration_reg = reg
        # self.energy_configuration_ancilla_reg = ancilla_reg
        # self.add_param(reg, ancilla_reg)
        self.add_param(reg)

    def set_time_evolution_work_registers(
            self, vx_const_numerator_reg: QuantumRegister):
        """ Set the work registers for V(x) calculation """
        self.vx_const_numerator_reg = vx_const_numerator_reg
        self.add_local(vx_const_numerator_reg)

class FirstQIntegrator:
    """ First quantization integrator.
        Time evolution (integration) of the time dependent
        Schrödinger equation according to Suzuki-Trotter method.
    """

    def __init__(self):
        """ The constructor will set default values for member variables """
        self.dimension = 1
        self.coordinate_bits = 8
        self.space_length = 4.0
        self.max_particle_charge = 1
        self.num_electrons = 0
        self.num_moving_nuclei = 0
        self.num_nucleus_orbitals = 0
        self.num_moving_nuclei = 0
        self.num_stationary_nuclei = 0
        self.num_energy_configurations = 1
        self.energy_configuration_weights = [ 1.0 ]
        self.should_initialize_energy = True
        self.should_use_motion_block_gates = False
        self.should_calculate_electron_motion = True
        self.should_calculate_nucleus_motion = True
        self.should_calculate_kinetic_term = True
        self.should_calculate_potential_term = True
        self.should_apply_qft = True
        self.should_revert_potential_ancilla_value = True
        self.should_apply_potential_to_phase = True
        self.should_use_spin = False
        self.should_mask_potential_singularity = False
        self.num_atom_iterations = 1
        self.num_elec_per_atom_iterations = 1836
        self.delta_t = 1.0
        self.antisymmetrization_method = 3
        self.evo_spec: time_evolution.TimeEvolutionSpec
        self.wfr_spec: wave_function.WaveFunctionRegisterSpec
        self.ham_spec: hamiltonian.HamiltonianSpec
        self.disc_spec: discretization.DiscretizationSpec
        self.asy_spec: antisymmetrization.AntisymmetrizationSpec
        self.ene_spec: energy_initialization.EnergyConfigurationSpec
        # calculated values:
        self.delta_x: float
        self.delta_k: float
        self.nuclei_data: list[dict[str,float]]
        self.num_orbital_index_bits: int
        self.energy_configuration_bits: int
        # initial_electron_orbitals[energy_state_index][electron][dimension][coordinate]
        self.initial_electron_orbitals: list[list[list[list[float]]]]
        self.initial_nucleus_orbitals: list[list[list[list[float]]]]
        self.v_numerator_int_bits: int
        self.k: float
        self.frame: IntegratorFrame
        self.shuffle_frame: slater.BRegisterFrame
        # progress flags
        self.size_calculation_done = False
        self.register_allocation_done = False
        self.circuit_is_built = False
        self.nuclei_data_is_set = False
        # sub structures
        self.time_evolution_block: time_evolution.SuzukiTrotterMethodBlock
        self.slater_areg: slater.ARegister
        self.slater_breg: slater.BRegister

    def circuit(self) -> QuantumCircuit:
        """ returns the currently held circuit """
        return self.frame.circuit

    def set_dimension(self, d: int):
        """ Set the dimension(1-3) of coordinates """
        self.dimension = d

    def set_space_length(self, x: float):
        """ Set the length (in atomic units) along one dimension of the simulated space """
        self.space_length = x

    def set_coordinate_bits(self, coordinate_bits: int):
        """ Set the number of bits used for spatial/frequency coordinates per dimension"""
        self.coordinate_bits = coordinate_bits

    def set_delta_t(self, delta_t: float):
        """ Set the time resolution in atomic unis."""
        self.delta_t = delta_t

    def set_antisymmetrization_method(self, method: int):
        """ 1: conventional (with shuffle), 2: unary encoding (without shuffle)"""
        self.antisymmetrization_method = method

    def set_num_energy_configurations(self, n: int):
        """ Set the number(count) of energy configurations that will be
            used to form the initial thermal pure state
        """
        self.num_energy_configurations = n

    def set_energy_configuration_weights(self, weight_list: list):
        """ Set a list of weight values for the series of energy configurations
        """
        self.energy_configuration_weights = weight_list

    def set_max_particle_charge(self, charge: int):
        """ Set the maximum value of a particle charge"""
        self.max_particle_charge = charge

    def set_num_particles(self, num_electrons: int,
                          num_moving_nuclei: int,
                          num_stationary_nuclei: int = 0):
        """ Set the number of electrons and nuclei """
        self.num_electrons = num_electrons
        self.num_moving_nuclei = num_moving_nuclei
        self.num_stationary_nuclei = num_stationary_nuclei

    def set_use_spin(self, should_use_spin: bool):
        """ set use_spin """
        self.should_use_spin = should_use_spin

    def set_num_atom_iterations(self, num: int):
        """ Set the number of nucleus-scale time iteration steps to run."""
        self.num_atom_iterations = num

    def set_num_elec_per_atom_iterations(self, num: int):
        """ Set the number of electron-scale time iteration steps
            per 1 atom step
        """
        self.num_elec_per_atom_iterations = num

    def set_initialize_energy(self, flag: bool):
        """ Turn on/off state initialization """
        self.should_initialize_energy = flag

    def set_use_motion_block_gates(self, flag: bool):
        """ Use gates for big motion block components.
            Not good for drawing circuit diagrams.
        """
        self.should_use_motion_block_gates = flag

    def set_calculate_electron_motion(self, flag: bool):
        """ Turn on/off electron motion calculation """
        self.should_calculate_electron_motion = flag

    def set_calculate_nucleus_motion(self, flag: bool):
        """ Turn on/off nucleus motion calculation """
        self.should_calculate_nucleus_motion = flag

    def set_calculate_kinetic_term(self, flag: bool):
        """ Turn on/off kinetic term calculation """
        self.should_calculate_kinetic_term = flag

    def set_calculate_potential_term(self, flag: bool):
        """ Turn on/off potential term calculation """
        self.should_calculate_potential_term = flag

    def set_mask_potential_singularity(self, flag: bool):
        """ Turn on/off potential singularity masking logic for same spin only models. """
        self.should_mask_potential_singularity = flag

    def set_apply_qft(self, flag: bool):
        """ Turn on/off QFT/iQFT """
        self.should_apply_qft = flag

    def set_apply_potential_to_phase(self, flag: bool):
        """ Turn on/off apply potential value to phase"""
        self.should_apply_potential_to_phase = flag

    def set_revert_potential_ancilla_value(self, flag: bool):
        """ Turn on/off revert potential ancilla value """
        self.should_revert_potential_ancilla_value = flag

    def set_initial_electron_orbitals(self, orbitals: list[list[list[list[float]]]]):
        """ Set data for initial electron orbitals
            orbitals[energy_config][electron][dimension][coord] = amplitude
            
            amplitude is a relative value within each dimension.
        """
        self.initial_electron_orbitals = orbitals

    def set_initial_nucleus_orbitals(self, orbitals: list[list[list[list[float]]]]):
        """ Set data for initial electron orbitals
            orbitals[energy_config][nucleus][dimension][coord] = amplitude
            
            amplitude is a relative value within each dimension.
        """
        self.initial_nucleus_orbitals = orbitals
        self.num_nucleus_orbitals = len(orbitals[0])

    def make_dummy_orbitals(self):
        """ make dummy orbital data for the initial wave function.
        """
        self._make_dummy_electron_orbitals()
        self._make_dummy_nucleus_orbitals()

    def _make_dummy_electron_orbitals(self):
        if self.should_initialize_energy:
            n = self.coordinate_bits
            wfr_spec = wave_function.WaveFunctionRegisterSpec(
                self.dimension, n, self.space_length, self.num_electrons,
                self.num_moving_nuclei,self.num_stationary_nuclei,
                self.should_use_spin)
            configs = wave_function.make_test_elec_orbitals(
                wfr_spec, self.num_energy_configurations)
        else:
            configs = wave_function.make_null_elec_orbitals(self.num_energy_configurations)
        self.initial_electron_orbitals = configs

    def _make_dummy_nucleus_orbitals(self):
        if self.should_initialize_energy:
            n = self.coordinate_bits
            wfr_spec = wave_function.WaveFunctionRegisterSpec(
                self.dimension, n, self.space_length, self.num_electrons,
                self.num_moving_nuclei, self.num_stationary_nuclei,
                self.should_use_spin)
            configs = wave_function.make_test_nucl_orbitals(
                wfr_spec, self.num_energy_configurations)
        else:
            configs = wave_function.make_null_nucl_orbitals(self.num_energy_configurations)
        self.initial_nucleus_orbitals = configs

    def make_dummy_nuclei_data(self):
        """ make dummy nuclei charge and mass data"""
        s = []
        for _a in range(self.num_moving_nuclei):
            ent: dict[str,Any] = {
                "mass": 1680.0,
                "charge": 1.0
            }
            s.append(ent)
        for _a in range(self.num_stationary_nuclei):
            ent = {
                "mass": 1680.0,
                "charge": 1.0,
                "pos": tuple(0 for _ in range(self.dimension))
            }
            s.append(ent)
        self.nuclei_data = s
        self.nuclei_data_is_set = True

    def set_nuclei_data(self, nuclei_data: list[dict]):
        """ Set an array of nuclei charge and mass.

            :param nuclei_data: list of the form [ { "mass": 1, "charge": 1, "pos": (0,0,0) }, ... ]
            
        """
        self.nuclei_data = nuclei_data
        assert len(nuclei_data) == self.num_moving_nuclei + self.num_stationary_nuclei
        self.nuclei_data_is_set = True

    def calculate_sizes(self):
        """ Calculate the number of bits required """
        assert not self.size_calculation_done
        n = self.coordinate_bits
        delta_x = self.space_length / (2**n)
        self.delta_x = delta_x
        # wave number resolution
        self.delta_k = 2*math.pi / (2**n)
        # bit size for constant in V(x)
        qA = self.max_particle_charge
        self.v_numerator_int_bits = 1+int(2*math.log2(qA))
        # wave number
        k = 2*math.pi/(2**n)
        self.k = k
        # bit size for constant in T(p)
        self.num_orbital_index_bits = int(math.ceil(math.log2(self.num_electrons)))
        self.energy_configuration_bits = int(math.ceil(math.log2(self.num_energy_configurations)))
        self.wfr_spec = wave_function.WaveFunctionRegisterSpec(
            self.dimension, n, self.space_length, self.num_electrons,
            self.num_moving_nuclei, self.num_stationary_nuclei, self.should_use_spin)
        self.ham_spec = hamiltonian.HamiltonianSpec(self.wfr_spec, self.nuclei_data)
        self.ham_spec.set_should_apply_potential_to_phase(self.should_apply_potential_to_phase)
        self.ham_spec.set_should_revert_potential_ancilla_value(
            self.should_revert_potential_ancilla_value)
        self.ham_spec.set_should_mask_potential_singularity(
            self.should_mask_potential_singularity)
        self.disc_spec = discretization.DiscretizationSpec(self.delta_t)
        self.asy_spec = antisymmetrization.AntisymmetrizationSpec(
            self.wfr_spec, self.antisymmetrization_method)
        self.evo_spec = time_evolution.TimeEvolutionSpec(
            self.ham_spec, self.disc_spec,
            self.num_atom_iterations, self.num_elec_per_atom_iterations)
        self.evo_spec.set_should_apply_qft(self.should_apply_qft)
        self.evo_spec.set_should_calculate_electron_motion(self.should_calculate_electron_motion)
        self.evo_spec.set_should_calculate_nucleus_motion(self.should_calculate_nucleus_motion)
        self.evo_spec.set_should_calculate_potential_term(self.should_calculate_potential_term)
        self.evo_spec.set_should_calculate_kinetic_term(self.should_calculate_kinetic_term)
        self.ene_spec = energy_initialization.EnergyConfigurationSpec(
            self.energy_configuration_weights,
            self.initial_electron_orbitals,
            self.initial_nucleus_orbitals
        )
        self.ene_spec.set_should_initialize_energy(self.should_initialize_energy)
        self.size_calculation_done = True

    def allocate_registers(self) -> IntegratorFrame:
        """ Allocate the main registers.

            Registers required for arithmetic intermediate values will be allocated later.
            Must call calculate_sizes() before calling this.
        """
        assert self.size_calculation_done
        assert not self.register_allocation_done

        # Step 1. call allocate_registers() on all direct sub frames

        # areg = slater.ARegister(self.coordinate_bits, self.num_electrons, self.dimension)
        # breg = slater.BRegister(self.num_orbital_index_bits,
        #                         self.num_electrons, self.dimension)
        # self.slater_areg = areg
        # self.slater_breg = breg
        # self.slater_breg.set_areg(self.slater_areg)

        # Step 2. allocate registers for the own frame
        frame = IntegratorFrame()

        self._add_wave_function_registers(frame)
        self._add_energy_configuration_registers(frame)

        self.frame = frame
        self.register_allocation_done = True
        return frame

    def _add_energy_configuration_registers(self, frame: IntegratorFrame):
        if self.energy_configuration_bits == 0:
            return
        conf_reg = QuantumRegister(self.energy_configuration_bits, "p")
        # conf_ancilla_reg = QuantumRegister(self.energy_configuration_bits, "pa")
        frame.set_energy_configuration_regs(conf_reg,
                                            # conf_ancilla_reg
                                            )

    def _add_wave_function_registers(self, frame: IntegratorFrame):
        self._add_coord_index_registers(frame)

    def _add_coord_index_registers(self, frame: IntegratorFrame):
        # We want the elec coords to appear first.
        # This will make the elec coords on the lower bits of the state vector.
        self._add_elec_coord_index_registers(frame)
        self._add_nuclei_coord_index_registers(frame)
        sigma_regs = self.asy_spec.allocate_sigma_regs()
        ancilla = self.asy_spec.allocate_ancilla_register()
        frame.set_slater_regs(sigma_regs, ancilla)

    def _add_nuclei_coord_index_registers(self, frame: IntegratorFrame):
        nregs = self.wfr_spec.allocate_nucl_registers()
        frame.set_nuclei_index_regs(nregs)

    def _add_elec_coord_index_registers(self, frame: IntegratorFrame):
        eregs = self.wfr_spec.allocate_elec_registers()
        frame.set_elec_index_regs(eregs)

    def _add_time_evolution_work_registers(self, frame: IntegratorFrame):
        n = self.coordinate_bits
        vx_const_numerator_reg = QuantumRegister(n + 1, "one")
        frame.set_time_evolution_work_registers(vx_const_numerator_reg)

    def build_circuit(self):
        """ Build the gates for the circuit.
            You must call calculate_sizes() and allocate_registers() before calling this.
        """
        assert self.register_allocation_done
        assert not self.circuit_is_built
        use_new_scheme = True
        if use_new_scheme:
            self.build_circuit_y()
            return

    def build_circuit_y(self):
        """ build circuit, new edition"""
        self._build_time_evolution_step_y()

    def _build_initialization_step(self):
        init_block = self.build_initialization_block()
        self.frame.invoke(
            init_block.bind(
                eregs=self.frame.e_index_regs,
                nregs=self.frame.n_index_regs,
                bregs=self.frame.slater_bregs,
                shuffle=self.frame.slater_ancilla,
                p=self.frame.energy_configuration_reg,
                ))

    def build_initialization_block(self):
        """ build the block object"""
        init_block = energy_initialization.GeneralStatePreparationBlock(
            self.ene_spec,
            self.asy_spec
        )
        return init_block

    def _build_time_evolution_step_y(self):
        evo_block = self.build_time_evolution_block()
        with check_time("TimeEvolutionBlock.invoke"):
            self.frame.invoke(evo_block.bind(
                eregs=self.frame.e_index_regs,
                nregs=self.frame.n_index_regs,
                slater_indices=self.frame.slater_bregs,
                slater_ancilla=self.frame.slater_ancilla,
                p=self.frame.energy_configuration_reg,
                ),
                invoke_as_instruction=True)

    def build_time_evolution_block(self) -> time_evolution.SuzukiTrotterMethodBlock:
        """ build the block object"""
        evo_block = time_evolution.SuzukiTrotterMethodBlock(
            self.evo_spec, self.ene_spec, self.asy_spec,
            use_motion_block_gates=self.should_use_motion_block_gates)
        logger.info("TimeEvolutionBlock.num_qubits=%d", evo_block.circuit.num_qubits)
        self.time_evolution_block = evo_block
        return evo_block

    def get_time_evolution_block(self) -> time_evolution.SuzukiTrotterMethodBlock:
        """ get time evolution block"""
        return self.time_evolution_block

    def build_electron_motion_block(self) -> time_evolution.ElectronMotionBlock:
        """ build the block object """
        elec_motion_block = time_evolution.ElectronMotionBlock(self.evo_spec)
        return elec_motion_block

    def _build_electron_motion_block(self, frame: IntegratorFrame):
        use_new_scheme = True
        if self.should_calculate_potential_term:
            self._build_elec_potential_step_y(frame)
        if self.should_apply_qft:
            self._build_apply_qft_step_y(frame)
        if self.should_calculate_kinetic_term:
            self._build_elec_kinetic_step_y(frame)
        if self.should_apply_qft:
            self._build_apply_qft_step_y(frame, inverse=True)

    def build_nucleus_motion_block(self) -> time_evolution.NucleusMotionBlock:
        """ build the block object """
        nucl_motion_block = time_evolution.NucleusMotionBlock(self.evo_spec)
        return nucl_motion_block

    def _build_nuclei_motion_block(self, frame: IntegratorFrame):
        use_new_scheme = True
        if self.should_calculate_potential_term:
            self._build_nuclei_potential_step_y(frame)
        if self.should_apply_qft:
            self._build_apply_qft_step_y(frame)
        if self.should_calculate_kinetic_term:
            self._build_nuclei_kinetic_step_y(frame)
        if self.should_apply_qft:
            self._build_apply_qft_step_y(frame, inverse=True)

    def _build_elec_potential_step_y(self, frame: IntegratorFrame):
        block = self.build_elec_potential_block()
        t1 = time.time()
        frame.invoke(block.bind(eregs=self.frame.e_index_regs, nregs=self.frame.n_index_regs))
        dt = time.time() - t1
        if dt > LOG_TIME_THRESH:
            logger.info("  ElectroPotentialBlock.invoke took. %d ms", round(dt*1000))

    def build_elec_potential_block(self, allocate=True, build=True
                                   ) -> hamiltonian.ElectronPotentialBlock:
        """ build a ElectronPotentialBlock instance."""
        block = hamiltonian.ElectronPotentialBlock(
            self.ham_spec, self.disc_spec, allocate=allocate, build=build)
        return block

    def build_nucl_potential_block(self, allocate=True, build=True):
        """ build a NucleusPotentialBlock instance."""
        block = hamiltonian.NucleusPotentialBlock(
            self.ham_spec, self.disc_spec, allocate=allocate, build=build)
        return block

    def _build_nuclei_potential_step_y(self, frame: IntegratorFrame):
        block = hamiltonian.NucleusPotentialBlock(self.ham_spec, self.disc_spec)
        frame.invoke(block.bind(nregs=self.frame.n_index_regs))

    def build_elec_kinetic_block(self, build=True) -> hamiltonian.ElectronKineticBlock:
        """ build ElectronKineticBlock """
        block = hamiltonian.ElectronKineticBlock(self.wfr_spec, self.disc_spec, build=build)
        return block

    def _build_elec_kinetic_step_y(self, frame: IntegratorFrame):
        block = hamiltonian.ElectronKineticBlock(self.wfr_spec, self.disc_spec)
        frame.invoke(block.bind(eregs=self.frame.e_index_regs))

    def build_nuclei_kinetic_block(self) -> hamiltonian.NucleusKineticBlock:
        """ build NucleusKineticBlock """
        block = hamiltonian.NucleusKineticBlock(self.wfr_spec, self.disc_spec, self.ham_spec)
        return block

    def _build_nuclei_kinetic_step_y(self, frame: IntegratorFrame):
        block = hamiltonian.NucleusKineticBlock(self.wfr_spec, self.disc_spec, self.ham_spec)
        frame.invoke(block.bind(nregs=self.frame.n_index_regs))

    def build_apply_qft_block(self, inverse: bool = False, build=True):
        """ build a QFT block on all index registers """
        block = qft.QFTOnWaveFunctionsBlock(self.wfr_spec, inverse=inverse, build=build)
        return block

    def _build_apply_qft_step_y(self, frame: IntegratorFrame, inverse: bool = False):
        """ apply QFT on all index registers """
        block = qft.QFTOnWaveFunctionsBlock(self.wfr_spec, inverse=inverse)
        frame.invoke(block.bind(
            eregs=self.frame.e_index_regs,
            nregs=self.frame.n_index_regs))

    def dump_sizes(self):
        """ Dump the computed sizes"""
        print("{")
        print(f" dimension: {self.dimension},")
        print(f" coordinate_bits: {self.coordinate_bits},")
        print(f" space_length: {self.space_length},")
        print(f" delta_x: {self.delta_x},")
        print(f" k: {self.k},")
        print(f" max_particle_charge: {self.max_particle_charge}")
        print(f" v_numerator_int_bits: {self.v_numerator_int_bits},")
        total_index_bits = (self.num_electrons + self.num_moving_nuclei) * self.dimension * self.coordinate_bits
        print(f" total_index_bits: {total_index_bits},")
        print("}")
