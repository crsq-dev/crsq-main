"""Time evolution by Suzuki-Trotter decomposition (or split operator method)
"""

from contextlib import contextmanager
from typing import List
import logging
import time
from qiskit import QuantumRegister
from crsq_heap import heap
import crsq.utils.statevector as utils_svec
from crsq.blocks import (
    energy_initialization,
    antisymmetrization,
    hamiltonian,
    qft,
    rfqhamiltonian,
)
from crsq.blocks.time_evolution import spec

logger = logging.getLogger(__name__)
LOG_TIME_THRESH = 1


@contextmanager
def check_time(label: str):
    """context manager to check time.
    :param label: label used for logs.
    """
    logger.info("%s start", label)
    t1 = time.time()
    yield
    dt = time.time() - t1
    if dt > LOG_TIME_THRESH:
        logger.info("%s took %d msec", label, round(dt * 1000))
    else:
        logger.info("%s end", label)


class ElectronMotionBlock(heap.Frame):
    """H_ep, QFT, H_ek, QFT\dagger

    Electron motion by Suzuki-Trotter decomposition
    """

    def __init__(
        self,
        evo_spec: spec.TimeEvolutionSpec,
        sim_time: float,
        is_last_elec_iter: bool = False,
        label=" TEV_e(x)",
        allocate=True,
        build=True,
    ):
        super().__init__(label=label)
        t1 = time.time()
        self._evo_spec = evo_spec
        self._rfq_spec = evo_spec.rfq_spec
        self._ham_spec = evo_spec.ham_spec
        self._disc_spec = evo_spec.disc_spec
        self._wfr_spec = self._ham_spec.wfr_spec
        self._sim_time = sim_time
        self._is_last_elec_iter = is_last_elec_iter
        # registers
        self._e_index_regs: List[List[QuantumRegister]]
        self._n_index_regs: List[List[QuantumRegister]]
        # blocks
        # self._elec_potential_block: hamiltonian.ElectronPotentialBlock
        logger.info(f"should_save_p_state_vector: {evo_spec.should_save_p_state_vector}")
        if allocate:
            self.allocate_registers()
            if build:
                self.build_circuit()
        dt = time.time() - t1
        if dt > LOG_TIME_THRESH:
            logger.info("ElectronMotionBlock() took %d msec", round(dt * 1000))

    def allocate_registers(self):
        """allocate"""
        wfr_spec = self._wfr_spec
        evo_spec = self._evo_spec
        self._e_index_regs = wfr_spec.allocate_elec_registers()
        self._n_index_regs = wfr_spec.allocate_nucl_registers()
        self.add_param(("eregs", self._e_index_regs), ("nregs", self._n_index_regs))
        if evo_spec.should_use_rfq_gray_code:
            self._target = QuantumRegister(1, "target")
            self.add_param(self._target)
            # additionally, we want to allocate padding registers when saving state vector.
            # so that the circuit size will match the total outer-most circuit size.
            if evo_spec.should_save_p_state_vector and self._is_last_elec_iter:
                padding_bits = (wfr_spec.num_coordinate_bits - 1) * wfr_spec.dimension
                self._padding_regs = QuantumRegister(padding_bits, "pad")
                self.add_local(self._padding_regs)

    def build_circuit_on(self, other_frame: heap.Frame):
        """Build the instructions on another compatible quantum circuit."""
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
        """build the gates for time evolution.
        There are several variations for this.
        """
        save_p_state_vector = (
            self._is_last_elec_iter and self._evo_spec.should_save_p_state_vector
        )
        logger.info(
            "ElectronMotionBlock.build_circuit(save_qft_state_vector = %s) start",
            save_p_state_vector,
        )
        wfr_spec = self._wfr_spec
        evo_spec = self._evo_spec
        if (
            evo_spec.should_calculate_potential_term
            and wfr_spec.has_elec_potential_term
        ):
            self._build_elec_potential_step()
        else:
            logger.info("Skipping electron potential term")
        if evo_spec.should_apply_qft:
            self._build_apply_electron_qft_step(inverse=True, save_p_state_vector=False)
        else:
            logger.info("Skipping electron qft")
        if evo_spec.should_calculate_kinetic_term:
            self._build_elec_kinetic_step()
        else:
            logger.info("Skipping electron kinetic term")
        if evo_spec.should_apply_qft:
            self._build_apply_electron_qft_step(
                inverse=False,
                save_p_state_vector=save_p_state_vector,
            )
        logger.info("ElectronMotionBlock.build_circuit end")

    def _save_state_vector_with_label(self, key: str):
        label = self._evo_spec.make_state_vector_label(self._sim_time, key)
        logger.info(
            "save_statevector: %s, num_qubits = %d", label, self.circuit.num_qubits
        )
        self.circuit.save_statevector(label=label)

    # ☆☆☆
    def _build_elec_potential_step(self):
        method = self._evo_spec.method
        if method == spec.SUZUKI_TROTTER_ARITHMETIC:
            self._build_elec_potential_step_arithmetic()
        elif method == spec.SUZUKI_TROTTER_QROM:
            self._build_elec_potential_step_qrom()

    def _build_elec_potential_step_arithmetic(self):
        block = self.build_elec_potential_block_arithmetic()
        logger.info(
            "ElectronPotentialBlock[ARITHMETIC].num_qubits = %d",
            block.circuit.num_qubits,
        )
        with check_time("ElectronPotentialBlock.invoke"):
            self.invoke(block.bind(eregs=self._e_index_regs, nregs=self._n_index_regs))

    def build_elec_potential_block_arithmetic(self, allocate=True, build=True):
        """build a ElectronPotentialBlock instance."""
        block = hamiltonian.ElectronPotentialBlock(
            self._ham_spec, self._disc_spec, allocate=allocate, build=build
        )
        return block

    def _build_elec_potential_step_qrom(self):
        block: rfqhamiltonian.RfqElectronPotentialBlock = (
            self.build_elec_potential_block_qrom()
        )
        # we cannot save state vector at this point.
        logger.info(
            "RfqElectronPotentialBlock[QROM].num_qubits = %d", block.circuit.num_qubits
        )
        if self._rfq_spec.should_save_state_vector_per_qrom:
            # pre-allocate the temporary qubits required by the qrom block.
            t = self._temp_allocator.allocate(block._temp_allocator.size, "tmp")
            self._temp_allocator.free(t)
            self._save_state_vector_with_label("qrom0")
        with check_time("RfqElectronPotentialBlock.invoke"):
            if self._evo_spec.should_use_rfq_gray_code:
                bound = block.bind(
                    eregs=self._e_index_regs,
                    nregs=self._n_index_regs,
                    target=self._target,
                )
            else:
                bound = block.bind(eregs=self._e_index_regs, nregs=self._n_index_regs)
            self.invoke(bound, invoke_as_instruction=True)
        if self._rfq_spec.should_save_state_vector_per_qrom:
            self._save_state_vector_with_label("qrom1")

    def build_elec_potential_block_qrom(self, allocate=True, build=True):
        """build a RfqElectronPotentialBlock instance."""
        block = rfqhamiltonian.RfqElectronPotentialBlock(
            self._evo_spec.rfq_spec,
            self._ham_spec,
            self._disc_spec,
            allocate=allocate,
            build=build,
        )
        return block
    # ☆☆☆

    def _build_apply_electron_qft_step(
        self, inverse: bool = False, save_p_state_vector: bool = False
    ):
        """apply QFT on all index registers"""
        logger.info("_build_apply_electron_qft_step, sim_time=%f", self._sim_time)
        if save_p_state_vector:
            # record before qft back to q-space
            self._save_state_vector_with_label("qft")

        qft_block = qft.QFTOnWaveFunctionsBlock(
            self._wfr_spec, on_electrons=True, inverse=inverse
        )
        # The QFT block contains non-gate instructions.
        with check_time("QFTOnWaveFunctionsBlock(e).invoke"):
            self.invoke(
                qft_block.bind(eregs=self._e_index_regs), invoke_as_instruction=True
            )
        # if inverse and self._evo_spec.should_save_state_vector_per_qft:
        #     # record after qft dagger
        #     self._save_state_vector_with_suffix("_qft")

    def build_elec_kinetic_block(self):
        """build a ElectronKineticBlock instance."""
        block = hamiltonian.ElectronKineticBlock(self._wfr_spec, self._disc_spec)
        return block

    def _build_elec_kinetic_step(self):
        block = self.build_elec_kinetic_block()
        with check_time("ElectronKineticBlock.invoke"):
            self.invoke(block.bind(eregs=self._e_index_regs))

    def bind(
        self,
        eregs: List[List[QuantumRegister]],
        nregs: List[List[QuantumRegister]],
        target: QuantumRegister = None,
    ):
        """bind arguments to the function"""
        if self._evo_spec.should_use_rfq_gray_code:
            return heap.Binding(
                self, {"eregs": eregs, "nregs": nregs, "target": target}
            )
        else:
            return heap.Binding(self, {"eregs": eregs, "nregs": nregs})


class NucleusMotionBlock(heap.Frame):
    """H_np, QFT, H_nk, QFT\dagger"""

    def __init__(
        self,
        evo_spec: spec.TimeEvolutionSpec,
        sim_time: float,
        label=" TEV_n(x)",
        allocate=True,
        build=True,
    ):
        super().__init__(label=label)
        self._evo_spec = evo_spec
        self._ham_spec = evo_spec.ham_spec
        self._disc_spec = evo_spec.disc_spec
        self._wfr_spec = self._ham_spec.wfr_spec
        self._sim_time = sim_time
        # registers
        self._n_index_regs: List[List[QuantumRegister]]
        if allocate:
            self.allocate_registers()
            if build:
                with check_time("NucleusMotionBlock.build_circuit"):
                    self.build_circuit()

    def allocate_registers(self):
        """allocate"""
        wfr_spec = self._wfr_spec
        self._n_index_regs = wfr_spec.allocate_nucl_registers()
        self.add_param(("nregs", self._n_index_regs))

    def build_circuit_on(self, other_frame: heap.Frame):
        """Build the instructions on another compatible quantum circuit."""
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
        """build the gates for time evolution.
        There are several variations for this.
        """
        if self._wfr_spec.num_moving_nuclei == 0:
            return
        evo_spec = self._evo_spec
        if evo_spec.should_calculate_potential_term:
            self._build_nuclei_potential_step()
        if evo_spec.should_apply_qft:
            self._build_apply_nucleus_qft_step(inverse=True)
        if evo_spec.should_calculate_kinetic_term:
            self._build_nuclei_kinetic_step()
        if evo_spec.should_apply_qft:
            self._build_apply_nucleus_qft_step()

    def _build_nuclei_potential_step(self):
        block = hamiltonian.NucleusPotentialBlock(self._ham_spec, self._disc_spec)
        with check_time("NucleusPotentialBlock.invoke"):
            self.invoke(block.bind(nregs=self._n_index_regs))

    def _build_apply_nucleus_qft_step(self, inverse: bool = False):
        """apply QFT on all index registers"""
        block = qft.QFTOnWaveFunctionsBlock(
            self._wfr_spec, on_nuclei=True, inverse=inverse
        )
        # The QFT block contains non-gate instructions.
        with check_time("QFTOnWaveFunctionsBlock(n).invoke"):
            self.invoke(
                block.bind(nregs=self._n_index_regs), invoke_as_instruction=True
            )

    def _build_nuclei_kinetic_step(self):
        block = hamiltonian.NucleusKineticBlock(
            self._wfr_spec, self._disc_spec, self._ham_spec
        )
        with check_time("NucleusKineticBlock.invoke"):
            self.invoke(block.bind(nregs=self._n_index_regs))

    def bind(self, nregs: List[List[QuantumRegister]]):
        """bind arguments to the function"""
        return heap.Binding(self, {"nregs": nregs})


class SuzukiTrotterMethodBlock(heap.Frame):
    """Suzuki Trotter method time evolution Block"""

    def __init__(
        self,
        evo_spec: spec.TimeEvolutionSpec,
        ene_spec: energy_initialization.EnergyConfigurationSpec,
        asy_spec: antisymmetrization.AntisymmetrizationSpec,
        label="SuzukiTrotter",
        allocate=True,
        build=True,
        use_motion_block_gates=False,
    ):
        super().__init__(label=label)
        assert isinstance(evo_spec, spec.TimeEvolutionSpec)
        assert isinstance(ene_spec, energy_initialization.EnergyConfigurationSpec)
        assert isinstance(asy_spec, antisymmetrization.AntisymmetrizationSpec)
        self._evo_spec = evo_spec
        self._ene_spec = ene_spec
        self._asy_spec = asy_spec
        self._ham_spec = evo_spec.ham_spec
        self._disc_spec = evo_spec.disc_spec
        self._wfr_spec = self._ham_spec.wfr_spec
        self._use_motion_block_gates = use_motion_block_gates
        logger.info(f"method: {evo_spec.method}")
        logger.info(f"use_motion_block_gates: {use_motion_block_gates}")
        logger.info(f"evo_spec.should_save_p_state_vector: {evo_spec.should_save_p_state_vector}")
        # if evo_spec.method == spec.SUZUKI_TROTTER_QROM and not use_motion_block_gates:
        #     raise ValueError(
        #         "SuzukiTrotterMethodBlock: QROM requires use_motion_block_gates=True"
        #     )
        # registers
        self._e_index_regs: List[List[QuantumRegister]]
        self._n_index_regs: List[List[QuantumRegister]]
        self._energy_configuration_reg: QuantumRegister = None
        self._slater_indices: list[QuantumRegister] = []
        self._slater_ancilla: QuantumRegister = None
        self._elec_motion_block = None
        if allocate:
            self.allocate_registers()
            if build:
                with check_time("SuzukiTrotterMethodBlock.build_circuit"):
                    self.build_circuit()

    def allocate_registers(self):
        """allocate"""
        wfr_spec = self._wfr_spec
        ene_spec = self._ene_spec
        asy_spec = self._asy_spec
        self._e_index_regs = wfr_spec.allocate_elec_registers()
        self._n_index_regs = wfr_spec.allocate_nucl_registers()
        self.add_param(("eregs", self._e_index_regs), ("nregs", self._n_index_regs))
        if self._evo_spec.should_use_rfq_gray_code:
            self._target = QuantumRegister(1, "target")
            self.add_param(self._target)
        self._slater_ancilla = asy_spec.allocate_ancilla_register()
        self.add_param(self._slater_ancilla)
        self._slater_indices = asy_spec.allocate_sigma_regs()
        self.add_param(("slater_indices", self._slater_indices))

        if ene_spec.num_energy_configuration_bits > 0:
            self._energy_configuration_reg = QuantumRegister(
                ene_spec.num_energy_configuration_bits, "p"
            )
            self.add_param(self._energy_configuration_reg)

    def build_circuit(self):
        """build the gates for time evolution.
        There are several variations for this.
        """
        self._build_initialization_block()
        evo_spec = self._evo_spec
        if evo_spec.should_save_q_state_vector:
            self._build_time_evolution_circuit_with_save()
        else:
            self._build_time_evolution_circuit_without_save()

    def _build_time_evolution_circuit_without_save(self):
        qc = self.circuit
        evo_spec = self._evo_spec
        n_atom_it = evo_spec.num_atom_iterations
        n_elec_it = evo_spec.num_elec_per_atom_iterations
        sim_time = 0.0
        delta_t = self._evo_spec.disc_spec.delta_t
        if evo_spec.should_use_for_loop_gate:
            self._build_time_evolution_circuit_without_save_for_loop_gate(
                n_atom_it, n_elec_it, sim_time, delta_t
            )
        else:
            self._build_time_evolution_circuit_without_save_flat_loop(
                n_atom_it, n_elec_it, sim_time, delta_t
            )

    def _build_time_evolution_circuit_without_save_for_loop_gate(
        self, n_atom_it, n_elec_it, sim_time, delta_t
    ):
        qc = self.circuit
        evo_spec = self._evo_spec
        n_atom_it = evo_spec.num_atom_iterations
        n_elec_it = evo_spec.num_elec_per_atom_iterations
        with qc.for_loop(range(n_atom_it)):
            with qc.for_loop(range(n_elec_it)):
                sim_time += delta_t
                if self._evo_spec.should_calculate_electron_motion:
                    self._build_electron_motion_step(sim_time, False)
            if self._evo_spec.should_calculate_nucleus_motion:
                self._build_nuclei_motion_block(sim_time)

    def _build_time_evolution_circuit_without_save_flat_loop(
        self, n_atom_it, n_elec_it, sim_time, delta_t
    ):
        evo_spec = self._evo_spec
        n_atom_it = evo_spec.num_atom_iterations
        n_elec_it = evo_spec.num_elec_per_atom_iterations
        for _atom_it in range(n_atom_it):
            for _elec_it in range(n_elec_it):
                sim_time += delta_t
                if evo_spec.should_calculate_electron_motion:
                    self._build_electron_motion_step(sim_time, False)
            if evo_spec.should_calculate_nucleus_motion:
                self._build_nuclei_motion_block(sim_time)

    def _build_time_evolution_circuit_with_save(self):
        evo_spec = self._evo_spec
        n_atom_it = evo_spec.num_atom_iterations
        n_elec_it = evo_spec.num_elec_per_atom_iterations
        sim_time = 0.0
        delta_t = self._evo_spec.disc_spec.delta_t
        # cannot save state vector for t=0.
        # we need to go through the circuit one loop to get all registers allocated.
        # self._save_state_vector(time)

        if evo_spec.should_use_for_loop_gate:
            logger.info("Using for loop gate")
            self._build_time_evolution_circuit_with_save_for_loop_gate(
                n_atom_it, n_elec_it, sim_time, delta_t
            )
        else:
            logger.info("Using flat loop")
            self._build_time_evolution_circuit_with_save_flat_loop(
                n_atom_it, n_elec_it, sim_time, delta_t
            )

    def _build_time_evolution_circuit_with_save_for_loop_gate(
        self, n_atom_it, n_elec_it, sim_time, delta_t
    ):
        qc = self.circuit
        evo_spec = self._evo_spec
        n_atom_it = evo_spec.num_atom_iterations
        n_elec_it = evo_spec.num_elec_per_atom_iterations
        with qc.for_loop(range(n_atom_it)):
            with qc.for_loop(range(n_elec_it)):
                sim_time += delta_t
                logger.info("before electron motion step. sim_time=%f", sim_time)
                if evo_spec.should_calculate_electron_motion:
                    self._build_electron_motion_step(sim_time)
            if evo_spec.should_calculate_nucleus_motion:
                self._build_nuclei_motion_block(sim_time)
            self._save_state_vector(sim_time)

    def _build_time_evolution_circuit_with_save_flat_loop(
        self, n_atom_it, n_elec_it, sim_time, delta_t
    ):
        evo_spec = self._evo_spec
        n_atom_it = evo_spec.num_atom_iterations
        n_elec_it = evo_spec.num_elec_per_atom_iterations
        for _atom_it in range(n_atom_it):
            for _elec_it in range(n_elec_it):
                sim_time += delta_t
                logger.info("before electron motion step. sim_time=%f", sim_time)
                if evo_spec.should_calculate_electron_motion:
                    is_final_elec_iter = _elec_it == n_elec_it - 1
                    self._build_electron_motion_step(sim_time, is_final_elec_iter)
            if evo_spec.should_calculate_nucleus_motion:
                self._build_nuclei_motion_block(sim_time)
            self._save_state_vector(sim_time)

    def _save_state_vector(self, sim_time):
        label = self._evo_spec.make_state_vector_label(sim_time)
        logger.info("save_statevector: %s", label)
        self.circuit.save_statevector(label=label)

    def _build_initialization_block(self):
        if self._ene_spec.num_energy_configurations > 1:
            self._initialize_with_general_state()
        else:
            self._initialize_with_sd_state()

    def _initialize_with_general_state(self):
        g_block = energy_initialization.GeneralStatePreparationBlock(
            self._ene_spec, self._asy_spec
        )
        with check_time("GeneralStatePreparationBlock.invoke"):
            self.invoke(
                g_block.bind(
                    eregs=self._e_index_regs,
                    nregs=self._n_index_regs,
                    bregs=self._slater_indices,
                    shuffle=self._slater_ancilla,
                    p=self._energy_configuration_reg,
                )
            )

    def _initialize_with_sd_state(self):
        sd_block = energy_initialization.SlaterDeterminantPreparationBlock(
            self._ene_spec, self._asy_spec, 0
        )
        logger.info(
            "SlaterDeterminantPreparationBlock.num_qubits=%d",
            sd_block.circuit.num_qubits,
        )
        with check_time("SlaterDeterminantPreparationBlock.invoke"):
            self.invoke(
                sd_block.bind(
                    eregs=self._e_index_regs,
                    nregs=self._n_index_regs,
                    bregs=self._slater_indices,
                    shuffle=self._slater_ancilla,
                )
            )

    def _build_electron_motion_step(self, sim_time, is_final_elec_iter):
        wfr_spec = self._wfr_spec
        if wfr_spec.num_electrons == 0:
            return
        if self._use_motion_block_gates:
            logger.info("Build ElectronMotionBlock:")
            elec_motion_block = self.build_electron_motion_block(
                sim_time, is_final_elec_iter
            )
            logger.info(
                "ElectronMotionBlock.num_qubits = %d",
                elec_motion_block.circuit.num_qubits,
            )
            with check_time("ElectronMotionBlock.invoke"):
                if self._evo_spec.should_use_rfq_gray_code:
                    bound = elec_motion_block.bind(
                        eregs=self._e_index_regs,
                        nregs=self._n_index_regs,
                        target=self._target,
                    )
                else:
                    bound = elec_motion_block.bind(
                        eregs=self._e_index_regs, nregs=self._n_index_regs
                    )
                self.invoke(bound, invoke_as_instruction=True)
            return
        logger.info("ElectronMotionBlock: using individual steps")
        evo_spec = self._evo_spec
        if (
            evo_spec.should_calculate_potential_term
            and wfr_spec.has_elec_potential_term
        ):
            self._build_elec_potential_step()
        if evo_spec.should_apply_qft:
            self._build_apply_electron_qft_step(inverse=True)
        if evo_spec.should_calculate_kinetic_term:
            self._build_elec_kinetic_step()
        if evo_spec.should_apply_qft:
            self._build_apply_electron_qft_step()

    def build_electron_motion_block(self, sim_time: float, is_final_elec_iter=False):
        # cache the block.
        dont_cache = self._evo_spec.should_save_p_state_vector and is_final_elec_iter
        if dont_cache:
            # cannot reuse when if_final_elec_iter is True.:
            elec_motion_block = ElectronMotionBlock(
                self._evo_spec, sim_time, is_final_elec_iter
            )
            return elec_motion_block
        # otherwise cacheable.
        if self._elec_motion_block is None:
            self._elec_motion_block = ElectronMotionBlock(
                self._evo_spec, sim_time, False
            )
        return self._elec_motion_block

    def _build_elec_potential_step(self):
        method = self._evo_spec.method
        if method == spec.SUZUKI_TROTTER_ARITHMETIC:
            self._build_elec_potential_step_arithmetic()
        elif method == spec.SUZUKI_TROTTER_QROM:
            self._build_elec_potential_step_qrom()

    def _build_elec_potential_step_arithmetic(self):
        block = self.build_elec_potential_block_arithmetic()
        logger.info(
            "ElectronPotentialBlock[ARITHMETIC].num_qubits = %d",
            block.circuit.num_qubits,
        )
        with check_time("ElectronPotentialBlock.invoke"):
            self.invoke(block.bind(eregs=self._e_index_regs, nregs=self._n_index_regs))

    def build_elec_potential_block_arithmetic(self, allocate=True, build=True):
        """build a ElectronPotentialBlock instance."""
        block = hamiltonian.ElectronPotentialBlock(
            self._ham_spec, self._disc_spec, allocate=allocate, build=build
        )
        return block

    def _build_elec_potential_step_qrom(self):
        block: rfqhamiltonian.RfqElectronPotentialBlock = (
            self.build_elec_potential_block_qrom()
        )
        # we cannot save state vector at this point.
        logger.info(
            "RfqElectronPotentialBlock[QROM].num_qubits = %d", block.circuit.num_qubits
        )
        with check_time("RfqElectronPotentialBlock.invoke"):
            if self._evo_spec.should_use_rfq_gray_code:
                bound = block.bind(
                    eregs=self._e_index_regs,
                    nregs=self._n_index_regs,
                    target=self._target,
                )
            else:
                bound = block.bind(eregs=self._e_index_regs, nregs=self._n_index_regs)
            self.invoke(bound, invoke_as_instruction=True)

    def build_elec_potential_block_qrom(self, allocate=True, build=True):
        """build a RfqElectronPotentialBlock instance."""
        block = rfqhamiltonian.RfqElectronPotentialBlock(
            self._evo_spec.rfq_spec,
            self._ham_spec,
            self._disc_spec,
            allocate=allocate,
            build=build,
        )
        return block

    def _build_apply_electron_qft_step(self, inverse: bool = False):
        """apply QFT on all index registers"""
        block = qft.QFTOnWaveFunctionsBlock(
            self._wfr_spec, on_electrons=True, inverse=inverse
        )
        # The QFT block contains non-gate instructions.
        with check_time("QFTOnWaveFunctionsBlock(e).invoke"):
            self.invoke(
                block.bind(eregs=self._e_index_regs), invoke_as_instruction=True
            )

    def _build_nuclei_motion_block(self, sim_time: float):
        if self._wfr_spec.num_moving_nuclei == 0:
            return
        if self._use_motion_block_gates:
            nucl_motion_block = NucleusMotionBlock(self._evo_spec, sim_time)
            with check_time("NucleusMotionBlock.invoke"):
                self.invoke(
                    nucl_motion_block.bind(nregs=self._n_index_regs),
                    invoke_as_instruction=True,
                )
            return

        evo_spec = self._evo_spec
        if evo_spec.should_calculate_potential_term:
            self._build_nuclei_potential_step()
        if evo_spec.should_apply_qft:
            self._build_apply_nucleus_qft_step(inverse=True)
        if evo_spec.should_calculate_kinetic_term:
            self._build_nuclei_kinetic_step()
        if evo_spec.should_apply_qft:
            self._build_apply_nucleus_qft_step()

    def _build_nuclei_potential_step(self):
        assert self._wfr_spec.num_moving_nuclei > 0
        block = hamiltonian.NucleusPotentialBlock(self._ham_spec, self._disc_spec)
        with check_time("NucleusPotentialBlock.invoke"):
            self.invoke(block.bind(nregs=self._n_index_regs))


    def _build_apply_nucleus_qft_step(self, inverse: bool = False):
        """apply QFT on all index registers"""
        assert self._wfr_spec.num_moving_nuclei > 0
        block = qft.QFTOnWaveFunctionsBlock(
            self._wfr_spec, on_nuclei=True, inverse=inverse
        )
        # The QFT block contains non-gate instructions.
        with check_time("QFTOnWaveFunctionsBlock(n).invoke"):
            self.invoke(
                block.bind(nregs=self._n_index_regs), invoke_as_instruction=True
            )

    def _build_elec_kinetic_step(self):
        block = hamiltonian.ElectronKineticBlock(self._wfr_spec, self._disc_spec)
        with check_time("ElectronKineticBlock.invoke"):
            self.invoke(block.bind(eregs=self._e_index_regs))

    def _build_nuclei_kinetic_step(self):
        assert self._wfr_spec.num_moving_nuclei > 0
        block = hamiltonian.NucleusKineticBlock(
            self._wfr_spec, self._disc_spec, self._ham_spec
        )
        with check_time("NucleusKineticBlock.invoke"):
            self.invoke(block.bind(nregs=self._n_index_regs))

    def bind(
        self,
        eregs: List[List[QuantumRegister]],
        nregs: List[List[QuantumRegister]],
        target: QuantumRegister,
        slater_indices: List[QuantumRegister],
        slater_ancilla: QuantumRegister,
        p: QuantumRegister,
    ) -> heap.Binding:
        """bind arguments to a binding object"""
        arg_map = {
            "eregs": eregs,
            "nregs": nregs,
            "slater_indices": slater_indices,
            "p": p,
        }
        if self._slater_ancilla is not None:
            arg_map[self._slater_ancilla.name] = slater_ancilla
        if self._evo_spec.should_use_rfq_gray_code:
            arg_map[self._target.name] = target
        return heap.Binding(self, arg_map)
