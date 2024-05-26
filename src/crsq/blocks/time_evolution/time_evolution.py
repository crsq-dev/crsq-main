""" Time Evolution module.
"""

from typing import List
import logging
import time
from contextlib import contextmanager
from qiskit import QuantumRegister
from crsq import heap
from crsq.blocks import (
    hamiltonian, qft, energy_initialization, antisymmetrization
)
from crsq.blocks.time_evolution import spec, suzuki_trotter

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

class TimeEvolutionBlock(heap.Frame):
    """ Time Evolution Block """
    def __init__(self,
                 evo_spec: spec.TimeEvolutionSpec,
                 ene_spec: energy_initialization.EnergyConfigurationSpec,
                 asy_spec: antisymmetrization.AntisymmetrizationSpec,
                 label="TEV", allocate=True, build=True, use_motion_block_gates=False):
        super().__init__(label=label)
        self._evo_spec = evo_spec
        self._ene_spec = ene_spec
        self._asy_spec = asy_spec
        self._ham_spec = evo_spec.ham_spec
        self._disc_spec = evo_spec.disc_spec
        self._wfr_spec = self._ham_spec.wfr_spec
        self._use_motion_block_gates = use_motion_block_gates
        # registers
        self._e_index_regs: List[List[QuantumRegister]]
        self._n_index_regs: List[List[QuantumRegister]]
        self._energy_configuration_reg: QuantumRegister = None
        self._slater_indices: list[QuantumRegister] = []
        self._slater_ancilla: QuantumRegister = None
        if allocate:
            self.allocate_registers()
            if build:
                with check_time("TimeEvolutionBlock.build_circuit"):
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
        self._slater_ancilla = asy_spec.allocate_ancilla_register()
        self.add_param(self._slater_ancilla)
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
        evo_spec = self._evo_spec

        self._build_initialization_block()

        n_atom_it = evo_spec.num_atom_iterations
        n_elec_it = evo_spec.num_elec_per_atom_iterations
        with qc.for_loop(range(n_atom_it)):
            with qc.for_loop(range(n_elec_it)):
                if evo_spec.should_calculate_electron_motion:
                    self._build_electron_motion_block()
            if evo_spec.should_calculate_nucleus_motion:
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
                    shuffle=self._slater_ancilla,
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
                    shuffle=self._slater_ancilla
                )
            )

    def _build_electron_motion_block(self):
        if self._use_motion_block_gates:
            elec_motion_block = suzuki_trotter.ElectronMotionBlock(self._evo_spec)
            logger.info("ElectronMotionBlock.num_qubits = %d", elec_motion_block.circuit.num_qubits)
            with check_time("ElectronMotionBlock.invoke"):
                self.invoke(elec_motion_block.bind(
                    eregs=self._e_index_regs,
                    nregs=self._n_index_regs
                ), invoke_as_instruction=True)
            return
        evo_spec = self._evo_spec
        if evo_spec.should_calculate_potential_term:
            self._build_elec_potential_step()
        if evo_spec.should_apply_qft:
            self._build_apply_electron_qft_step(inverse=True)
        if evo_spec.should_calculate_kinetic_term:
            self._build_elec_kinetic_step()
        if evo_spec.should_apply_qft:
            self._build_apply_electron_qft_step()

    def _build_nuclei_motion_block(self):
        if self._use_motion_block_gates:
            nucl_motion_block = suzuki_trotter.NucleusMotionBlock(self._evo_spec)
            with check_time("NucleusMotionBlock.invoke"):
                self.invoke(nucl_motion_block.bind(
                    nregs=self._n_index_regs), invoke_as_instruction=True)
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

    def _build_elec_potential_step(self):
        block = self.build_elec_potential_block()
        with check_time("ElectronPotentialBlock.invoke"):
            self.invoke(block.bind(eregs=self._e_index_regs, nregs=self._n_index_regs))

    def _build_nuclei_potential_step(self):
        block = hamiltonian.NucleusPotentialBlock(self._ham_spec, self._disc_spec)
        with check_time("NucleusPotentialBlock.invoke"):
            self.invoke(block.bind(nregs=self._n_index_regs))

    def build_elec_potential_block(self, allocate=True, build=True):
        """ build a ElectronPotentialBlock instance."""
        block = hamiltonian.ElectronPotentialBlock(
            self._ham_spec, self._disc_spec, allocate=allocate, build=build)
        return block

    def _build_apply_electron_qft_step(self, inverse: bool = False):
        """ apply QFT on all index registers """
        block = qft.QFTOnWaveFunctionsBlock(self._wfr_spec, on_electrons=True, inverse=inverse)
        # The QFT block contains non-gate instructions. 
        with check_time("QFTOnWaveFunctionsBlock(e).invoke"):
            self.invoke(block.bind(
                        eregs=self._e_index_regs),
                        invoke_as_instruction=True)

    def _build_apply_nucleus_qft_step(self, inverse: bool = False):
        """ apply QFT on all index registers """
        block = qft.QFTOnWaveFunctionsBlock(self._wfr_spec, on_nucleus=True, inverse=inverse)
        # The QFT block contains non-gate instructions.
        with check_time("QFTOnWaveFunctionsBlock(n).invoke"):
            self.invoke(block.bind(
                        nregs=self._n_index_regs),
                        invoke_as_instruction=True)

    def _build_elec_kinetic_step(self):
        block = hamiltonian.ElectronKineticBlock(self._wfr_spec, self._disc_spec)
        with check_time("ElectronKineticBlock.invoke"):
            self.invoke(block.bind(eregs=self._e_index_regs))

    def _build_nuclei_kinetic_step(self):
        block = hamiltonian.NucleusKineticBlock(self._wfr_spec, self._disc_spec, self._ham_spec)
        with check_time("NucleusKineticBlock.invoke"):
            self.invoke(block.bind(nregs=self._n_index_regs))

    def bind(self,
             eregs: List[List[QuantumRegister]],
             nregs: List[List[QuantumRegister]],
             slater_indices: List[QuantumRegister],
             slater_ancilla: QuantumRegister,
             p: QuantumRegister
             ) -> heap.Binding:
        """ bind arguments to a binding object """
        arg_map = {
            "eregs": eregs,
            "nregs": nregs,
            "slater_indices": slater_indices,
            "p": p
        }
        if self._slater_ancilla is not None:
            arg_map[self._slater_ancilla.name] = slater_ancilla
        return heap.Binding(self, arg_map)
