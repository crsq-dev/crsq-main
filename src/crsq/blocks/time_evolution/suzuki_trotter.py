"""Time evolution by Suzuki-Trotter decomposition (or split operator method)
"""
from contextlib import contextmanager
from typing import List
import logging
import time
from qiskit import QuantumRegister
from crsq import heap
from crsq.blocks import (
    hamiltonian, qft
)
from crsq.blocks.time_evolution import spec

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


class SuzukiTrotterIntegrator(heap.Frame):
    """ Time Evolution Block """
    def __init__(self,
                 evo_spec: spec.TimeEvolutionSpec,
                 label="ST", allocate=True, build=True, use_motion_block_gates=False):
        super().__init__(label=label)
        self._evo_spec = evo_spec
        self._ham_spec = evo_spec.ham_spec
        self._disc_spec = evo_spec.disc_spec
        self._wfr_spec = evo_spec.wfr_spec
        self._use_motion_block_gates = use_motion_block_gates
        # registers
        self._e_index_regs: List[List[QuantumRegister]]
        self._n_index_regs: List[List[QuantumRegister]]
        self._energy_configuration_reg: QuantumRegister = None
        if allocate:
            self.allocate_registers()
            if build:
                with check_time("SuzukiTrotterIntegrator.build_circuit"):
                    self.build_circuit()

    def allocate_registers(self):
        """ allocate """
        wfr_spec = self._wfr_spec
        self._e_index_regs = wfr_spec.allocate_elec_registers()
        self._n_index_regs = wfr_spec.allocate_nucl_registers()
        self.add_param(("eregs", self._e_index_regs),
                       ("nregs", self._n_index_regs))

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
        qc = self.circuit
        evo_spec = self._evo_spec

        n_atom_it = evo_spec.num_atom_iterations
        n_elec_it = evo_spec.num_elec_per_atom_iterations
        with qc.for_loop(range(n_atom_it)):
            with qc.for_loop(range(n_elec_it)):
                if evo_spec.should_calculate_electron_motion:
                    self._build_electron_motion_block()
            if evo_spec.should_calculate_nucleus_motion:
                self._build_nuclei_motion_block()

    def _build_electron_motion_block(self):
        if self._use_motion_block_gates:
            elec_motion_block = ElectronMotionBlock(self._evo_spec)
            logger.info("ElectronMotionBlock.num_qubits = %d", elec_motion_block.circuit.num_qubits)
            with check_time("ElectronMotionBlock.invoke"):
                self.invoke(elec_motion_block.bind(
                    eregs=self._e_index_regs,
                    nregs=self._n_index_regs
                ), invoke_as_instruction=True)
            return
        # don't use the motion block gates.
        elec_motion_block = ElectronMotionBlock(self._evo_spec, build=False)
        elec_motion_block.build_circuit_on(self)

    def _build_nuclei_motion_block(self):
        if self._use_motion_block_gates:
            nucl_motion_block = NucleusMotionBlock(self._evo_spec)
            with check_time("NucleusMotionBlock.invoke"):
                self.invoke(nucl_motion_block.bind(
                    nregs=self._n_index_regs), invoke_as_instruction=True)
            return
        # don't use the motion block gates.
        nucl_motion_block = NucleusMotionBlock(self._evo_spec, build=False)
        qc = self.circuit
        nucl_motion_block.build_circuit_on(self)

    def bind(self,
             eregs: List[List[QuantumRegister]],
             nregs: List[List[QuantumRegister]]
             ) -> heap.Binding:
        """ bind arguments to a binding object """
        arg_map = {
            "eregs": eregs,
            "nregs": nregs,
        }
        return heap.Binding(self, arg_map)


class ElectronMotionBlock(heap.Frame):
    """ H_ep, QFT, H_ek, QFT\dagger

        Electron motion by Suzuki-Trotter decomposition
    """
    def __init__(self,
                 evo_spec: spec.TimeEvolutionSpec,
                 label=" TEV_e(x)", allocate=True, build=True):
        super().__init__(label=label)
        t1 = time.time()
        self._evo_spec = evo_spec
        self._ham_spec = evo_spec.ham_spec
        self._disc_spec = evo_spec.disc_spec
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
        """ allocate """
        wfr_spec = self._wfr_spec
        self._e_index_regs = wfr_spec.allocate_elec_registers()
        self._n_index_regs = wfr_spec.allocate_nucl_registers()
        self.add_param(("eregs", self._e_index_regs),
                       ("nregs", self._n_index_regs))

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
        evo_spec = self._evo_spec
        if evo_spec.should_calculate_potential_term:
            self._build_elec_potential_step()
        if evo_spec.should_apply_qft:
            self._build_apply_electron_qft_step(inverse=True)
        if evo_spec.should_calculate_kinetic_term:
            self._build_elec_kinetic_step()
        if evo_spec.should_apply_qft:
            self._build_apply_electron_qft_step()

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

    def _build_apply_electron_qft_step(self, inverse: bool = False):
        """ apply QFT on all index registers """
        block = qft.QFTOnWaveFunctionsBlock(self._wfr_spec, on_electrons=True, inverse=inverse)
        # The QFT block contains non-gate instructions.
        with check_time("QFTOnWaveFunctionsBlock(e).invoke"):
            self.invoke(block.bind(
                        eregs=self._e_index_regs),
                        invoke_as_instruction=True)

    def _build_elec_kinetic_step(self):
        block = hamiltonian.ElectronKineticBlock(self._wfr_spec, self._disc_spec)
        with check_time("ElectronKineticBlock.invoke"):
            self.invoke(block.bind(eregs=self._e_index_regs))

    def bind(self,
             eregs: List[List[QuantumRegister]],
             nregs: List[List[QuantumRegister]]):
        """ bind arguments to the function """
        return heap.Binding(self, {
            "eregs": eregs,
            "nregs": nregs
        })


class NucleusMotionBlock(heap.Frame):
    """ H_np, QFT, H_nk, QFT\dagger """
    def __init__(self,
                 evo_spec: spec.TimeEvolutionSpec,
                 label=" TEV_n(x)", allocate=True, build=True):
        super().__init__(label=label)
        self._evo_spec = evo_spec
        self._ham_spec = evo_spec.ham_spec
        self._disc_spec = evo_spec.disc_spec
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
        """ apply QFT on all index registers """
        block = qft.QFTOnWaveFunctionsBlock(self._wfr_spec, on_nucleus=True, inverse=inverse)
        # The QFT block contains non-gate instructions.
        with check_time("QFTOnWaveFunctionsBlock(n).invoke"):
            self.invoke(block.bind(
                        nregs=self._n_index_regs),
                        invoke_as_instruction=True)

    def _build_nuclei_kinetic_step(self):
        block = hamiltonian.NucleusKineticBlock(self._wfr_spec, self._disc_spec, self._ham_spec)
        with check_time("NucleusKineticBlock.invoke"):
            self.invoke(block.bind(nregs=self._n_index_regs))

    def bind(self,
             nregs: List[List[QuantumRegister]]):
        """ bind arguments to the function """
        return heap.Binding(self, {
            "nregs": nregs
        })
