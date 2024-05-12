""" Time Evolution module.
"""

from typing import List
import logging
import time
from contextlib import contextmanager
from qiskit import QuantumRegister
from crsq import heap
from crsq.blocks import hamiltonian, discretization, qft, energy_initialization, antisymmetrization

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

class TimeEvolutionSpec:
    """ Time Evolution block parameters """
    def __init__(self,
                 ham_spec: hamiltonian.HamiltonianSpec,
                 disc_spec: discretization.DiscretizationSpec,
                 num_atom_iterations: int, num_elec_per_atom_iterations: int):
        self._ham_spec = ham_spec
        self._disc_spec = disc_spec
        self._num_atom_iterations = num_atom_iterations
        self._num_elec_per_atom_iterations = num_elec_per_atom_iterations
        self._should_calculate_electron_motion = True
        self._should_calculate_nucleus_motion = True
        self._should_calculate_potential_term = True
        self._should_calculate_kinetic_term = True
        self._should_apply_qft = True

    @property
    def ham_spec(self):
        """ Hamiltonian spec"""
        return self._ham_spec

    @property
    def wfr_spec(self):
        """ wave function register spec """
        return self._ham_spec.wfr_spec

    @property
    def disc_spec(self):
        """ discretization spec """
        return self._disc_spec

    def set_should_calculate_electron_motion(self, flag):
        """ set flag """
        self._should_calculate_electron_motion = flag

    @property
    def should_calculate_electron_motion(self):
        """ flag that tells in the time evolution spec, electron motion should bec calculated """
        return self._should_calculate_electron_motion

    def set_should_calculate_nucleus_motion(self, flag):
        """ set flag """
        self._should_calculate_nucleus_motion = flag

    @property
    def should_calculate_nucleus_motion(self):
        """ flag that tells nucleus motion should be calculated"""
        return self._should_calculate_nucleus_motion

    def set_should_calculate_potential_term(self, flag):
        """ set flag """
        self._should_calculate_potential_term = flag

    @property
    def should_calculate_potential_term(self):
        """ flag that tells potential term should be calculated """
        return self._should_calculate_potential_term

    def set_should_calculate_kinetic_term(self, flag):
        """ set flag """
        self._should_calculate_kinetic_term = flag

    @property
    def should_calculate_kinetic_term(self):
        """ flag that tells kinetic term should be calculated """
        return self._should_calculate_kinetic_term

    def set_should_apply_qft(self, flag):
        """ set flag"""
        self._should_apply_qft = flag

    @property
    def should_apply_qft(self):
        """ flag that tells QFT should be applied"""
        return self._should_apply_qft

    @property
    def num_atom_iterations(self):
        """ atomic scale iterations to run """
        return self._num_atom_iterations

    @property
    def num_elec_per_atom_iterations(self):
        """ electron scale iterations per one atomic scale iteration """
        return self._num_elec_per_atom_iterations


class TimeEvolutionBlock(heap.Frame):
    """ Time Evolution Block """
    def __init__(self,
                 evo_spec: TimeEvolutionSpec,
                 ene_spec: energy_initialization.EnergyConfigurationSpec,
                 asy_spec: antisymmetrization.AntisymmetrizationSpec,
                 label="TEV", allocate=True, build=True, use_motion_block_gates=False):
        super().__init__(label=label)
        # logger.info("TimeEvolutionBlock")
        self._evo_spec = evo_spec
        self._ene_spec = ene_spec
        self._asy_spec = asy_spec
        self._ham_spec = evo_spec.ham_spec
        self._disc_spec = evo_spec.disc_spec
        self._wfr_spec = self._ham_spec.wfr_spec
        self._use_motion_block_gates = use_motion_block_gates
        # self._electron_motion_block: ElectronMotionBlock
        # self._electron_potential_block: hamiltonian.ElectronPotentialBlock
        # self._nucleus_motion_block: NucleusMotionBlock
        # registers
        self._e_index_regs: List[List[QuantumRegister]]
        self._n_index_regs: List[List[QuantumRegister]]
        self._energy_configuration_reg: QuantumRegister = None
        # self._energy_configuration_ancilla_reg: QuantumRegister = None
        self._slater_indices: list[QuantumRegister] = []
        self._slater_ancilla: QuantumRegister = None
        if allocate:
            self.allocate_registers()
            if build:
                with check_time("TimeEvolutionBlock.build_circuit"):
                    self.build_circuit()

    # def get_electron_motion_block(self) -> "ElectronMotionBlock":
    #     """ get the constructed electron motion block."""
    #     return self._electron_motion_block

    # def get_electron_potential_block(self) -> hamiltonian.ElectronPotentialBlock:
    #     """ get the electron potential block"""
    #     if hasattr(self, '_electron_motion_block'):
    #         return self._electron_motion_block.get_elec_potential_block()
    #     else:
    #         return self._electron_potential_block

    # def get_nucleus_motion_block(self) -> "NucleusMotionBlock":
    #     """ get the constructed nucleus motion block."""
    #     return self._nucleus_motion_block

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
            # self._energy_configuration_ancilla_reg = QuantumRegister(
            #     ene_spec.num_energy_configuration_bits, "pa")
            self.add_param(
                self._energy_configuration_reg,
                # self._energy_configuration_ancilla_reg
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
                    p=self._energy_configuration_reg,
                    # pa=self._energy_configuration_ancilla_reg
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
            elec_motion_block = ElectronMotionBlock(self._evo_spec)
            logger.info("ElectronMotionBlock.num_qubits = %d", elec_motion_block.circuit.num_qubits)
            with check_time("ElectronMotionBlock.invoke"):
                self.invoke(elec_motion_block.bind(
                    eregs=self._e_index_regs,
                    nregs=self._n_index_regs
                ), invoke_as_instruction=True)
            # self._electron_motion_block = elec_motion_block
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
            nucl_motion_block = NucleusMotionBlock(self._evo_spec)
            with check_time("NucleusMotionBlock.invoke"):
                self.invoke(nucl_motion_block.bind(
                    nregs=self._n_index_regs), invoke_as_instruction=True)
            # self._nucleus_motion_block = nucl_motion_block
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
             p: QuantumRegister,
            #  pa: QuantumRegister
             ) -> heap.Binding:
        """ bind arguments to a binding object """
        arg_map = {
            "eregs": eregs,
            "nregs": nregs,
            "slater_indices": slater_indices,
            "p": p,
            # "pa": pa
        }
        if self._slater_ancilla is not None:
            arg_map[self._slater_ancilla.name] = slater_ancilla
        return heap.Binding(self, arg_map)


class ElectronMotionBlock(heap.Frame):
    """ H_ep, QFT, H_ek, QFT\dagger """
    def __init__(self,
                 evo_spec: TimeEvolutionSpec,
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

    def build_circuit_on(self, other_circuit):
        """ Build the instructions on another compatible quantum circuit."""
        stash = self._circuit
        self._circuit = other_circuit
        self.build_circuit()
        self._circuit = stash

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

    # def get_elec_potential_block(self) -> hamiltonian.ElectronPotentialBlock:
    #     """ get cached electron potential block
    #     """
    #     return self._elec_potential_block

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
                 evo_spec: TimeEvolutionSpec,
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
