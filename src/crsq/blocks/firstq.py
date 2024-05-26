
from contextlib import contextmanager
import logging
import time

from qiskit import QuantumRegister
from crsq import heap
from crsq.blocks import antisymmetrization, energy_initialization, time_evolution

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


class Simulator(heap.Frame):
    """First quantization based simulator
    """
    def __init__(self,
                 evo_spec: time_evolution.TimeEvolutionSpec,
                 ene_spec: energy_initialization.EnergyConfigurationSpec,
                 asy_spec: antisymmetrization.AntisymmetrizationSpec,
                 label="Sim", allocate=True, build=True,
                 use_motion_block_gates=False):
        super().__init__(label=label)
        self._evo_spec = evo_spec
        self._ene_spec = ene_spec
        self._asy_spec = asy_spec
        self._ham_spec = evo_spec.ham_spec
        self._disc_spec = evo_spec.disc_spec
        self._wfr_spec = self._ham_spec.wfr_spec
        self._use_motion_block_gates = use_motion_block_gates
        if allocate:
            self.allocate_registers()
            if build:
                with check_time("Simulator.build_circuit"):
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
        """ build gates for time evolution.
        """
        self._build_initialization_block()
        self._build_time_evolution_block()

    def _build_initialization_block(self):
        """ build initialization block"""
        if self._ene_spec.num_energy_configurations > 1:
            self._initialize_with_general_state()
        else:
            self._initialize_with_single_sd_state()

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

    def _initialize_with_single_sd_state(self):
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

    def _build_time_evolution_block(self):
        """ build time evolution block"""
        evo_spec = self._evo_spec
        if evo_spec.method == time_evolution.SUZUKI_TROTTER:
            self._build_time_evolution_with_suzuki_trotter()
    
    def _build_time_evolution_with_suzuki_trotter(self):
        if self._use_motion_block_gates:
            st_block = time_evolution.SuzukiTrotterIntegrator(self._evo_spec)
            with check_time("SuzukiTrotterIntegrator.invoke"):
                self.invoke(
                    st_block.bind(
                        eregs=self._e_index_regs,
                        nregs=self._n_index_regs
                    ),
                    invoke_as_instruction=True
                )
            return
        st_block = time_evolution.SuzukiTrotterIntegrator(self._evo_spec, use_motion_block_gates=False)
        st_block.build_circuit_on(self)
        
