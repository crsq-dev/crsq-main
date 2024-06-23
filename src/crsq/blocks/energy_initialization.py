""" Energy initialization block gate
"""

from typing import List
import math
import logging
import time

from qiskit import QuantumRegister
from crsq_heap.heap import Frame, Binding
from crsq.blocks import antisymmetrization, embed

logger = logging.getLogger(__name__)
LOG_TIME_THRESH = 1

class EnergyConfigurationSpec:
    """ Energy initialization spec"""
    def __init__(self,
                 energy_configuration_weights: List[float],
                 initial_electron_orbitals: List[List[List[List[float]]]],
                 initial_nucleus_orbitals: List[List[List[List[float]]]],
                 ):
        self._energy_configuration_weights = energy_configuration_weights
        self._initial_electron_orbitals = initial_electron_orbitals
        self._initial_nucleus_orbitals = initial_nucleus_orbitals
        self._num_energy_configurations = len(energy_configuration_weights)
        self._num_energy_configuration_bits = math.ceil(math.log2(self._num_energy_configurations))
        self._should_initialize_energy = True
        logger.info("EnergyConfigurationSpec: num_energy_configurations = %d", self._num_energy_configurations)

    def set_should_initialize_energy(self, flag):
        """ set whether to initialize energy """
        self._should_initialize_energy = flag
        logger.info("EnergyConfigurationSpec: should_initialize_energy = %d", flag)

    @property
    def should_initialize_energy(self)->bool:
        """ whether to initialize or omit """
        return self._should_initialize_energy

    @property
    def energy_configuration_weights(self):
        """ energy configuration weights """
        return self._energy_configuration_weights

    @property
    def initial_electron_orbitals(self):
        """ initial electron orbitals """
        return self._initial_electron_orbitals

    @property
    def initial_nucleus_orbitals(self):
        """ initial nucleus orbitals """
        return self._initial_nucleus_orbitals

    @property
    def num_energy_configurations(self):
        """ number of energy configurations """
        return len(self._energy_configuration_weights)

    @property
    def num_energy_configuration_bits(self):
        """ num of bits for the energy configuration index register """
        return self._num_energy_configuration_bits


class GeneralStatePreparationBlock(Frame):
    """ General state preparation block

        Contains all elements for energy preparation.

        :param energy_configuration_weights: List of probabilities for each of
            the energy configurations
        :param initial_electron_orbitals: array of shape [energy_conf, electron,
            dimension, position]
        
        :param antisym_method: 1: conventional, 2: unary coded,
            ancilla-shuffle-less
    """
    def __init__(self,
                 ene_spec: EnergyConfigurationSpec,
                 asy_spec: antisymmetrization.AntisymmetrizationSpec,
                 build=True):
        super().__init__(label="Ψg")
        assert isinstance(ene_spec, EnergyConfigurationSpec)
        self._ene_spec = ene_spec
        assert isinstance(asy_spec, antisymmetrization.AntisymmetrizationSpec)
        self._asy_spec = asy_spec
        self._wfr_spec = asy_spec.wfr_spec
        num_conf = ene_spec.num_energy_configurations
        self._num_energy_configurations = num_conf
        self._num_energy_configuration_bits = int(math.ceil(math.log2(num_conf)))
        self._p_reg: QuantumRegister = None
        # self._pa_reg: QuantumRegister = None
        self._eregs: List[List[QuantumRegister]] = []
        self._nregs: List[List[QuantumRegister]] = []
        self._sigma_regs: List[QuantumRegister]
        self._shuffle_ancilla: QuantumRegister
        self.allocate_registers()
        if build:
            self.build_circuit()

    def allocate_registers(self):
        """ Allocate registers """
        self._eregs = self._wfr_spec.allocate_elec_registers()
        self._nregs = self._wfr_spec.allocate_nucl_registers()
        self.add_param(('eregs', self._eregs),
                       ('nregs', self._nregs))
        asy_spec = self._asy_spec
        self._sigma_regs = asy_spec.allocate_sigma_regs()
        self.add_param(('bregs', self._sigma_regs))
        self._shuffle_ancilla = asy_spec.allocate_ancilla_register()
        self.add_param(self._shuffle_ancilla)
        if self._num_energy_configurations > 1:
            p = QuantumRegister(self._num_energy_configuration_bits, "p")
            # pa = QuantumRegister(self._num_energy_configuration_bits, "pa")
            # self.add_param(p, pa)
            self.add_param(p)
        else:
            p = None
            # pa = None
        self._p_reg = p
        # self._pa_reg = pa


    def build_circuit(self):
        """ Build circuit """
        if not self._ene_spec.should_initialize_energy:
            return
        self._set_initial_orbitals()

    def _set_initial_orbitals(self):
        if self._num_energy_configurations == 1:
            state_idx = 0
            state_init_block = self._build_energy_state_initialization_block(state_idx)
            self.invoke(state_init_block.bind(
                eregs=self._eregs, nregs=self._nregs,
                bregs=self._sigma_regs, shuffle=self._shuffle_ancilla))
        else:
            preparation_block = self._build_energy_state_bits_preparation_block()
            # self.invoke(preparation_block.bind(p=self._p_reg, pa=self._pa_reg))
            self.invoke(preparation_block.bind(p=self._p_reg))
            num_conf = self._num_energy_configurations
            num_conf_bits = self._num_energy_configuration_bits
            for state_idx in range(num_conf):
                state_init_block = self._build_energy_state_initialization_block(state_idx)
                ctrl_str = bin((1 << num_conf_bits) + state_idx)[-num_conf_bits:]
                ctrl_bits = self._p_reg[:]
                self.invoke_with_control(
                    state_init_block.bind(
                        eregs=self._eregs, nregs=self._nregs,
                        bregs=self._sigma_regs, shuffle=self._shuffle_ancilla),
                    ctrl_bits, ctrl_str)

    def _build_energy_state_initialization_block(
            self, state_idx: int) -> "SlaterDeterminantPreparationBlock":
        """ build sub block """
        ene_spec = self._ene_spec
        asy_spec = self._asy_spec
        block = SlaterDeterminantPreparationBlock(ene_spec, asy_spec, state_idx)
        return block

    def _build_energy_state_bits_preparation_block(self) -> "EnergyConfigBitsPreparationBlock":
        """ build sub block """
        ene_spec = self._ene_spec
        block = EnergyConfigBitsPreparationBlock(
            ene_spec.energy_configuration_weights)
        return block

    def bind(self,
             eregs: List[List[List[QuantumRegister]]],
             nregs: List[List[List[QuantumRegister]]],
             bregs: List[QuantumRegister],
             shuffle: QuantumRegister|None,
             p: QuantumRegister,
            #  pa: QuantumRegister) -> Binding:
             ) -> Binding:
        """ Prepare invocable """
        return Binding(self, {
            "eregs": eregs,
            "nregs": nregs,
            "bregs": bregs,
            "shuffle": shuffle,
            "p": p,
            # "pa": pa
            })


class EnergyConfigBitsPreparationBlock(Frame):
    """ Create energy state bits.
        moved from FirstQIntegrator._build_energy_state_bits_preparation_block
    """
    def __init__(self, energy_configuration_weights: List[float], build=True):
        super().__init__(label="ρ")
        self._energy_configuration_weights = energy_configuration_weights
        num_conf = len(energy_configuration_weights)
        self._num_energy_configurations = num_conf
        self._num_energy_configuration_bits = int(math.ceil(math.log2(num_conf)))
        self._conf_reg: QuantumRegister = None
        # self._conf_ancilla_reg: QuantumRegister = None
        self.allocate_registers()
        if build:
            self.build_circuit()

    def allocate_registers(self):
        """ allocate registers """
        if self._num_energy_configuration_bits == 0:
            return
        self._conf_reg = QuantumRegister(self._num_energy_configuration_bits, "p")
        # self._conf_ancilla_reg = QuantumRegister(self._num_energy_configuration_bits, "pa")
        # self.add_param(self._conf_reg, self._conf_ancilla_reg)
        self.add_param(self._conf_reg)

    def build_circuit(self):
        """ build the circuit """
        if self._num_energy_configuration_bits == 0:
            return
        ary = self._energy_configuration_weights
        emb_gate = embed.StateEmbedGate(ary)
        self.invoke(emb_gate.bind(q=self._conf_reg))
        # for i in range(self._num_energy_configuration_bits):
        #     self.circuit.cx(self._conf_reg[i], self._conf_ancilla_reg[i])

    # def bind(self, p: QuantumRegister|None, pa: QuantumRegister|None) -> Binding:
    def bind(self, p: QuantumRegister|None) -> Binding:
        """ prepare invocable """
        arg_dict = {}
        if p is not None:
            arg_dict["p"] = p
        # if pa is not None:
        #     arg_dict["pa"] = pa
        return Binding(self, arg_dict)


class SlaterDeterminantPreparationBlock(Frame):
    """ Create a single energy state
        moved from FirstQIntegrator._build_energy_state_initialization_gate
    """
    def __init__(self,
                 ene_spec: EnergyConfigurationSpec,
                 asy_spec: antisymmetrization.AntisymmetrizationSpec,
                 energy_state_index: int,
                 build=True):
        """
            :param energy_state_index: which energy state to build
        """
        super().__init__(label="Ψsd")
        t1 = time.time()
        # logger.info("start: SlaterDeterminantPreparationBlock()")
        assert isinstance(ene_spec, EnergyConfigurationSpec)
        self._ene_spec = ene_spec
        assert isinstance(asy_spec, antisymmetrization.AntisymmetrizationSpec)
        self._asy_spec = asy_spec
        self._wfr_spec = asy_spec.wfr_spec
        self._energy_state_index = energy_state_index
        self._e_index_regs: List[List[List[QuantumRegister]]] = []
        self._n_index_regs: List[List[List[QuantumRegister]]] = []
        self._sigma_regs: List[QuantumRegister]
        self._shuffle_ancilla: QuantumRegister
        self.allocate_registers()
        if build:
            self.build_circuit()
        t2 = time.time()
        if t2 - t1 > LOG_TIME_THRESH:
            logger.info("end  : SlaterDeterminantPreparationBlock(%f msec)", round((t2-t1)*1000))

    @property
    def _dimension(self):
        """ dimension of the model """
        return self._wfr_spec.dimension

    @property
    def _num_electrons(self):
        """ number of electrons """
        return self._wfr_spec.num_electrons

    @property
    def _num_nuclei(self):
        """ number of nuclei """
        return self._wfr_spec.num_moving_nuclei

    def allocate_registers(self):
        """ allocate registers """
        wfr_spec = self._wfr_spec
        eregs = wfr_spec.allocate_elec_registers()
        nregs = wfr_spec.allocate_nucl_registers()
        self._e_index_regs = eregs
        self._n_index_regs = nregs
        self.add_param(('eregs', eregs),('nregs', nregs))
        asy_spec = self._asy_spec
        self._sigma_regs = asy_spec.allocate_sigma_regs()
        self.add_param(('bregs', self._sigma_regs))
        self._shuffle_ancilla = asy_spec.allocate_ancilla_register()
        self.add_param(self._shuffle_ancilla)

    def build_circuit(self):
        """ build the circuit """
        if not self._ene_spec.should_initialize_energy:
            return
        self._set_orbital_data()
        wfr_spec = self._wfr_spec
        asy_spec = self._asy_spec
        if wfr_spec.num_electrons >= 2 and asy_spec.should_do_anti_symmetrization:
            self._build_antisymmetrization()

    def _set_orbital_data(self):
        ene_spec = self._ene_spec
        electrons = ene_spec.initial_electron_orbitals[self._energy_state_index]
        for i in range(self._num_electrons):
            e = electrons[i]
            for d, _t in enumerate(e):
                array = e[d]
                reg = self._e_index_regs[i][d]
                emb = embed.StateEmbedGate(array)
                self.invoke(emb.bind(q=reg))
                # setdist.setdist(qc, reg, array)

        nuclei = ene_spec.initial_nucleus_orbitals[self._energy_state_index]
        for a in range(self._num_nuclei):
            n = nuclei[a]
            for d, _t in enumerate(n):
                array = n[d]
                reg = self._n_index_regs[a][d]
                emb = embed.StateEmbedGate(array)
                self.invoke(emb.bind(q=reg))
                # setdist.setdist(qc, reg, array)

    def _build_antisymmetrization(self):
        """ build the antisymmetrization block """
        method = self._asy_spec.method
        wfr_spec = self._wfr_spec
        asy_spec = self._asy_spec
        block: Frame
        logger.info("SlaterDeterminantPreparationBlock: assymetrization method = %d", method)
        if method == 1:
            block = antisymmetrization.ABRegisterPermutationBlock(wfr_spec)
            self.invoke(block.bind(sigmas=self._sigma_regs,
                                   ancilla=self._shuffle_ancilla, eregs=self._e_index_regs))
        elif method == 2:
            block = antisymmetrization.UnaryCodedPermutationBlock(asy_spec)
            self.invoke(block.bind(eregs=self._e_index_regs, aregs=self._sigma_regs))
        elif method == 3:
            block = antisymmetrization.BinaryCodedPermutationBlock(asy_spec)
            self.invoke(block.bind(eregs=self._e_index_regs,
                                   swap=self._shuffle_ancilla, aregs=self._sigma_regs))
        else:
            raise ValueError(f"Unknown method number {method}")

    def bind(self,
             eregs: List[List[List[QuantumRegister]]],
             nregs: List[List[List[QuantumRegister]]],
             bregs: List[QuantumRegister],
             shuffle: QuantumRegister|None
             ) -> Binding:
        """ prepare invocable """
        return Binding(self, {
            "eregs": eregs,
            "nregs": nregs,
            "bregs": bregs,
            "shuffle": shuffle
            })
