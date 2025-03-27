""" Wave Function support class module
"""
from typing import List
import math
import logging
import numpy as np
from qiskit import QuantumRegister

logger = logging.getLogger(__name__)

class WaveFunctionRegisterSpec:
    """ class that holds size specs of wave functions and can
        generate wave function registers.
    """
    def __init__(self,
                 dimension: int,
                 num_coordinate_bits: int,
                 space_length: float,
                 num_electrons: int,
                 num_moving_nuclei: int,
                 num_stationary_nuclei: int,
                 use_spin = False,
                 i_to_x=None,
                 j_to_y=None,
                 k_to_z=None):
        """ constructor """
        self._dimension = dimension
        self._num_coordinate_bits = num_coordinate_bits
        self._space_length = space_length
        self._num_electrons = num_electrons
        self._num_moving_nuclei = num_moving_nuclei
        self._num_stationary_nuclei = num_stationary_nuclei
        self._use_spin = use_spin
        self._num_orbital_index_bits =(
            int(math.ceil(math.log2(num_electrons))))
        M = 2**num_coordinate_bits
        self._delta_q = space_length/M
        self._delta_k = 2*math.pi/space_length
        self._i_to_x = i_to_x
        self._j_to_y = j_to_y
        self._k_to_z = k_to_z
        logger.info("WaveFunctionRegisterSpec: dimension = %d", dimension)
        logger.info("WaveFunctionRegisterSpec: num_coordinate_bits = %d", num_coordinate_bits)
        logger.info("WaveFunctionRegisterSpec: space_length = %f", space_length)
        logger.info("WaveFunctionRegisterSpec: num_electrons = %d", num_electrons)
        logger.info("WaveFunctionRegisterSpec: num_moving_nuclei = %d", num_moving_nuclei)
        logger.info("WaveFunctionRegisterSpec: num_stationary_nuclei = %d", num_stationary_nuclei)
        logger.info("WaveFunctionRegisterSpec: use_spin = %d", use_spin)
        logger.info("WaveFunctionRegisterSpec: num_orbital_index_bits = %d",
                    self._num_orbital_index_bits)


    @property
    def dimension(self) -> int:
        """ dimension of the model """
        return self._dimension

    @property
    def num_coordinate_bits(self) -> int:
        """ number of coordinate bits"""
        return self._num_coordinate_bits

    @property
    def space_length(self) -> float:
        """ length of the simulated volume, along 1 dimention """
        return self._space_length

    @property
    def delta_q(self) -> float:
        """ list of dq. dq is the quantization unit of spatial length """
        return self._delta_q

    @property
    def delta_k(self) -> float:
        """ return dk. dk is the quantization unit of momentum space """
        return self._delta_k

    @property
    def num_electrons(self) -> int:
        """ number of electrons """
        return self._num_electrons

    @property
    def num_orbital_index_bits(self) -> int:
        """ number of bits to represent orbital index """
        return self._num_orbital_index_bits

    @property
    def num_nuclei(self) -> int:
        """ number of total nuclei"""
        return self._num_moving_nuclei + self._num_stationary_nuclei

    @property
    def num_stationary_nuclei(self) -> int:
        """ number of nuclei to be treated with B.O.A."""
        return self._num_stationary_nuclei

    @property
    def num_moving_nuclei(self) -> int:
        """ number of nuclei represented as wave functions"""
        return self._num_moving_nuclei

    @property
    def use_spin(self) -> bool:
        """ whether to use spin """
        return self._use_spin

    @property
    def has_elec_potential_term(self) -> bool:
        """ has electron-electron or electron-nucleus potential term """
        return (self._num_electrons >= 1 and self.num_nuclei >= 1) or self.num_electrons >= 2
    
    @property
    def i_to_x(self):
        """ mapping from i to x """
        return self._i_to_x
    
    @property
    def j_to_y(self):
        """ mapping from j to y """
        return self._j_to_y
    
    @property
    def k_to_z(self):
        """ mapping from k to z """
        return self._k_to_z

    def allocate_elec_registers(self) -> List[List[QuantumRegister]]:
        """ allocate electron index registers"""
        return self.allocate_elec_registers_upto(self._num_electrons)

    def allocate_elec_registers_upto(self, subset_size: int) -> List[List[QuantumRegister]]:
        """ allocate subset of electron registers"""
        electrons = []
        dim_labels = 'xyz'
        for i in range(subset_size):
            dims = []
            for d in range(self._dimension):
                dims.append(QuantumRegister(self._num_coordinate_bits, f"{dim_labels[d]}q{i}"))
            # add the spin register
            if self._use_spin:
                dims.append(QuantumRegister(1, f"sq{i}"))
            electrons.append(dims)
        return electrons

    def allocate_nucl_registers(self) -> List[List[QuantumRegister]]:
        """ allocate nucleus registers """
        nuclei = []
        dim_labels='XYZ'
        for i in range(self._num_moving_nuclei):
            dims = []
            for d in range(self._dimension):
                dims.append(QuantumRegister(self._num_coordinate_bits, f"{dim_labels[d]}{i}q"))
            nuclei.append(dims)
        return nuclei

    def allocate_antisymmetrization_ancilla_registers(self):
        """ allocate ancilla registers used by the antisymmetrization block"""

def make_null_elec_orbitals(num_ene_conf: int) -> List[List[List[List[float]]]]:
    """ make test values for electron wave functions when
        energy initialization is turned off.
    """
    configs = []
    for _conf in range(num_ene_conf):
        electrons:List[np.ndarray] = []
        configs.append(electrons)
    return configs

def make_test_elec_orbitals(wfr_spec: WaveFunctionRegisterSpec,
                          num_ene_conf: int) -> List[List[List[List[float]]]]:
    """ make test values for electron wave functions
        using cosine curves.
    """
    num_electrons = wfr_spec.num_electrons
    num_positions = 1 << wfr_spec.num_coordinate_bits
    dim = wfr_spec.dimension
    configs = []
    for _conf in range(num_ene_conf):
        electrons = []
        for elec in range(num_electrons):
            if dim == 1:
                array = np.zeros(num_positions, dtype=complex)
                for x in range(num_positions):
                    k = 2*math.pi/num_positions*(2**elec)
                    phi = 0.5 + 0.5*math.cos(k*x)
                    array[x] = phi
            elif dim == 2:
                array = np.zeros((num_positions, num_positions), dtype=complex)
                for x in range(num_positions):
                    for y in range(num_positions):
                        k = 2*math.pi/num_positions*(2**elec)
                        phi = 0.5 + 0.5*math.cos(k*x)*math.cos(k*y)
                        array[x, y] = phi
            elif dim == 3:
                array = np.zeros((num_positions, num_positions, num_positions), dtype=complex)
                for x in range(num_positions):
                    for y in range(num_positions):
                        for z in range(num_positions):
                            k = 2*math.pi/num_positions*elec
                            phi = 0.5 + 0.5*math.cos(k*x)*math.cos(k*y)*math.cos(k*z)
                            array[x, y, z] = phi
            electrons.append(array)
        configs.append(electrons)
    return configs


def make_null_nucl_orbitals(num_ene_conf: int) -> List[List[List[List[float]]]]:
    """ make test values for electron wave functions when
        energy initialization is turned off.
    """
    configs = []
    for _conf in range(num_ene_conf):
        nuclei:List[np.ndarray] = []
        configs.append(nuclei)
    return configs

def make_test_nucl_orbitals(wfr_spec: WaveFunctionRegisterSpec,
                          num_ene_conf: int) -> List[List[List[List[float]]]]:
    """ make test values for electron wave functions
        using cosine curves.
    """
    num_nuclei = wfr_spec.num_moving_nuclei
    num_positions = 1 << wfr_spec.num_coordinate_bits
    dim = wfr_spec.dimension
    configs = []
    for _conf in range(num_ene_conf):
        nuclei = []
        for nucl in range(num_nuclei):
            if dim == 1:
                array = np.zeros(num_positions, dtype=complex)
                for x in range(num_positions):
                    k = 2*math.pi/num_positions*(2**nucl)
                    phi = 0.5 + 0.5*math.cos(k*x)
                    array[x] = phi
            elif dim == 2:
                array = np.zeros((num_positions, num_positions), dtype=complex)
                for x in range(num_positions):
                    for y in range(num_positions):
                        k = 2*math.pi/num_positions*(2**nucl)
                        phi = 0.5 + 0.5*math.cos(k*x)*math.cos(k*y)
                        array[x, y] = phi
            elif dim == 3:
                array = np.zeros((num_positions, num_positions, num_positions), dtype=complex)
                for x in range(num_positions):
                    for y in range(num_positions):
                        for z in range(num_positions):
                            k = 2*math.pi/num_positions*(2**nucl)
                            phi = 0.5 + 0.5*math.cos(k*x)*math.cos(k*y)*math.cos(k*z)
                            array[x, y, z] = phi
            nuclei.append(array)
        configs.append(nuclei)
    return configs
