""" Time evolution spec
"""

from crsq.blocks import (
    hamiltonian, discretization
)

SUZUKI_TROTTER='ST'

class TimeEvolutionSpec:
    """ Time Evolution block parameters

        :param method: 'ST' for Suzuki-Trotter decomposition
    """
    def __init__(self,
                 ham_spec: hamiltonian.HamiltonianSpec,
                 disc_spec: discretization.DiscretizationSpec,
                 num_atom_iterations: int, num_elec_per_atom_iterations: int,
                 method=SUZUKI_TROTTER):
        self._ham_spec = ham_spec
        self._disc_spec = disc_spec
        self._num_atom_iterations = num_atom_iterations
        self._num_elec_per_atom_iterations = num_elec_per_atom_iterations
        self._should_calculate_electron_motion = True
        self._should_calculate_nucleus_motion = True
        self._should_calculate_potential_term = True
        self._should_calculate_kinetic_term = True
        self._should_apply_qft = True
        valid_methods = [SUZUKI_TROTTER]
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
        self._method = method

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

    @property
    def method(self):
        """ The method to use for integration
        """
        return self._method