""" Time evolution spec
"""
from qiskit.circuit import Parameter
from crsq.blocks import (
    hamiltonian, discretization, wave_function, rfqhamiltonian
)

SUZUKI_TROTTER_ARITHMETIC='STAR'
SUZUKI_TROTTER_QROM='STQR'

class TimeEvolutionSpec:
    """ Time Evolution block parameters

        :param method: 'ST' for Suzuki-Trotter decomposition
    """
    def __init__(self,
                 ham_spec: hamiltonian.HamiltonianSpec,
                 disc_spec: discretization.DiscretizationSpec,
                 num_atom_iterations: int,
                 num_elec_per_atom_iterations: int,
                 method: str = SUZUKI_TROTTER_ARITHMETIC,
                 rfq_spec: rfqhamiltonian.RfqPotentialSpec = None,
                 save_state_vector_per_atom_iteration: bool = False,
                 save_state_vector_per_qft: bool = False
                 ):
        assert isinstance(ham_spec, hamiltonian.HamiltonianSpec)
        assert isinstance(disc_spec, discretization.DiscretizationSpec)
        assert isinstance(num_atom_iterations, int)
        assert isinstance(num_elec_per_atom_iterations, int)
        assert rfq_spec is None or isinstance(rfq_spec, rfqhamiltonian.RfqPotentialSpec)
        
        self._ham_spec = ham_spec
        self._disc_spec = disc_spec
        self._num_atom_iterations = num_atom_iterations
        self._num_elec_per_atom_iterations = num_elec_per_atom_iterations
        self._should_calculate_electron_motion = True
        self._should_calculate_nucleus_motion = True
        self._should_calculate_potential_term = True
        self._should_calculate_kinetic_term = True
        self._should_save_state_vector_per_atom_iteration = save_state_vector_per_atom_iteration
        self._should_save_state_vector_per_qft = save_state_vector_per_qft
        self._should_apply_qft = True
        valid_methods = [SUZUKI_TROTTER_ARITHMETIC, SUZUKI_TROTTER_QROM]
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
        if method == SUZUKI_TROTTER_QROM and rfq_spec is None:
            raise ValueError("rfq_spec is required when method is SUZUKI_TROTTER_QROM")
        self._method = method
        self._rfq_spec = rfq_spec

    @property
    def ham_spec(self) -> hamiltonian.HamiltonianSpec:
        """ Hamiltonian spec"""
        return self._ham_spec

    @property
    def rfq_spec(self) -> rfqhamiltonian.RfqPotentialSpec:
        """ rfq potential spec """
        return self._rfq_spec

    @property
    def should_use_rfq_gray_code(self) -> bool:
        """ flag that tells rfq hamiltonian should use gray code """
        return self._rfq_spec and self._rfq_spec.should_use_gray_code

    @property
    def wfr_spec(self) -> wave_function.WaveFunctionRegisterSpec:
        """ wave function register spec """
        return self._ham_spec.wfr_spec

    @property
    def disc_spec(self) -> discretization.DiscretizationSpec:
        """ discretization spec """
        return self._disc_spec

    def set_should_calculate_electron_motion(self, flag: bool):
        """ set flag """
        self._should_calculate_electron_motion = flag

    @property
    def should_calculate_electron_motion(self) -> bool:
        """ flag that tells in the time evolution spec, electron motion should bec calculated """
        return self._should_calculate_electron_motion

    def set_should_calculate_nucleus_motion(self, flag: bool):
        """ set flag """
        self._should_calculate_nucleus_motion = flag

    @property
    def should_calculate_nucleus_motion(self) -> bool:
        """ flag that tells nucleus motion should be calculated"""
        return self._should_calculate_nucleus_motion

    def set_should_calculate_potential_term(self, flag: bool):
        """ set flag """
        self._should_calculate_potential_term = flag

    @property
    def should_calculate_potential_term(self) -> bool:
        """ flag that tells potential term should be calculated """
        return self._should_calculate_potential_term

    def set_should_calculate_kinetic_term(self, flag: bool):
        """ set flag """
        self._should_calculate_kinetic_term = flag

    @property
    def should_calculate_kinetic_term(self) -> bool:
        """ flag that tells kinetic term should be calculated """
        return self._should_calculate_kinetic_term

    def set_should_apply_qft(self, flag: bool):
        """ set flag"""
        self._should_apply_qft = flag

    @property
    def should_apply_qft(self) -> bool:
        """ flag that tells QFT should be applied"""
        return self._should_apply_qft

    @property
    def num_atom_iterations(self) -> int:
        """ atomic scale iterations to run """
        return self._num_atom_iterations

    @property
    def num_elec_per_atom_iterations(self) -> int:
        """ electron scale iterations per one atomic scale iteration """
        return self._num_elec_per_atom_iterations

    @property
    def should_save_state_vector_per_atom_iteration(self) -> bool:
        """ flag that tells state vector should be saved per atom iteration """
        return self._should_save_state_vector_per_atom_iteration
    
    def make_state_vector_file_name(self, time: float, suffix = ""):
        """ make state vector file name """
        return f"state_vector_{time:04.3f}{suffix}.csv"

    def make_state_vector_label(self, time: float, suffix = ""):
        """ make state vector label """
        return f"sv_t{time:04.3f}{suffix}"

    @property
    def should_save_state_vector_per_qft(self) -> bool:
        """ flag that tells state vector should be saved per qft """
        return self._should_save_state_vector_per_qft

    @property
    def method(self):
        """ The method to use for integration
        """
        return self._method