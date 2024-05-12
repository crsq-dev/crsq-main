""" exploratory tests for firstq integrator
"""
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from qiskit import transpile

import crsq.top.firstq as tmr
import crsq.utils.statevector as svec

def setup():
    """ Set up"""
    ti = tmr.FirstQIntegrator()
    ti.set_dimension(1)
    m = 3
    M = 2**m
    ti.set_coordinate_bits(m)
    ti.set_calculate_electron_motion(True)
    ti.set_calculate_nucleus_motion(False)
    ti.set_delta_t(0.1)
    ti.set_energy_configuration_weights([1])
    ti.set_num_energy_configurations(1)
    ti.set_num_particles(1,1)
    ti.set_num_atom_iterations(1)
    ti.set_num_elec_per_atom_iterations(1)
    e0_spatial_x_orb = [0]*M
    e0_spatial_x_orb[M//4] = 1.0
    # e0_spins = [1.0, 0]
    # e0_orb = [e0_spatial_x_orb, e0_spins]
    e0_orb = [e0_spatial_x_orb]
    elec_orbs = [e0_orb]
    elec_configs = [elec_orbs]
    ti.set_initial_electron_orbitals(elec_configs)
    n0_spatial_x_orb = [0]*M
    n0_spatial_x_orb[M//2] = 1.0
    n0_spatial_orb = [n0_spatial_x_orb]
    nuc_configs = [[n0_spatial_orb]]
    ti.set_initial_nucleus_orbitals(nuc_configs)
    ti.make_dummy_nuclei_data()
    return ti


def run_sim_and_dump(circ: QuantumCircuit, label: str):
    """ run simulator and dump the statevector """
    backend = AerSimulator()
    trans = transpile(circ, backend)
    result = backend.run(trans).result()
    sv = result.data()[label]
    svec.dump_statevector(sv, circ)


def run_init(ti: tmr.FirstQIntegrator):
    """ run the initialization step only """
    ti.calculate_sizes()
    ti.allocate_registers()
    ti.build_circuit()
    ti.circuit().save_statevector("after_elec_potential_step")
    run_sim_and_dump(ti.circuit(), "after_elec_potential_step")


def test_firstq():
    """ Test FirstQIntegrator """
    ti = setup()
    run_init(ti)

if __name__ == '__main__':
    test_firstq()
