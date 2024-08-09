import math, os, argparse
import numpy as np
import scipy.special as sp

from crsq.blocks.antisymmetrization import AntisymmetrizationSpec
from crsq.blocks.discretization import DiscretizationSpec
from crsq.blocks.energy_initialization import EnergyConfigurationSpec
from crsq.blocks.hamiltonian import HamiltonianSpec
from crsq.blocks.time_evolution.spec import TimeEvolutionSpec
from crsq.blocks.wave_function import WaveFunctionRegisterSpec
from crsq.blocks.time_evolution.suzuki_trotter import SuzukiTrotterMethodBlock

from qiskit_aer import AerSimulator
from qiskit import transpile
import crsq.utils.statevector as svec

import logging

# make one dimentional wave function data for H atom.

def hydrogen1d_psi(xa: np.ndarray,x0: float,N: int):
    hbar = 1
    me = 1
    qe = 1
    a0 = 1
    a0 = hbar*hbar/(me*qe*qe)
    x = xa - x0
    absx = np.abs(x)
    A = np.sqrt(2/((a0**3)*(N**5)*math.factorial(N)**2))
    lg = sp.assoc_laguerre(2*absx/(N*a0), N-1, 1)
    psi = A*x*np.exp(-absx/(N*a0))*lg
    return psi

# build the simulator

class Parameters:
    def __init__(self, device='GPU', cuStateVec_enable=True, precision='single', n1=5):
        self.device = device
        self.cuStateVec_enable = cuStateVec_enable
        self.precision = precision
        self.dim = 1  # 1 dimension
        self.n1 = n1 # bits per coordinate
        self.M=1<<n1
        self.L = 16 # 16 bohr
        self.dq = self.L/self.M
        self.eta = 1 # num of electrons
        self.Ln = 0 # moving nucleus
        self.Ls = 1 # stationary nucleus
        self.antisym_method = 3 # binary coded antisymmetrization method
        self.wfr_spec = WaveFunctionRegisterSpec(self.dim, self.n1, self.L, self.eta, self.Ln, self.Ls)

        self.delta_t = 0.001 # a.u.
        self.disc_spec = DiscretizationSpec(self.delta_t)
        self.asy_spec = AntisymmetrizationSpec(self.wfr_spec, self.antisym_method)
        self.nuclei_data = [{"mass": 1680, "charge": 1, "pos": (self.M//2, self.M//2)}]

        self.ham_spec = HamiltonianSpec(self.wfr_spec, nuclei_data=self.nuclei_data)
    
    def draw_circuits(self, outdir):
        logger = logging.getLogger('TEV')


        self.x = np.linspace(0, self.L-self.dq, self.M)
        self.x0 = 8
        self.psix = hydrogen1d_psi(self.x, self.x0, N=1)
        # ini_dims = [psix, psiy]
        ini_dims = [self.psix]
        ini_electrons = [ini_dims]
        ini_configs = [ini_electrons]
        initial_electron_orbitals = ini_configs

        initial_nucleus_orbitals = [[]]
        self.ene_spec = EnergyConfigurationSpec([1], initial_electron_orbitals, initial_nucleus_orbitals)

        #psix = np.zeros(M)
        # psiy = np.zeros(M)

        self.num_nucl_it = 1 # number of nucleus iterations
        self.num_elec_it = 10 # number of electron iterations
        self.evo_spec = TimeEvolutionSpec(self.ham_spec, self.disc_spec, self.num_nucl_it, self.num_elec_it)

        stm = SuzukiTrotterMethodBlock(self.evo_spec, self.ene_spec, self.asy_spec)

        logger.info('draw the circuit')

        fname= outdir + "/h1d.circuit.png"
        stm.circuit.draw(output='mpl', filename=fname, scale=0.6)

        # draw the circuit

        epot = stm.build_elec_potential_block()
        fname= outdir + "/h1d.circuit.elec_potential.png"
        epot.circuit.draw(output='mpl', filename=fname, scale=0.6)


    def run_circuit(self, outdir):
        # run the simulator
        logger = logging.getLogger('TEV')

        logger.info('run the simulator')

        run_calculation=True
        # run_calculation=False

        dirname= outdir

        basename="h1d"

        #num_iters = [1, 10, 100, 500, 1000, 1500]
        num_iters = [1]
        # num_iters = [10]

        # backend = AerSimulator(method="statevector", device="GPU", cuStateVec_enable=False, precision="single")
        backend = AerSimulator(method="statevector", device=self.device, cuStateVec_enable=self.cuStateVec_enable, precision=self.precision)
        backend.set_options(max_parallel_threads=0)
        for num_elec_it in num_iters:
            num_nucl_it = 1 # number of nucleus iterations
            evo_spec = TimeEvolutionSpec(self.ham_spec, self.disc_spec, num_nucl_it, num_elec_it)
            stm = SuzukiTrotterMethodBlock(evo_spec, self.ene_spec, self.asy_spec)

            circ = stm.circuit
            circ.save_statevector()
            logger.info('transpile START')
            transpiled = transpile(circ, backend)
            logger.info('transpile END, run START')
            results = backend.run(transpiled).result()
            logger.info('run END')
            sv = results.get_statevector()
            time=num_elec_it*self.delta_t
            fname = f"{dirname}/{basename}.{self.n1}b.mu{self.x0}t{time}.csv"
            logger.info("Saving to : %s", fname)
            svec.save_to_file(fname, sv, eps=1e-12)


def run_experiment(par: Parameters, tag):
    outdir = "output/" + tag
    os.makedirs(outdir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(name)s] %(message)s',
                        filename= outdir + '/time_evo_h1.log', encoding='utf-8',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger('TEV')
    logger.setLevel(logging.INFO)

    par.draw_circuits(outdir)
    par.run_circuit(outdir)
    logger.info('done')


if __name__ == "__main__":
    print("pwd: ", os.getcwd())
    parser = argparse.ArgumentParser(prog="time_evo_h1",
                                     description='Time evolution of H atom')
    parser.add_argument('--device', type=str, default="CPU")
    parser.add_argument('--cuStateVec_enable', type=str, default="False")
    parser.add_argument('--precision', type=str, default="single")
    parser.add_argument('--bits', type=int, default=5)
    args = parser.parse_args()
    print("Device : ", args.device)
    print("cuStateVec_enable : ", args.cuStateVec_enable)
    print("Precision : ", args.precision)
    use_cuStateVec = "cuStateVec" if args.cuStateVec_enable == 'True' else "statevector"
    tag=f'{args.device}_{use_cuStateVec}_{args.precision}_{args.bits}b'
    print("Tag : ", tag)

    par = Parameters(args.device, args.cuStateVec_enable == 'True', args.precision, args.bits)
    run_experiment(par, tag)
