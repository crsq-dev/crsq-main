import math, os, argparse
import numpy as np
import scipy.special as sp
from matplotlib import pyplot as plt

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

logger = logging.getLogger('TEV')

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
    def __init__(self, device='GPU', enable_cuStateVec=True, dim=1, precision='single', n1=5, use_saved_data=False):
        self.device = device
        self.enable_cuStateVec = enable_cuStateVec
        self.precision = precision
        self.dim = dim  # 1 dimension
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
        self.use_saved_data = use_saved_data
        self.stm_block = None
    
    def draw_circuits(self, outdir):

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

        self.stm_block = SuzukiTrotterMethodBlock(self.evo_spec, self.ene_spec, self.asy_spec)

        logger.info('draw the circuit')

        fname= outdir + "/h1d.circuit.png"
        self.stm_block.circuit.draw(output='mpl', filename=fname, scale=0.6)

        # draw the circuit

        epot = self.stm_block.build_elec_potential_block()
        fname= outdir + "/h1d.circuit.elec_potential.png"
        epot.circuit.draw(output='mpl', filename=fname, scale=0.6)


    def _make_filename(self, outdir, time):
        dirname = outdir
        basename = "h1d"
        fname = f"{dirname}/{basename}.{self.n1}b.mu{self.x0}t{time}.csv"
        return fname

    def run_circuit(self, outdir):
        # run the simulator
        logger.info('run the simulator')
        #num_iters = [1, 10, 100, 500, 1000, 1500]
        num_iters = [1]
        # num_iters = [10]

        backend = AerSimulator(method="statevector", device=self.device, cuStateVec_enable=self.enable_cuStateVec, precision=self.precision)
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
            fname = self._make_filename(outdir, time)
            logger.info("Saving to : %s", fname)
            svec.save_to_file(fname, sv, eps=1e-12)
    
    def draw_graph(self, outdir, times: list[int]):
        """ draw the graph based on the results file."""
        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(6,7))
        ax1.set_title("amplitude")
        ax2.set_title("phase")
        x = np.linspace(0, self.L, self.M + 1)
        def wrap(x):
            return np.append(x, x[:1])
        for tm in times:
            fname = self._make_filename(outdir, tm)
            logger.info("Reading: %s", fname)
            qc = self.stm_block.circuit
            sv = svec.read_from_file(fname)
            data = svec.extract_dist(qc, sv, "e0x", eps=1e-12)
            norm = np.linalg.norm(data)
            logger.info("norm(t=%d)=%f", tm, norm)
            y = wrap(data)
            a = np.abs(y) / math.sqrt(self.dq)
            th = np.angle(y)/math.pi
            ax1.plot(x, a, label=f"t={tm}")
            ax2.plot(x, th, label=f"t={tm}")
        
        ax1.legend()
        ax2.legend()
        fig.savefig(outdir + "/ex0_dist.png")


def run_experiment(par: Parameters, tag: str):
    outdir = "output/" + tag
    os.makedirs(outdir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(name)s] %(message)s',
                        filename= outdir + '/time_evo_h1.log', encoding='utf-8',
                        level=logging.WARNING,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)

    par.draw_circuits(outdir)
    if par.use_saved_data:
        logger.info('skip running the simulator')
    else:
        par.run_circuit(outdir)
    times = [0]
    par.draw_graph(outdir, times)
    logger.info('done')


if __name__ == "__main__":
    print("pwd: ", os.getcwd())
    parser = argparse.ArgumentParser(prog="time_evo_h1",
                                     description='Time evolution of H atom')
    parser.add_argument('--device', type=str, default="CPU")
    parser.add_argument('--enable-cuStateVec', type=str, default="False")
    parser.add_argument('--dim', type=int, default=1)
    parser.add_argument('--precision', type=str, default="single")
    parser.add_argument('--bits', type=int, default=5)
    parser.add_argument('--use-saved-data', type=str, default="False")
    args = parser.parse_args()
    logger.info("Device : %s", args.device)
    logger.info("enable_cuStateVec : %s", args.enable_cuStateVec)
    logger.info("Dimension : %s", args.dim)
    logger.info("Precision : %s", args.precision)
    use_cuStateVec = "cuStateVec" if args.enable_cuStateVec == 'True' else "statevector"
    tag=f'{args.device}_{use_cuStateVec}_{args.dim}D_{args.precision}_{args.bits}b'
    logger.info("Tag : %s", tag)

    par = Parameters(args.device, args.enable_cuStateVec == "True", args.dim, args.precision, args.bits, args.use_saved_data == "True")
    run_experiment(par, tag)
