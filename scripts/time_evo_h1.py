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

logger = logging.getLogger("TEV")

# make one dimentional wave function data for H atom.


def hydrogen1d_psi(xa: np.ndarray, x0: float, N: int):
    hbar = 1
    me = 1
    qe = 1
    a0 = 1
    a0 = hbar * hbar / (me * qe * qe)
    x = xa - x0
    absx = np.abs(x)
    A = np.sqrt(2 / ((a0**3) * (N**5) * math.factorial(N) ** 2))
    lg = sp.assoc_laguerre(2 * absx / (N * a0), N - 1, 1)
    psi = A * x * np.exp(-absx / (N * a0)) * lg
    return psi


# build the simulator


class Parameters:
    def __init__(
        self,
        outdir="output/default",
        device="GPU",
        enable_cuStateVec=True,
        dim=1,
        precision="single",
        n1=5,
        num_nucl_iters=1,
        num_elec_iters=1,
        use_saved_data=False,
    ):
        self.outdir = outdir
        self.device = device
        self.enable_cuStateVec = enable_cuStateVec
        self.precision = precision
        self.dim = dim  # 1 dimension
        self.n1 = n1  # bits per coordinate
        self.M = 1 << n1
        self.L = 16  # 16 bohr
        self.dq = self.L / self.M
        self.eta = 1  # num of electrons
        self.Ln = 0  # moving nucleus
        self.Ls = 1  # stationary nucleus
        self.num_nucl_iters = num_nucl_iters
        self.num_elec_iters = num_elec_iters
        self.antisym_method = 3  # binary coded antisymmetrization method
        self.wfr_spec = WaveFunctionRegisterSpec(
            self.dim, self.n1, self.L, self.eta, self.Ln, self.Ls
        )

        self.delta_t = 0.001  # a.u.
        self.disc_spec = DiscretizationSpec(self.delta_t)
        self.asy_spec = AntisymmetrizationSpec(self.wfr_spec, self.antisym_method)
        self.nuclei_data = [
            {"mass": 1680, "charge": 1, "pos": (self.M // 2, self.M // 2)}
        ]

        self.ham_spec = HamiltonianSpec(self.wfr_spec, nuclei_data=self.nuclei_data)
        self.use_saved_data = use_saved_data
        self.stm_block = None

    def draw_circuits(self):

        self.x = np.linspace(0, self.L - self.dq, self.M)
        self.x0 = 8
        self.psix = hydrogen1d_psi(self.x, self.x0, N=1)
        # ini_dims = [psix, psiy]
        ini_dims = [self.psix]
        ini_electrons = [ini_dims]
        ini_configs = [ini_electrons]
        initial_electron_orbitals = ini_configs

        initial_nucleus_orbitals = [[]]
        self.ene_spec = EnergyConfigurationSpec(
            [1], initial_electron_orbitals, initial_nucleus_orbitals
        )

        # psix = np.zeros(M)
        # psiy = np.zeros(M)

        self.evo_spec = TimeEvolutionSpec(
            self.ham_spec,
            self.disc_spec,
            self.num_nucl_iters,
            self.num_elec_iters,
            save_state_vector_per_atom_iteration=False,
        )

        self.stm_block = SuzukiTrotterMethodBlock(
            self.evo_spec, self.ene_spec, self.asy_spec
        )

        logger.info("draw the circuit")

        fname = self.outdir + "/h1d.circuit.png"
        self.stm_block.circuit.draw(output="mpl", filename=fname, scale=0.6)

        # draw the circuit

        epot = self.stm_block.build_elec_potential_block()
        fname = self.outdir + "/h1d.circuit.elec_potential.png"
        epot.circuit.draw(output="mpl", filename=fname, scale=0.6)

    def run_circuit(self):
        # run the simulator
        logger.info("run the simulator")

        backend = AerSimulator(
            method="statevector",
            device=self.device,
            cuStateVec_enable=self.enable_cuStateVec,
            precision=self.precision,
        )
        backend.set_options(max_parallel_threads=0)

        evo_spec = TimeEvolutionSpec(
            self.ham_spec,
            self.disc_spec,
            self.num_nucl_iters,
            self.num_elec_iters,
            save_state_vector_per_atom_iteration=True,
        )
        stm = SuzukiTrotterMethodBlock(evo_spec, self.ene_spec, self.asy_spec)

        circ = stm.circuit
        logger.info("transpile START")
        transpiled = transpile(circ, backend)
        logger.info("transpile END, run START")
        results = backend.run(transpiled).result()
        logger.info("run END")
        dt = self.disc_spec.delta_t
        t = 0
        for _nucl_it in range(self.num_nucl_iters):
            t += dt * self.evo_spec.num_elec_per_atom_iterations
            self._save_result_sv(results, t)

    def _save_result_sv(self, results, t):
        label = self.evo_spec.make_state_vector_label(t)
        sv = results.data()[label]
        fname = self.outdir + "/" + self.evo_spec.make_state_vector_file_name(t)
        logger.info("Saving to : %s", fname)
        svec.save_to_file(fname, sv, eps=1e-12)

    def draw_graph(self):
        """draw the graph based on the results file."""
        fig, axs = plt.subplots(3, 1, figsize=(6, 12))
        axs[0].set_title("abs")
        axs[1].set_title("real")
        axs[2].set_title("imag")
        x = np.linspace(0, self.L, self.M + 1)

        def wrap(x):
            return np.append(x, x[:1])

        dt = self.disc_spec.delta_t
        t = 0
        for _nucl_it in range(self.num_nucl_iters):
            t += dt * self.evo_spec.num_elec_per_atom_iterations
            self._add_plot(axs, t, x, wrap)

        axs[0].legend()
        axs[1].legend()
        axs[2].legend()
        fig.savefig(self.outdir + f"/ex0_{self.n1}b.{self.num_nucl_iters}n.{self.num_elec_iters}e.dist.png")

    def _add_plot(self, axs, time, x, wrap):
        fname = self.outdir + "/" + self.evo_spec.make_state_vector_file_name(time)
        logger.info("Reading: %s", fname)
        qc = self.stm_block.circuit
        sv = svec.read_from_file(fname)
        data = svec.extract_dist(qc, sv, "e0x", eps=1e-12)
        norm = np.linalg.norm(data)
        logger.info("norm(t=%d)=%f", time, norm)
        y = wrap(data)
        ab = np.abs(y) / math.sqrt(self.dq)
        re = np.real(y) / math.sqrt(self.dq)
        im = np.imag(y) / math.sqrt(self.dq)
        axs[0].plot(x, ab, label=f"t={time}")
        axs[1].plot(x, re, label=f"t={time}")
        axs[2].plot(x, im, label=f"t={time}")


def run_experiment(par: Parameters, tag: str):

    par.draw_circuits()
    times = [0]
    if par.use_saved_data:
        logger.info("skip running the simulator")
    else:
        par.run_circuit()
    par.draw_graph()
    logger.info("done")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="time_evo_h1", description="Time evolution of H atom"
    )
    parser.add_argument("--device", type=str, default="CPU")
    parser.add_argument("--enable-cuStateVec", type=str, default="False")
    parser.add_argument("--dim", type=int, default=1)
    parser.add_argument("--precision", type=str, default="single")
    parser.add_argument("--bits", type=int, default=5)
    parser.add_argument("--num-nucl-iters", type=int, default=1)
    parser.add_argument("--num-elec-iters", type=int, default=1)
    parser.add_argument("--use-saved-data", type=str, default="False")
    args = parser.parse_args()

    use_cuStateVec = "cuStateVec" if args.enable_cuStateVec == "True" else "statevector"

    tag = f"{args.device}_{use_cuStateVec}_{args.dim}D_{args.precision}_{args.bits}b"

    outdir = "output/" + tag
    os.makedirs(outdir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        filename=outdir + "/time_evo_h1.log",
        encoding="utf-8",
        level=logging.WARNING,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger('crsq').setLevel(logging.INFO)
    logger.setLevel(logging.INFO)

    logger.info("Device : %s", args.device)
    logger.info("enable_cuStateVec : %s", args.enable_cuStateVec)
    logger.info("Dimension : %s", args.dim)
    logger.info("Precision : %s", args.precision)
    logger.info("num nucl iters : %d", args.num_nucl_iters)
    logger.info("num elec iters : %d", args.num_elec_iters)
    logger.info("use saved data : %s", args.use_saved_data)
    logger.info("Tag : %s", tag)

    par = Parameters(
        outdir,
        args.device,
        args.enable_cuStateVec == "True",
        args.dim,
        args.precision,
        args.bits,
        args.num_nucl_iters,
        args.num_elec_iters,
        args.use_saved_data == "True",
    )
    run_experiment(par, tag)
