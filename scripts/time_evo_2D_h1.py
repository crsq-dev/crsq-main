import math, os, argparse
# import numpy as np
import cupy as np
import scipy.special as sp
from matplotlib import pyplot as plt

from crsq.blocks.antisymmetrization import AntisymmetrizationSpec
from crsq.blocks.discretization import DiscretizationSpec
from crsq.blocks.energy_initialization import EnergyConfigurationSpec
from crsq.blocks.hamiltonian import HamiltonianSpec
from crsq.blocks.time_evolution.spec import TimeEvolutionSpec, SUZUKI_TROTTER_QROM
from crsq.blocks.wave_function import WaveFunctionRegisterSpec
from crsq.blocks.time_evolution.suzuki_trotter import SuzukiTrotterMethodBlock
from crsq.blocks.rfqhamiltonian import RfqPotentialSpec

from crsq.models.hydrogen2d import PsiH2D

from qiskit_aer import AerSimulator
from qiskit import transpile
import crsq.utils.statevector as svec

import logging

logger = logging.getLogger("TEV")

def elec_proton_potential(r, dq):
    return -1/(r+dq)

def elec_elec_potential(r, dq):
    return 1/(r+0.2)


# build the simulator

class Parameters:
    def __init__(
        self,
        outdir="output/default",
        device="GPU",
        enable_cuStateVec=True,
        dim=2,
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
            {"mass": 1680, "charge": 1, "pos": (0, 0)}
        ]

        self.ham_spec = HamiltonianSpec(self.wfr_spec, nuclei_data=self.nuclei_data)
        self.use_saved_data = use_saved_data
        self.stm_block = None

        self.rfq_spec = RfqPotentialSpec(self.wfr_spec, elec_elec_potential, elec_proton_potential)

    def draw_circuits(self):

        M = self.M
        self.xv = np.zeros((M,M))
        self.yv = np.zeros((M,M))
        for i in range(M):
            self.yv[:,i] = np.linspace(-0, self.L-self.dq, M)
            self.xv[i,:] = np.linspace(-0, self.L-self.dq, M)
        self.x0 = 0
        self.y0 = 0
        # quantum numbers
        qn = 0
        qm = 0
        delta_q = self.dq / 2
        psifunc2d = PsiH2D(self.x0, self.y0, delta_q, qn, qm)
        psixy = psifunc2d(self.xv, self.yv)
        ini_electrons = [psixy]
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
            method=SUZUKI_TROTTER_QROM,
            rfq_spec=self.rfq_spec,
            save_state_vector_per_atom_iteration=False  # False when we are just drawing
        )

        self.stm_block = SuzukiTrotterMethodBlock(
            self.evo_spec, self.ene_spec, self.asy_spec, use_motion_block_gates=True
        )


        fname = self.outdir + "/h2d.circuit.png"
        self.stm_block.circuit.draw(output="mpl", filename=fname, scale=0.6)
        logger.info("draw the circuit to %s", fname)


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
            method=SUZUKI_TROTTER_QROM,
            rfq_spec=self.rfq_spec,
            save_state_vector_per_atom_iteration=True  # True when we are running
        )
        stm = SuzukiTrotterMethodBlock(
            evo_spec, self.ene_spec, self.asy_spec, use_motion_block_gates=True)

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
        np_x = np.asnumpy(x)
        np_ab = np.asnumpy(np.abs(y) / math.sqrt(self.dq))
        np_re = np.asnumpy(np.real(y) / math.sqrt(self.dq))
        np_im = np.asnumpy(np.imag(y) / math.sqrt(self.dq))
        axs[0].plot(np_x, np_ab, label=f"t={time}")
        axs[1].plot(np_x, np_re, label=f"t={time}")
        axs[2].plot(np_x, np_im, label=f"t={time}")


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
        prog="time_evo_2", description="Time evolution of H atom"
    )
    parser.add_argument("--device", type=str, default="CPU")
    parser.add_argument("--enable-cuStateVec", type=str, default="False")
    parser.add_argument("--precision", type=str, default="single")
    parser.add_argument("--bits", type=int, default=5)
    parser.add_argument("--num-nucl-iters", type=int, default=1)
    parser.add_argument("--num-elec-iters", type=int, default=1)
    parser.add_argument("--use-saved-data", type=str, default="False")
    args = parser.parse_args()

    use_cuStateVec = "cuStateVec" if args.enable_cuStateVec == "True" else "statevector"

    dim = 2

    tag = f"{args.device}_{use_cuStateVec}_{dim}D_{args.precision}_{args.bits}b"

    outdir = "output/" + tag
    os.makedirs(outdir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        filename=outdir + "/time_evo_2D_h1.log",
        encoding="utf-8",
        level=logging.WARNING,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger('crsq').setLevel(logging.INFO)
    logger.setLevel(logging.INFO)

    logger.info("==== Starting ====")
    logger.info("Device : %s", args.device)
    logger.info("enable_cuStateVec : %s", args.enable_cuStateVec)
    logger.info("Precision : %s", args.precision)
    logger.info("num nucl iters : %d", args.num_nucl_iters)
    logger.info("num elec iters : %d", args.num_elec_iters)
    logger.info("use saved data : %s", args.use_saved_data)
    logger.info("Tag : %s", tag)
    logger.info("outdir : %s", outdir)

    par = Parameters(
        outdir,
        args.device,
        args.enable_cuStateVec == "True",
        dim,
        args.precision,
        args.bits,
        args.num_nucl_iters,
        args.num_elec_iters,
        args.use_saved_data == "True",
    )
    run_experiment(par, tag)
