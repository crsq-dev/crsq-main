import math, os, argparse
# import numpy as np
import cupy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import cmasher as cmr
import scipy.special as sp

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

def elec_proton_potential(r: float) -> float:
    if r == 0:
        raise ValueError("r == 0")
    return -1/r

def elec_elec_potential(r: float) -> float:
    if r == 0:
        raise ValueError("r == 0")
    return 1/r


# build the simulator

class Parameters:
    def __init__(
        self,
        outdir,
        device,
        enable_cuStateVec,
        dim,
        qn,
        qm,
        delta_t,
        precision="single",
        n1=5,
        num_nucl_iters=1,
        num_elec_iters=2,
        use_saved_data=False,
    ):
        self.outdir = outdir
        self.device = device
        self.enable_cuStateVec = enable_cuStateVec
        self.precision = precision
        self.dim = dim  # 1 dimension
        self.qn = qn
        self.qm = qm
        self.n1 = n1  # bits per coordinate
        self.M = 1 << n1
        self.L = 16  # bohrs
        self.dq = self.L / self.M
        self.eta = 1  # num of electrons
        self.Ln = 0  # moving nucleus
        self.Ls = 1  # stationary nucleus
        self.num_nucl_iters = num_nucl_iters
        self.num_elec_iters = num_elec_iters
        # QROM optimization switches
        self.use_symmetry = False
        self.use_transpose = False
        self.use_gray_code = True
        self.save_state_vector_per_qrom = False
        self.save_state_vector_per_atom_iteration = True
        self.antisym_method = 3  # binary coded antisymmetrization method
        self.wfr_spec = WaveFunctionRegisterSpec(
            self.dim, self.n1, self.L, self.eta, self.Ln, self.Ls
        )

        self.delta_t = delta_t
        self.disc_spec = DiscretizationSpec(self.delta_t)
        self.asy_spec = AntisymmetrizationSpec(self.wfr_spec, self.antisym_method)
        self.nuclei_data = [
            {"mass": 1680, "charge": 1, "pos": (0, 0)}
        ]

        self.ham_spec = HamiltonianSpec(self.wfr_spec, nuclei_data=self.nuclei_data)
        self.use_saved_data = use_saved_data
        self.stm_block = None

        logger.info("dq: %f", self.dq)

    def draw_circuits(self):

        M = self.M
        # unsigned index
        ix = np.linspace(0, M-1, M)
        iy = np.linspace(0, M-1, M)
        # signed index
        six = (ix + M//2) % M - M//2
        siy = (iy + M//2) % M - M//2
        qx = six * self.dq
        qy = siy * self.dq
        gx, gy = np.meshgrid(qx, qy)
        self.xv = gx
        self.yv = gy

        self.x0 = 0
        self.y0 = 0
        # quantum numbers
        qn = self.qn
        qm = self.qm
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

        logger.info("use_symmetry: %s , use_transpose: %s", self.use_symmetry, self.use_transpose)
        self.rfq_spec = RfqPotentialSpec(
            self.wfr_spec,
            elec_elec_potential,
            elec_proton_potential,
            use_symmetry=self.use_symmetry,
            use_transpose=self.use_transpose,
            use_gray_code=self.use_gray_code,
            save_state_vector_per_qrom=False)


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
        num_threads = 0
        backend.set_options(max_parallel_threads=num_threads)

        self.rfq_spec = RfqPotentialSpec(
            self.wfr_spec,
            elec_elec_potential,
            elec_proton_potential,
            use_symmetry=self.use_symmetry,
            use_transpose=self.use_transpose,
            use_gray_code=self.use_gray_code,
            save_state_vector_per_qrom=self.save_state_vector_per_qrom)

        self.evo_spec = TimeEvolutionSpec(
            self.ham_spec,
            self.disc_spec,
            self.num_nucl_iters,
            self.num_elec_iters,
            method=SUZUKI_TROTTER_QROM,
            rfq_spec=self.rfq_spec,
            save_state_vector_per_atom_iteration=self.save_state_vector_per_atom_iteration,  # True when we are running
            save_state_vector_per_qft=False
        )
        stm = SuzukiTrotterMethodBlock(
            self.evo_spec, self.ene_spec, self.asy_spec, use_motion_block_gates=True)

        circ = stm.circuit
        logger.info("transpile START")
        transpiled = transpile(circ, backend)
        logger.info("transpile END, run START")
        results = backend.run(transpiled).result()
        logger.info("run END")
        dt = self.disc_spec.delta_t
        t = 0
        for _nucl_it in range(self.num_nucl_iters * self.num_elec_iters):
            t += dt
            self._save_result_sv(results, t)

    def _save_result_sv(self, results, t):
        suffixes = []
        if self.evo_spec.rfq_spec.should_save_state_vector_per_qrom:
            suffixes += ["_qrom0", "_qrom1"]
        if self.evo_spec.should_save_state_vector_per_qft:
            suffixes += ["_qftd1"]
        suffixes += [""]
        for suffix in suffixes:
            label = self.evo_spec.make_state_vector_label(t, suffix)
            if label in results.data():
                sv = results.data()[label]
                fname = self.outdir + "/" + self.evo_spec.make_state_vector_file_name(t, suffix)
                logger.info("State vector label: %s", label)
                if os.path.exists(fname):
                    logger.info("removing old file : %s", fname)
                    os.remove(fname)
                logger.info("Saving to : %s", fname)
                svec.save_to_file(fname, sv, eps=1e-12)


    def draw_graph(self):
        """draw the graph based on the results file."""

        dt = self.disc_spec.delta_t
        t = 0
        for _nucl_it in range(self.num_nucl_iters):
            fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={"projection": "3d"})
            axs[0].set_title("abs")
            axs[1].set_title("real")
            axs[2].set_title("imag")
            t += dt * self.evo_spec.num_elec_per_atom_iterations
            self._add_plot(axs, t, self.xv, self.yv)
            axs[0].legend()
            axs[1].legend()
            axs[2].legend()
            fig.savefig(self.outdir + f"/e0_{self.n1}b.{self.num_nucl_iters}n.{self.num_elec_iters}e.t{t:04.3f}.png")
            plt.close(fig)

    def _add_plot(self, axs: list[Axes], time, xg, yg):
        fname = self.outdir + "/" + self.evo_spec.make_state_vector_file_name(time)
        logger.info("Reading: %s", fname)
        qc = self.stm_block.circuit
        sv = svec.read_from_file(fname)
        np_data2d = svec.extract_dist2d(qc, sv, "e0y", "e0x")
        data2d = np.zeros((self.M, self.M), dtype=np.complex64)
        for ix in range(self.M):
            for iy in range(self.M):
                data2d[(ix+self.M//2) % self.M, (iy+self.M//2) % self.M] = np_data2d[ix, iy]
        norm = np.linalg.norm(data2d)
        logger.info("norm(t=%4.3f)=%f", time, norm)
        qx = np.linspace(-self.L/2, self.L/2-self.dq, self.M)
        qy = np.linspace(-self.L/2, self.L/2-self.dq, self.M)
        xg, yg = np.meshgrid(qx, qy)
        np_xg = np.asnumpy(xg)
        np_yg = np.asnumpy(yg)
        np_ab = np.asnumpy(np.abs(data2d) / self.dq)
        np_re = np.asnumpy(np.real(data2d) / self.dq)
        np_im = np.asnumpy(np.imag(data2d) / self.dq)
        colormap = plt.get_cmap("cmr.guppy")
        axs[0].plot_surface(np_xg, np_yg, np_ab, label=f"t={time}", cmap=colormap)
        axs[1].plot_surface(np_xg, np_yg, np_re, label=f"t={time}", cmap=colormap)
        axs[2].plot_surface(np_xg, np_yg, np_im, label=f"t={time}", cmap=colormap)


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
    parser.add_argument("--qnum-n", type=int, default=1)
    parser.add_argument("--qnum-m", type=int, default=0)
    parser.add_argument("--use-saved-data", type=str, default="False")
    args = parser.parse_args()

    use_cuStateVec = "cuStateVec" if args.enable_cuStateVec == "True" else "statevector"

    dim = 2
    qn = args.qnum_n
    qm = args.qnum_m
    delta_t = 0.01

    tag = f"{args.device}_{use_cuStateVec}_{args.bits}b_{args.precision}_{dim}D_n{qn}_m{qm}_dt{delta_t:4.3f}"

    outdir = "output/" + tag
    os.makedirs(outdir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        filename=outdir + "/time_evo_2D_h1.log",
        encoding="utf-8",
        level=logging.INFO,
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
    logger.info("qn : %d", args.qnum_n)
    logger.info("qm : %d", args.qnum_m)
    logger.info("use saved data : %s", args.use_saved_data)
    logger.info("Tag : %s", tag)
    logger.info("outdir : %s", outdir)

    par = Parameters(
        outdir,
        args.device,
        args.enable_cuStateVec == "True",
        dim,
        args.qnum_n,
        args.qnum_m,
        delta_t,
        args.precision,
        args.bits,
        args.num_nucl_iters,
        args.num_elec_iters,
        args.use_saved_data == "True",
    )
    try:
        run_experiment(par, tag)
    except ValueError as e:
        logger.error("ValueError: %s", e)
        print("ValueError: ", e)

