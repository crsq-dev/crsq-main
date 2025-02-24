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

from crsq.models import hydrogen2d

from qiskit_aer import AerSimulator
from qiskit import transpile
import crsq.utils.sparse_statevector as ssvec
import crsq.utils.statevector as svec
from crsq.reports import H1D2Report

import logging

logger = logging.getLogger("crsq-main.scripts")


def elec_proton_potential(r: float) -> float:
    if r == 0:
        raise ValueError("r == 0")
    return -1 / r


def elec_elec_potential(r: float) -> float:
    if r == 0:
        raise ValueError("r == 0")
    return 1 / r


# build the simulator


class Parameters:
    def __init__(
        self,
        outdir,
        device,
        enable_cuStateVec,
        dim,
        length,
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
        self.plot_type = "3d-3"
        self.device = device
        self.enable_cuStateVec = enable_cuStateVec
        self.precision = precision
        self.dim = dim  # 1 dimension
        self.qn = qn
        self.qm = qm
        self.n1 = n1  # bits per coordinate
        self.M = 1 << n1
        self.L = length  # bohrs
        self.delta_t = delta_t
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
        self.use_saved_data = use_saved_data
        self.Qx0 = self.L / 2
        self.Qy0 = self.L / 2

        logger.info("==== Starting ====")
        logger.info("Device : %s", self.device)
        logger.info("enable_cuStateVec : %s", self.enable_cuStateVec)
        logger.info("Precision : %s", self.precision)
        logger.info("n1 : %d", self.n1)
        logger.info("L: %f", self.L)
        logger.info("dq: %f", self.dq)
        logger.info("delta_t: %f", self.delta_t)
        logger.info("num nucl iters : %d", self.num_nucl_iters)
        logger.info("num elec iters : %d", self.num_elec_iters)
        logger.info("qn : %d", self.qn)
        logger.info("qm : %d", self.qm)
        logger.info("use saved data : %s", use_saved_data)
        logger.info("Tag : %s", tag)
        logger.info("outdir : %s", outdir)


        self.wfr_spec = WaveFunctionRegisterSpec(
            self.dim, self.n1, self.L, self.eta, self.Ln, self.Ls
        )

        self.disc_spec = DiscretizationSpec(self.delta_t)
        self.asy_spec = AntisymmetrizationSpec(self.wfr_spec, self.antisym_method)
        self.nuclei_data = [{"mass": 1680, "charge": 1, "pos": (int(self.Qx0/self.dq), int(self.Qy0/self.dq))}]

        self.ham_spec = HamiltonianSpec(self.wfr_spec, nuclei_data=self.nuclei_data)
        self.stm_block = None

        self._make_ene_spec()

        logger.info("dq: %f", self.dq)

        # report は self.init で作成
        # self.run_circuit の中では add でデータをファイルに補間。レポート生成はしない
        # self.draw_graph の中では open_report, add_plot, generate_report でファイル生成
        self.report = H1D2Report(
            outdir,
            self.plot_type,
            title=f"H1D2 {self.device} STQR {self.precision} {self.n1}b dt{self.delta_t:.3f}",
            psifunc_label=self.psifunc2d.label,
            num_coordinate_bits=self.n1,
            zmin=0,
            zmax=1,
            vmin=-1,
            vmax=1,
            space_length=self.L,
            delta_t=self.delta_t,
            num_elec_iters=self.num_elec_iters,
            num_nucl_iters=self.num_nucl_iters
        )

    def _make_ene_spec(self):
        """make EnergyConfigurationSpec"""
        M = self.M
        # unsigned index
        ix = np.linspace(0, M - 1, M)
        iy = np.linspace(0, M - 1, M)
        # signed index
        qx = ix * self.dq
        qy = iy * self.dq
        gx, gy = np.meshgrid(qx, qy)
        self.xv = gx
        self.yv = gy

        # quantum numbers
        qn = self.qn
        qm = self.qm
        self.psifunc2d = hydrogen2d.PsiH2D(self.Qx0, self.Qy0, self.dq, qn, qm)
        psixy = self.dq * self.psifunc2d(self.xv, self.yv)
        M = self.M
        for i in range(M//2 - 2, M//2 + 3):
            for j in range(M//2 - 2, M//2 + 3):
                logger.info("psixy[%d,%d]=psixy(%f,%f)=(%f,%f)", i, j, self.xv[i,j], self.yv[i,j], psixy[i,j].real, psixy[i,j].imag)
        ini_electrons = [psixy]
        ini_configs = [ini_electrons]
        initial_electron_orbitals = ini_configs

        initial_nucleus_orbitals = [[]]
        self.ene_spec = EnergyConfigurationSpec(
            [1], initial_electron_orbitals, initial_nucleus_orbitals
        )

    def draw_circuits(self):

        # psix = np.zeros(M)
        # psiy = np.zeros(M)

        logger.info(
            "use_symmetry: %s , use_transpose: %s",
            self.use_symmetry,
            self.use_transpose,
        )
        rfq_spec = RfqPotentialSpec(
            self.wfr_spec,
            elec_elec_potential,
            elec_proton_potential,
            use_symmetry=self.use_symmetry,
            use_transpose=self.use_transpose,
            use_gray_code=self.use_gray_code,
            save_state_vector_per_qrom=False,
        )

        evo_spec = TimeEvolutionSpec(
            self.ham_spec,
            self.disc_spec,
            self.num_nucl_iters,
            self.num_elec_iters,
            method=SUZUKI_TROTTER_QROM,
            rfq_spec=rfq_spec,
            save_q_state_vector=False,  # False when we are just drawing
        )

        self.stm_block = SuzukiTrotterMethodBlock(
            evo_spec, self.ene_spec, self.asy_spec, use_motion_block_gates=True
        )

        fname = self.outdir + "/h2d.circuit.png"
        self.stm_block.circuit.draw(output="mpl", filename=fname, scale=0.6, fold=100)
        logger.info("draw the circuit to %s", fname)

    def _make_evo_spec_for_running(self):
        rfq_spec = RfqPotentialSpec(
            self.wfr_spec,
            elec_elec_potential,
            elec_proton_potential,
            use_symmetry=self.use_symmetry,
            use_transpose=self.use_transpose,
            use_gray_code=self.use_gray_code,
            save_state_vector_per_qrom=self.save_state_vector_per_qrom,
        )

        evo_spec = TimeEvolutionSpec(
            self.ham_spec,
            self.disc_spec,
            self.num_nucl_iters,
            self.num_elec_iters,
            method=SUZUKI_TROTTER_QROM,
            rfq_spec=rfq_spec,
            save_q_state_vector=self.save_state_vector_per_atom_iteration,  # True when we are running
            save_p_state_vector=False,
        )
        return evo_spec

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

        evo_spec = self._make_evo_spec_for_running()

        stm = SuzukiTrotterMethodBlock(
            evo_spec, self.ene_spec, self.asy_spec, use_motion_block_gates=True
        )

        circ = stm.circuit
        logger.info("transpile START")
        transpiled = transpile(circ, backend)
        logger.info("transpile END, run START")
        results = backend.run(transpiled).result()
        logger.info("run END")
        dt = self.delta_t
        t = 0
        for _nucl_it in range(self.num_nucl_iters):
            t += dt * evo_spec.num_elec_per_atom_iterations
            self._save_result_state_vectors(results, t, evo_spec)

    def _save_result_state_vectors(self, results, t, evo_spec: TimeEvolutionSpec):
        state_label_prefix_to_file_suffix = { "sv": "q"}
        if evo_spec.rfq_spec.should_save_state_vector_per_qrom:
            sv["qrom0"] = "qrom0"
            sv["qrom1"] = "qrom1"
        if evo_spec.should_save_p_state_vector:
            sv["qft"] = "p"
        for prefix,suffix in state_label_prefix_to_file_suffix.items():
            label = evo_spec.make_state_vector_label(t, prefix)
            if label in results.data():
                sv = results.data()[label]
                ssv = ssvec.sv_to_sparse(sv)
                data2d = self._make_2d_data_from_ssv(ssv)
                self.report.add_data_sample(suffix, t, data2d)
            else:
                logger.warning("state vector %s was not found", label)

    def _make_2d_data_from_ssv(self, ssv):
        qc = self.stm_block.circuit
        np_data2d = ssvec.extract_dist2d(qc, ssv, "e0y", "e0x")
        return np_data2d

    def draw_graph(self):
        """draw the graph based on the results file."""
        self.report.open_report()
        dt = self.delta_t
        evo_spec = self._make_evo_spec_for_running()
        t = 0
        for _nucl_it in range(self.num_nucl_iters):
            t += dt * evo_spec.num_elec_per_atom_iterations
            self._add_plots_from_file(t, evo_spec)

        self.report.generate_report()

    def _add_plots_from_file(self, time, evo_spec: TimeEvolutionSpec):
        q_data = self.report.read_data_sample("q", time)
        if evo_spec.should_save_p_state_vector:
            p_data = self.report.read_data_sample("p", time)
        else:
            p_data = None
        self.report.produce_frame(time, q_data, p_data)

        # fname = self.outdir + "/" + evo_spec.make_state_vector_file_name(time)
        # logger.info("Reading: %s", fname)
        # qc = self.stm_block.circuit
        # ssv = ssvec.read_from_file(fname)
        # logger.info("Extracting distribution")
        # np_data2d = ssvec.extract_dist2d(qc, ssv, "e0y", "e0x")
        # logger.info("make 2d data")
        # data2d = np.zeros((self.M, self.M), dtype=np.complex64)
        # for ix in range(self.M):
        #     for iy in range(self.M):
        #         data2d[(ix + self.M // 2) % self.M, (iy + self.M // 2) % self.M] = (
        #             np_data2d[ix, iy]
        #         )
        # norm = np.linalg.norm(data2d)
        # logger.info("norm(t=%4.3f)=%f", time, norm)
        # qx = np.linspace(-self.L / 2, self.L / 2 - self.dq, self.M)
        # qy = np.linspace(-self.L / 2, self.L / 2 - self.dq, self.M)
        # xg, yg = np.meshgrid(qx, qy)
        # np_xg = np.asnumpy(xg)
        # np_yg = np.asnumpy(yg)
        # np_ab = np.asnumpy(np.abs(data2d) / self.dq)
        # np_re = np.asnumpy(np.real(data2d) / self.dq)
        # np_im = np.asnumpy(np.imag(data2d) / self.dq)
        # logger.info("Plot data")
        # colormap = plt.get_cmap("cmr.guppy")
        # axs[0].set_zlim(0, 0.175)
        # axs[0].plot_surface(
        #     np_xg, np_yg, np_ab, vmin=-0.2, vmax=0.2, label=f"t={time}", cmap=colormap
        # )
        # axs[1].set_zlim(-0.175, 0.175)
        # axs[1].plot_surface(
        #     np_xg, np_yg, np_re, vmin=-0.2, vmax=0.2, label=f"t={time}", cmap=colormap
        # )
        # axs[2].set_zlim(-0.175, 0.175)
        # axs[2].plot_surface(
        #     np_xg, np_yg, np_im, vmin=-0.2, vmax=0.2, label=f"t={time}", cmap=colormap
        # )


def run_experiment(par: Parameters, tag: str):

    par.draw_circuits()
    if par.use_saved_data:
        logger.info("skip running the simulator")
    else:
        par.run_circuit()
    par.draw_graph()
    logger.info("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="time_evo_2D_h1", description="Time evolution of H atom 2D model"
    )
    parser.add_argument("--device", type=str, choices=["CPU","GPU"], required=True)
    parser.add_argument("--enable-cuStateVec", type=str, choices=["True", "False"], required=True)
    parser.add_argument("--precision", type=str, choices=["double", "single"], required=True)
    parser.add_argument("--bits", type=int, default=5)
    parser.add_argument("--length", type=float, default=16.0)
    parser.add_argument("--num-nucl-iters", type=int, default=1)
    parser.add_argument("--num-elec-iters", type=int, default=1)
    parser.add_argument("--qnum-n", type=int, required=True)
    parser.add_argument("--qnum-m", type=int, required=True)
    parser.add_argument("--use-saved-data", type=str, default="False")
    parser.add_argument("--delta-t", type=float, required=True)
    args = parser.parse_args()

    use_cuStateVec = "cuStateVec" if args.enable_cuStateVec == "True" else "statevector"

    dim = 2
    qn = args.qnum_n
    qm = args.qnum_m
    delta_t = args.delta_t

    tag = f"{args.device}_{use_cuStateVec}_STQR_2D_{args.precision}_{args.bits}b_dt{delta_t:4.3f}_n{qn}_m{qm}/{args.num_nucl_iters}n.{args.num_elec_iters}e"

    outdir = "output/" + tag
    os.makedirs(outdir, exist_ok=True)
    logfilename = outdir + "/time_evo_2D_h1.log"
    if os.path.exists(logfilename):
        os.truncate(logfilename, 0)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        filename=logfilename,
        encoding="utf-8",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("crsq").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    print("Log file: ", logfilename)


    par = Parameters(
        outdir,
        args.device,
        args.enable_cuStateVec == "True",
        dim,
        args.length,
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
