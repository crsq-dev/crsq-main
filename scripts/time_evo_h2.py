import math, os, argparse
import numpy as np
import scipy.special as sp
from matplotlib import pyplot as plt
import cmasher as cmr
import ffmpeg

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
        run_simulation=False,
        draw_result = False
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
        self.eta = 2  # num of electrons
        self.Ln = 0  # moving nucleus
        self.Ls = 2  # stationary nucleus
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
            {"mass": 1680, "charge": 1, "pos": self.M // 4},
            {"mass": 1680, "charge": 1, "pos": self.M * 3 // 4},
        ]

        self.ham_spec = HamiltonianSpec(self.wfr_spec, nuclei_data=self.nuclei_data)
        self.run_simulation = run_simulation
        self.draw_result = draw_result
        self.stm_block = None

        self._zmin = -0.2
        self._zmax = 0.2
        self._colormap_name = 'cmr.guppy'

    def draw_circuits(self):

        self.x = np.linspace(0, self.L - self.dq, self.M)
        self.x0 = 4
        self.x1 = 12
        self.psi0x = hydrogen1d_psi(self.x, self.x0, N=1)
        self.psi1x = hydrogen1d_psi(self.x, self.x1, N=1)
        # ini_dims = [psix, psiy]
        ini_dims0 = [self.psi0x]
        ini_dims1 = [self.psi1x]
        ini_electrons = [ini_dims0, ini_dims1]
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

        fname = self.outdir + "/h2d.circuit.png"
        self.stm_block.circuit.draw(output="mpl", filename=fname, scale=0.6)

        # draw the circuit

        epot = self.stm_block.build_elec_potential_block()
        fname = self.outdir + "/h2d.circuit.elec_potential.png"
        epot.circuit.draw(output="mpl", filename=fname, scale=0.6, fold=60)

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

        def wrap(x):
            return np.append(x, x[:1])

        # make png files for each time step
        dt = self.disc_spec.delta_t
        t = 0
        for _nucl_it in range(self.num_nucl_iters):
            t += dt * self.evo_spec.num_elec_per_atom_iterations
            # self._create_frame(t)
            self.produce_frame3d3(t)

        # integrate the time steps into one video file
        moviefile = f'{self.outdir}/animation.mp4'
        print("producing video : ", moviefile)
        stream = ffmpeg.input(f'{self.outdir}/frame_*.png', pattern_type='glob', framerate=8)
        ffmpeg.output(stream, moviefile, pix_fmt='yuv420p').run()

    def _create_frame(self, time):
        fname = self.outdir + "/" + self.evo_spec.make_state_vector_file_name(time)
        logger.info("Reading state vector data from: %s", fname)
        qc = self.stm_block.circuit
        data2d = svec.extract_dist2d_from_file(qc, fname, "e0x", "e1x")
        logger.info("Reading done. producing png")
        ab = np.abs(data2d) / self.dq
        re = np.real(data2d) / self.dq
        im = np.imag(data2d) / self.dq
        fig, axs = plt.subplots(3, 1, figsize=(6, 12))
        axs[0].set_title("abs")
        axs[1].set_title("real")
        axs[2].set_title("imag")
        axs[0].imshow(ab)
        axs[1].imshow(re)
        axs[2].imshow(im)
        fig.savefig(self.outdir + f"/frame_{time:06.3f}.png")
        logger.info("png frame saved.")

    def produce_frame3d3(self, t: float):
        fname = self.outdir + "/" + self.evo_spec.make_state_vector_file_name(t)
        logger.info("Reading state vector data from: %s", fname)
        qc = self.stm_block.circuit
        data2d = svec.extract_dist2d_from_file(qc, fname, "e0x", "e1x")
        logger.info("Reading done. producing png")

#        par = self._par
        fig, axs = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(15, 6), layout='constrained')
        colormap = plt.get_cmap(self._colormap_name)
        dq = self.dq

        M = self.M
        xv = np.zeros((M, M))
        yv = np.zeros((M, M))
        for i in range(M):
            yv[:,i] = np.linspace(0, M-1, M)
            xv[i,:] = np.linspace(0, M-1, M)
        qxv = xv * dq
        qyv = yv * dq

        ax = axs[0]
        ax.set_title('|ψ|')
        ax.set_zlim3d(self._zmin, self._zmax)
        ax.set_xlabel('y')
        ax.set_ylabel('x')

        # np_qyv = np.asnumpy(qyv)
        # np_qxv = np.asnumpy(qxv)
        # np_psi5q = np.asnumpy(data2d)
        np_qyv = qyv
        np_qxv = qxv
        np_psi5q = data2d
        ax.plot_surface(np_qyv, np_qxv, (1/dq)*np.abs(np_psi5q), cmap=colormap, vmin=self._zmin, vmax=self._zmax)

        ax = axs[1]
        ax.set_title('Re(ψ)')
        ax.set_zlim3d(self._zmin, self._zmax)
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.plot_surface(np_qyv, np_qxv, (1/dq)*np.real(np_psi5q), cmap=colormap, vmin=self._zmin, vmax=self._zmax)

        ax = axs[2]
        ax.set_title('Im(ψ)')
        ax.set_zlim3d(self._zmin, self._zmax)
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.plot_surface(np_qyv, np_qxv, (1/dq)*np.imag(np_psi5q), cmap=colormap, vmin=self._zmin, vmax=self._zmax)

        psi_label = 'psi'
        dt = self.disc_spec.delta_t
        n1 = self.n1
        fig.suptitle(f't={t:6.3f},dt={dt},n1={n1},' + psi_label)
        filename = f'{self.outdir}/frame_{t:06.3f}.png'
        print("writing to file : ", filename)
        fig.savefig(filename)
        plt.close(fig)


def run_experiment(par: Parameters, tag: str):

    par.draw_circuits()
    times = [0]
    if par.run_simulation:
        par.run_circuit()
    else:
        logger.info("skip running the simulator")
    if par.draw_result:
        par.draw_graph()
    logger.info("done")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="time_evo_h2", description="Time evolution of H2 atoms"
    )
    parser.add_argument("--device", type=str, default="CPU")
    parser.add_argument("--enable-cuStateVec", type=str, default="False")
    parser.add_argument("--dim", type=int, default=1)
    parser.add_argument("--precision", type=str, default="single")
    parser.add_argument("--bits", type=int, default=5)
    parser.add_argument("--num-nucl-iters", type=int, default=1)
    parser.add_argument("--num-elec-iters", type=int, default=1)
    parser.add_argument("--run-simulation", type=str, default="True")
    parser.add_argument("--draw-result", type=str, default="True")
    args = parser.parse_args()

    use_cuStateVec = "cuStateVec" if args.enable_cuStateVec == "True" else "statevector"

    tag = f"{args.device}_{use_cuStateVec}_2H_{args.dim}D_{args.precision}_{args.bits}b"

    outdir = "output/" + tag
    os.makedirs(outdir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        filename=outdir + "/time_evo_h2.log",
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
    logger.info("run simulation : %s", args.run_simulation)
    logger.info("draw result : %s", args.draw_result)
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
        args.run_simulation == "True",
        args.draw_result == "True"
    )
    run_experiment(par, tag)
