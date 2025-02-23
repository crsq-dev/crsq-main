import math, os, argparse
import numpy as np
import scipy.special as sp
from matplotlib import pyplot as plt

from crsq.blocks.antisymmetrization import AntisymmetrizationSpec
from crsq.blocks.discretization import DiscretizationSpec
from crsq.blocks.energy_initialization import EnergyConfigurationSpec
from crsq.blocks.hamiltonian import HamiltonianSpec
from crsq.blocks.rfqhamiltonian import RfqPotentialSpec
from crsq.blocks.time_evolution.spec import (
    TimeEvolutionSpec,
    SUZUKI_TROTTER_ARITHMETIC,
    SUZUKI_TROTTER_QROM,
)
from crsq.blocks.wave_function import WaveFunctionRegisterSpec
from crsq.blocks.time_evolution.suzuki_trotter import SuzukiTrotterMethodBlock
from crsq.reports import H1D1Report
import crsq.utils.statevector as svec

from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.quantum_info import Statevector

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


def elec_proton_potential(r: float) -> float:
    if r == 0:
        raise ValueError("r == 0")
    qq = -1 * 1
    return qq / r


def elec_elec_potential(r: float) -> float:
    if r == 0:
        raise ValueError("r == 0")
    qq = -1 * -1
    return qq / r

# build the simulator

class Parameters:
    def __init__(
        self,
        outdir="output/default",
        device="GPU",
        enable_cuStateVec=True,
        precision="single",
        delta_t=0.001,
        n1=5,
        num_nucl_iters=1,
        num_elec_iters=1,
        st_method=SUZUKI_TROTTER_ARITHMETIC,
        use_saved_data=False,
    ):
        self.outdir = outdir
        self.device = device
        self.enable_cuStateVec = enable_cuStateVec
        self.precision = precision
        self.dim = 1  # 1 dimension
        self.n1 = n1  # bits per coordinate
        self.M = 1 << n1
        self.L = 16  # 16 bohr
        self.dq = self.L / self.M
        self.eta = 1  # num of electrons
        self.Ln = 0  # moving nucleus
        self.Ls = 1  # stationary nucleus
        self.num_nucl_iters = num_nucl_iters
        self.num_elec_iters = num_elec_iters
        self.st_method = st_method
        self.antisym_method = 3  # binary coded antisymmetrization method
        self.wfr_spec = WaveFunctionRegisterSpec(
            self.dim, self.n1, self.L, self.eta, self.Ln, self.Ls
        )

        # series of x coordinates.
        self.x = np.linspace(0, self.L - self.dq, self.M)
        # atom position
        # self.x0 = (self.M/2 + 0.5) * self.dq
        self.x0 = self.L / 2
        logger.info("atom pos x0=%f", self.x0)
        self.psix = hydrogen1d_psi(self.x, self.x0, N=1)
        ini_electrons = [self.psix]
        ini_configs = [ini_electrons]
        initial_electron_orbitals = ini_configs

        initial_nucleus_orbitals = [[]]
        self.ene_spec = EnergyConfigurationSpec(
            [1], initial_electron_orbitals, initial_nucleus_orbitals
        )

        self.delta_t = delta_t  # a.u.
        self.disc_spec = DiscretizationSpec(self.delta_t)
        self.asy_spec = AntisymmetrizationSpec(self.wfr_spec, self.antisym_method)
        self.nuclei_data = [{"mass": 1680, "charge": 1, "pos": int(self.x0 / self.dq)}]

        self._make_reverse_bit_index()

        if self.st_method == SUZUKI_TROTTER_QROM:
            self.rfq_spec = RfqPotentialSpec(
                self.wfr_spec,
                elec_elec_potential,
                elec_proton_potential,
                use_symmetry=False,
                use_transpose=False,
                use_gray_code=True,
            )
            self.use_motion_block_gates = True
        else:
            self.rfq_spec = None
            # self.use_motion_block_gates = False
            self.use_motion_block_gates = True

        self.ham_spec = HamiltonianSpec(self.wfr_spec, nuclei_data=self.nuclei_data)

        self.evo_spec = TimeEvolutionSpec(
            self.ham_spec,
            self.disc_spec,
            self.num_nucl_iters,
            self.num_elec_iters,
            self.st_method,
            self.rfq_spec,
            save_q_state_vector=True,
            save_p_state_vector=True
        )

        self.use_saved_data = use_saved_data
        self.stm_block = None

        self.report = H1D1Report(
            outdir,
            f"H1D1 {self.device} {self.st_method} {self.precision} {self.n1}b {self.delta_t:.3f}",
            num_coordinate_bits=n1,
            psi_axis_scale=0.6,
            space_length=self.L,
            delta_t=delta_t,
            num_elec_iters=num_elec_iters,
            num_nucl_iters=num_nucl_iters,
        )

    def _make_reverse_bit_index(self):
        self._reverse_bit_index = []
        for i in range(self.M):
            self._reverse_bit_index.append(Parameters.reverse_bits(i, self.n1))

    def reverse_bits(value, num_bits):
        result = 0
        for i in range(num_bits):
            if value & (1 << i):
                result |= 1 << (num_bits - 1 - i)
        return result

    def draw_circuits(self):

        evo_spec_for_draw = TimeEvolutionSpec(
            self.ham_spec,
            self.disc_spec,
            self.num_nucl_iters,
            self.num_elec_iters,
            self.st_method,
            self.rfq_spec,
            save_q_state_vector=False,
            use_for_loop_gate=True,
        )

        stm_block = SuzukiTrotterMethodBlock(
            evo_spec_for_draw,
            self.ene_spec,
            self.asy_spec,
            use_motion_block_gates=self.use_motion_block_gates,
        )

        logger.info("draw the circuit")
        self.report.add_circuit_diagram(stm_block.circuit, "circuit")

        # draw the circuit

        if self.st_method == SUZUKI_TROTTER_QROM:
            emb = stm_block.build_electron_motion_block(sim_time=0)
            self.report.add_circuit_diagram(emb.circuit, "elec_motion")

            epbq = emb.build_elec_potential_block_qrom()
            self.report.add_circuit_diagram(epbq.circuit, "elec_potential_qrom")
        else:
            emb = stm_block.build_electron_motion_block(sim_time=0)
            self.report.add_circuit_diagram(emb.circuit, "elec_motion")

            epot = emb.build_elec_potential_block_arithmetic()
            self.report.add_circuit_diagram(epot.circuit, "elec_potential")

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

        self.stm_block = SuzukiTrotterMethodBlock(
            self.evo_spec,
            self.ene_spec,
            self.asy_spec,
            use_motion_block_gates=self.use_motion_block_gates,
        )

        circ = self.stm_block.circuit
        logger.info("transpile START")
        transpiled = transpile(circ, backend)
        total_global_phase = transpiled.global_phase
        logger.info("accumulated global phase of the circuit: %f", total_global_phase)
        logger.info("transpile END, run START")
        result = backend.run(transpiled).result()
        if (not result.success):
            logger.error("simulation failed")
            raise ValueError("simulation failed")
            return
        logger.info("run END")
        dt = self.disc_spec.delta_t
        t = 0
        for _nucl_it in range(self.num_nucl_iters):
            # phase = total_global_phase * (_nucl_it + 1)/self.num_nucl_iters
            phase = total_global_phase
            t += dt * self.evo_spec.num_elec_per_atom_iterations
            self._save_result_sv(result, t, phase)

    def _save_result_sv(self, result, t, global_phase):
        logger.info("global phase at t=%f: %f", t, global_phase)
        q_state_label = self.evo_spec.make_state_vector_label(t)
        qsv: Statevector = result.data()[q_state_label]
        phase_adjusted_qdata = qsv.data * np.exp(-1j * global_phase)
        self.report.add_q_state_vector_file(t, qsv.dim, phase_adjusted_qdata)
        p_state_label = self.evo_spec.make_state_vector_label(t, "qft")
        psv: Statevector = result.data()[p_state_label]
        phase_adjusted_pdata = psv.data * np.exp(-1j * global_phase)
        reordered_data = self._reverse_electron_bits(phase_adjusted_pdata)
        self.report.add_p_state_vector_file(t, psv.dim, reordered_data)

    def _reverse_electron_bits(self, data):
        qc = self.stm_block.circuit
        n1 = self.wfr_spec.num_coordinate_bits
        M = 1 << n1
        result = np.zeros(M, dtype=np.complex128)
        for i in range(M):
            j = self._reverse_bit_index[i]
            result[j] = data[i]
        return result

    def draw_graph(self):
        """draw the graph based on the results file."""
        self.report.open_report()

        dt = self.disc_spec.delta_t
        t = 0
        for _nucl_it in range(self.num_nucl_iters):
            t += dt * self.evo_spec.num_elec_per_atom_iterations
            self._add_plot(t)

        self.report.generate_report()

    def _add_plot(self, time):
        qc = self.stm_block.circuit
        bit_range = svec.get_bit_range_for_reg(qc, "e0x")
        q_data = self.report.read_q_state_vector_file(time, bit_range)
        p_data = self.report.read_p_state_vector_file(time, bit_range)
        self.report.add_wave_function_plot(time, q_data, p_data)


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
    parser.add_argument("--device", type=str, choices=["CPU", "GPU"], required=True)
    parser.add_argument(
        "--enable-cuStateVec", type=str, choices=["True", "False"], required=True
    )
    parser.add_argument(
        "--precision", type=str, choices=["double", "single"], required=True
    )
    parser.add_argument("--bits", type=int, required=True)
    parser.add_argument("--num-nucl-iters", type=int, required=True)
    parser.add_argument("--num-elec-iters", type=int, required=True)
    parser.add_argument(
        "--st-method",
        type=str,
        choices=[SUZUKI_TROTTER_ARITHMETIC, SUZUKI_TROTTER_QROM],
        required=True
    )
    parser.add_argument("--use-saved-data", type=str, required=True)
    parser.add_argument("--delta-t", type=float, required=True)
    args = parser.parse_args()

    use_cuStateVec = "cuStateVec" if args.enable_cuStateVec == "True" else "statevector"

    tag = f"{args.device}_{use_cuStateVec}_{args.st_method}_1D_{args.precision}_{args.bits}b_dt{args.delta_t:.3f}/{args.num_nucl_iters}n.{args.num_elec_iters}e"

    outdir = "output/" + tag
    os.makedirs(outdir, exist_ok=True)
    logfilename = outdir + f"/time_evo_h1_{args.st_method}.log"
    if os.path.exists(logfilename):
        os.truncate(logfilename, 0)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        filename=logfilename,
        encoding="utf-8",
        level=logging.WARNING,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("crsq").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    print("Log file: ", logfilename)

    logger.info("Device : %s", args.device)
    logger.info("enable_cuStateVec : %s", args.enable_cuStateVec)
    logger.info("Precision : %s", args.precision)
    logger.info("delta_t : %f", args.delta_t)
    logger.info("num nucl iters : %d", args.num_nucl_iters)
    logger.info("num elec iters : %d", args.num_elec_iters)
    logger.info("ST method : %s", args.st_method)
    logger.info("use saved data : %s", args.use_saved_data)
    logger.info("Tag : %s", tag)

    par = Parameters(
        outdir,
        args.device,
        args.enable_cuStateVec == "True",
        args.precision,
        args.delta_t,
        args.bits,
        args.num_nucl_iters,
        args.num_elec_iters,
        args.st_method,
        args.use_saved_data == "True",
    )
    run_experiment(par, tag)
