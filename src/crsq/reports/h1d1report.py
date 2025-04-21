from qiskit import QuantumCircuit
import pandas as pd
import numpy as np
import numpy.typing as npt
import ffmpeg
import os
import glob

from typing import Tuple, Callable

import math

import matplotlib.pyplot as plt

from crsq.blocks import wave_function, time_evolution
import crsq.utils.statevector as svec

import logging

logger = logging.getLogger("crsq.reports")


class H1D1Report:
    """Report generator for 1 H atom, 1 dimension"""

    def __init__(
        self,
        outdir: str,
        title: str,
        psifunc_label: str,
        num_coordinate_bits: int,
        psi_axis_scale: float,
        space_length: float,
        window_radius: int,
        hp_func: Callable[[float], float],
        delta_t: float,
        num_elec_iters: int,
        num_nucl_iters: int,
        signed: bool,
    ):
        self._outdir = outdir
        self._frames_dir = outdir + "/frames"
        self._title = title
        self._psifunc_label = psifunc_label
        self._n1 = num_coordinate_bits
        self._M = 1 << self._n1
        self._WM = window_radius
        self._L = space_length
        self._hp_func = hp_func
        self._dq = self._L / self._M
        self._psi_axis_scale = psi_axis_scale
        self._dq = space_length / self._M
        self._delta_t = delta_t
        self._T = delta_t * num_elec_iters * num_nucl_iters
        self._num_elec_iters = num_elec_iters
        self._num_nucl_iters = num_nucl_iters
        self._signed = signed
        # x and qx values
        M = self._M
        if self._signed:
            self._xq = np.mod(np.linspace(-M // 2, M // 2 - 1, M), M) - M // 2
        else:
            self._xq = np.arange(M)
        self._x = self._xq * self._dq
        # potential energy function
        self._hpv = np.ndarray(self._M, np.float64)
        # hp_func cannot be applied to a numpy array
        for i in range(self._M):
            self._hpv[i] = self._hp_func(self._x[i])
        self._kq = np.mod(np.linspace(-M // 2, M // 2 - 1, M), M) - M // 2
        self._dp = 2 * math.pi / self._L
        self._k = self._kq * self._dp
        me = 1.0
        # kinetic energy function
        self._hkv = np.square(self._k) / (2.0 * me)
        self._prepare_dir()

    def _prepare_dir(self) -> None:
        dirname = self._frames_dir
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        for f in glob.glob(f"{dirname}/t_*.png"):
            os.remove(f)
        if os.path.exists(self.moviefile):
            os.remove(self.moviefile)

    def add_circuit_diagram(self, circuit: QuantumCircuit, block_name: str):
        """Add a circuit diagram to the report"""
        fname = f"{self._outdir}/{block_name}.png"
        logger.info(f"Saving circuit diagram of {block_name} to {fname}")
        circuit.draw(output="mpl", filename=fname, scale=0.6, fold=100)

    def open_report(self):
        """Start a plot"""
        self._fig, self._axs = plt.subplots(3, 1, figsize=(6, 12))
        self._fig.suptitle(self._title)
        self._axs[0].set_title("abs")
        self._axs[0].set_ylim(0, self._psi_axis_scale)
        self._axs[1].set_title("real")
        self._axs[1].set_ylim(-self._psi_axis_scale, self._psi_axis_scale)
        self._axs[2].set_title("imag")
        self._axs[2].set_ylim(-self._psi_axis_scale, self._psi_axis_scale)
        self._trace_time = []
        self._hk_trace = []
        self._hp_trace = []

    def add_q_state_vector_file(self, t: float, svdim: int, svdata):
        """Add a q-space statevector to the report"""
        fname = self._frames_dir + "/" + f"state_vector_{t:04.3f}_q.csv"
        logger.info("Saving to : %s", fname)
        svec.save_svdata_to_file(fname, svdim, svdata, eps=1e-12)

    def add_p_state_vector_file(self, t: float, svdim: int, svdata):
        """Add a p-space statevector to the report"""
        fname = self._frames_dir + "/" + f"state_vector_{t:04.3f}_p.csv"
        logger.info("Saving to : %s", fname)
        svec.save_svdata_to_file(fname, svdim, svdata, eps=1e-12)

    def read_q_state_vector_file(self, t: float, bit_range: Tuple[int, int]):
        fname = self._frames_dir + "/" + f"state_vector_{t:04.3f}_q.csv"
        logger.info("Reading: %s", fname)
        sv = svec.read_from_file(fname)
        data = svec.extract_dist_sub(sv, bit_range[0], bit_range[1], eps=1e-12)
        norm = np.linalg.norm(data)
        logger.info("norm(t=%f)=%f", t, norm)
        return data

    def read_p_state_vector_file(self, t: float, bit_range: Tuple[int, int]):
        fname = self._frames_dir + "/" + f"state_vector_{t:04.3f}_p.csv"
        logger.info("Reading: %s", fname)
        sv = svec.read_from_file(fname)
        data = svec.extract_dist_sub(sv, bit_range[0], bit_range[1], eps=1e-12)
        norm = np.linalg.norm(data)
        logger.info("norm(t=%d)=%f", t, norm)
        return data

    def swapv(self, v):
        if self._signed:
            return np.concatenate([v[self._M // 2 :], v[: self._M // 2]])
        else:
            return v

    def add_wave_function_plot(
        self,
        time: float,
        q_data: npt.NDArray[np.complex128],
        p_data: npt.NDArray[np.complex128],
    ):
        """Add a statevector to the report"""

        sw_qv = self.swapv(self._x)
        sw_q_data = self.swapv(q_data)

        ab = np.abs(sw_q_data) / math.sqrt(self._dq)
        re = np.real(sw_q_data) / math.sqrt(self._dq)
        im = np.imag(sw_q_data) / math.sqrt(self._dq)
        self._axs[0].plot(sw_qv, ab, label=f"t={time:4.3f}")
        self._axs[1].plot(sw_qv, re, label=f"t={time:4.3f}")
        self._axs[2].plot(sw_qv, im, label=f"t={time:4.3f}")
        """ add a frame for the video """
        self._produce_video_frame(time, sw_q_data, p_data)

    def record_energy(self, t, q_data, p_data):
        Hk = np.sum(np.abs(p_data * np.conjugate(p_data)) * self._hkv).item()
        Hp = np.sum(np.abs(q_data * np.conjugate(q_data)) * self._hpv).item()
        Htot = Hk + Hp
        logger.info("t=%f, Hk=%f, Hp=%f, Htot=%f", t, Hk, Hp, Htot)
        self._trace_time.append(t)
        self._hk_trace.append(Hk)
        self._hp_trace.append(Hp)

    def _plot_energy(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        psi_label = self._psifunc_label
        ax.set_title(f"{self._title} {psi_label}")
        npt = np.array(self._trace_time)
        nphk = np.array(self._hk_trace)
        nphp = np.array(self._hp_trace)
        nphtot = nphk + nphp
        ax.grid(True)
        ax.plot(npt, nphk, label="Hk(t)")
        ax.plot(npt, nphp, label="Hp(t)")
        ax.plot(npt, nphtot, label="Hk(t)+Hp(t)")
        ax.set_xlabel("t (time)")
        ax.set_ylabel("energy")
        ax.legend()
        filename = f"{self._outdir}/energy_trace.png"
        print("writing to file : ", filename)
        fig.savefig(fname=filename)
        plt.close(fig)

        energy_df = pd.DataFrame(
            index=npt,
            data={
                "Hk": nphk,
                "Hp": nphp,
                "Htot": nphtot,
            },
        )
        csv_filename = f"{self._outdir}/energy_trace.csv"
        energy_df.to_csv(csv_filename, index=True, index_label='t', header=True, float_format="%.6f")

    def _produce_video_frame(self, t: float, sw_q_data, p_data):
        fig, axs = plt.subplots(3, 1, figsize=(8, 12), layout="constrained")
        fig.suptitle(self._title + f" t={t:6.3f}")
        self._produce_psiq_frame(t, axs[0], sw_q_data)
        self._produce_psip_frame(t, axs[1], p_data)
        self._produce_logpsip_frame(t, axs[2], p_data)
        filename = f"{self._frames_dir}/t_{t:06.3f}.png"
        print("writing to file : ", filename)
        fig.savefig(filename)
        plt.close(fig)

    def _produce_psiq_frame(
        self, t: float, ax: plt.Axes, sw_q_data: npt.NDArray[np.complex128]
    ):
        ax.set_ylim([-1, 1])
        rdq = math.sqrt(self._dq)
        np_qv = self.swapv(self._x)
        np_psi5q = sw_q_data
        psi_label = "ψ(x)"
        ax.set_title(psi_label)
        ax.plot(np_qv, (1 / rdq) * np.abs(np_psi5q), label="|ψ(x)|")
        ax.plot(np_qv, (1 / rdq) * np.real(np_psi5q), label="Re(ψ(x))")
        ax.plot(np_qv, (1 / rdq) * np.imag(np_psi5q), label="Im(ψ(x))")
        ax.set_xlabel("x")
        ax.set_ylabel("amplitude")
        ax.legend()

    def _produce_psip_frame(
        self, t: float, ax: plt.Axes, p_data: npt.NDArray[np.complex128]
    ):
        # 離散波数が -WM ~ WM の範囲の成分を表示する
        ax.set_ylim([-1.0, 1.0])

        M = self._M
        WM = self._WM

        pv1 = self._k[0:WM]
        pv2 = self._k[M - WM : M]
        np_pv = np.concatenate([pv2, pv1])

        rdp = math.sqrt(self._dp)
        psi4p = p_data / rdp  # scale psi

        ps4p1 = psi4p[0:WM]
        ps4p2 = psi4p[M - WM : M]
        np_psi4p = np.concatenate([ps4p2, ps4p1])

        psi_label = "ψ\u0303(k)"
        ax.set_title(psi_label)
        ax.plot(np_pv, np.abs(np_psi4p), label="|ψ\u0303(k)|")
        ax.plot(np_pv, np.real(np_psi4p), label="Re(ψ\u0303(k))")
        ax.plot(np_pv, np.imag(np_psi4p), label="Im(ψ\u0303(k))")
        # the xlabel will clash with the title of the third graph.
        # ax.set_xlabel('p')
        ax.set_ylabel("amplitude")
        ax.legend()

    def _produce_logpsip_frame(
        self, t: float, ax: plt.Axes, p_data: npt.NDArray[np.complex128]
    ):
        ax.set_ylim([1.0e-4, 1.0])  # log range -30 to 0
        ax.set_yscale("log")
        ax.grid(True)
        M = self._M
        WM = self._WM

        pv1 = self._k[0:WM]
        pv2 = self._k[M - WM : M]
        np_pv = np.concatenate([pv2, pv1])

        rdp = math.sqrt(self._dp)
        psi4p = p_data / rdp  # scale psi
        abspsi = np.abs(psi4p)
        logp1 = abspsi[0:WM]
        logp2 = abspsi[M - WM : M]
        logp = np.concatenate([logp2, logp1])
        ax.plot(np_pv, logp, label="|ψ\u0303(k)|")

        # logt = np.log2(self._tarray)
        # logt1 = logt[0:WM]
        # logt2 = logt[M-WM:M]
        # np_logt = np.asnumpy(np.concatenate([logt2, logt1]))
        # ax.plot(np_pv, np_logt, label='log2(T)')

        ax.set_title(f"|ψ\u0303(k)|")

        ax.set_xlabel("k [rad/bohr]")
        ax.set_ylabel("|ψ\u0303|")
        ax.legend()

    @property
    def tagname(self):
        slabel = "signed" if self._signed else "unsigned"
        return (
            f"ex0_{self._n1}b.{slabel}.{self._num_nucl_iters}n.{self._num_elec_iters}e"
        )

    @property
    def moviefile(self):
        return f"{self._outdir}/{self.tagname}.mp4"

    def generate_report(self):
        self._axs[0].legend()
        # axs[1].legend()
        # axs[2].legend()
        self._fig.savefig(self._outdir + "/" + self.tagname + ".dist.png")
        plt.close(self._fig)

        self._plot_energy()
        self._produce_video()

    def _produce_video(self):
        print("producing video : ", self.moviefile)
        stream = ffmpeg.input(
            f"{self._frames_dir}/t_*.png", pattern_type="glob", framerate=8
        )
        ffmpeg.output(stream, self.moviefile, pix_fmt="yuv420p").run()
