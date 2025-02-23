
from qiskit import QuantumCircuit
import numpy as np
import numpy.typing as npt
import ffmpeg
import os;
import glob

from typing import Tuple

import math

import matplotlib.pyplot as plt

from crsq.blocks import wave_function, time_evolution
import crsq.utils.statevector as svec

import logging

logger = logging.getLogger("crsq.reports")

class H1D1Report:
    """Report generator for 1 H atom, 1 dimension
    """

    def __init__(
        self,
        outdir: str,
        title: str,
        num_coordinate_bits: int,
        psi_axis_scale: float,
        space_length: float,
        delta_t: float,
        num_elec_iters: int,
        num_nucl_iters: int
    ):
        self._outdir = outdir
        self._framesdir = outdir + "/frames"
        self._title = title
        self._n1 = num_coordinate_bits
        self._M = 1 << self._n1
        self._WM = self._M // 2
        self._L = space_length
        self._dq = self._L / self._M
        self._psi_axis_scale = psi_axis_scale
        self._dq = space_length / self._M
        self._delta_t = delta_t
        self._T = delta_t*num_elec_iters*num_nucl_iters
        self._num_elec_iters = num_elec_iters
        self._num_nucl_iters = num_nucl_iters
        self._xv = np.linspace(0, self._M-1, self._M)
        self._qv = self._xv * self._dq
        self._kv = np.concatenate([np.linspace(0, self._M/2-1, self._M//2), np.linspace(-self._M/2, -1, self._M//2)])
        self._dp = 2*math.pi/self._L
        self._pv = self._kv * self._dp
        self._prepare_dir()

    def _prepare_dir(self) -> None:
        dirname = self._framesdir
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        for f in glob.glob(f'{dirname}/t_*.png'):
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
    
    def add_q_state_vector_file(self, t: float, svdim: int, svdata):
        """Add a q-space statevector to the report"""
        fname = self._framesdir + "/" + f"state_vector_{t:04.3f}_q.csv"
        logger.info("Saving to : %s", fname)
        svec.save_svdata_to_file(fname, svdim, svdata, eps=1e-12)
    
    def add_p_state_vector_file(self, t: float, svdim: int, svdata):
        """Add a p-space statevector to the report"""
        fname = self._framesdir + "/" + f"state_vector_{t:04.3f}_p.csv"
        logger.info("Saving to : %s", fname)
        svec.save_svdata_to_file(fname, svdim, svdata, eps=1e-12)
    
    def read_q_state_vector_file(self, t: float, bit_range: Tuple[int, int]):
        fname = self._framesdir + "/" + f"state_vector_{t:04.3f}_q.csv"
        logger.info("Reading: %s", fname)
        sv = svec.read_from_file(fname)
        data = svec.extract_dist_sub(sv, bit_range[0], bit_range[1], eps=1e-12)
        norm = np.linalg.norm(data)
        logger.info("norm(t=%d)=%f", t, norm)
        return data
    
    def read_p_state_vector_file(self, t: float, bit_range: Tuple[int, int]):
        fname = self._framesdir + "/" + f"state_vector_{t:04.3f}_p.csv"
        logger.info("Reading: %s", fname)
        sv = svec.read_from_file(fname)
        data = svec.extract_dist_sub(sv, bit_range[0], bit_range[1], eps=1e-12)
        norm = np.linalg.norm(data)
        logger.info("norm(t=%d)=%f", t, norm)
        return data

    def add_wave_function_plot(self, time: float, q_data: npt.NDArray[np.complex128], p_data: npt.NDArray[np.complex128]):
        """Add a statevector to the report"""
        ab = np.abs(q_data) / math.sqrt(self._dq)
        re = np.real(q_data) / math.sqrt(self._dq)
        im = np.imag(q_data) / math.sqrt(self._dq)
        self._axs[0].plot(self._qv, ab, label=f"t={time:4.3f}")
        self._axs[1].plot(self._qv, re, label=f"t={time:4.3f}")
        self._axs[2].plot(self._qv, im, label=f"t={time:4.3f}")
        """ add a frame for the video """
        self._produce_video_frame(time, self._T, q_data, p_data)

    def _produce_video_frame(self, t: float, T, q_data, p_data):
        fig, axs = plt.subplots(3, 1, figsize=(8, 12), layout='constrained')
        fig.suptitle(self._title + f" t={t:6.3f}")
        self._produce_psiq_frame(t, T, axs[0], q_data)
        self._produce_psip_frame(t, T, axs[1], p_data)
        self._produce_logpsip_frame(t, T, axs[2], p_data)
        filename = f'{self._framesdir}/t_{t:06.3f}.png'
        print("writing to file : ", filename)
        fig.savefig(filename)
        plt.close(fig)

    def _produce_psiq_frame(self, t: float, T, ax: plt.Axes, q_data: npt.NDArray[np.complex128]):
        ax.set_ylim([-1, 1])
        rdq = math.sqrt(self._dq)
        np_qv = self._qv
        np_psi5q = q_data
        psi_label = "ψ_q"
        ax.set_title(psi_label)
        ax.plot(np_qv, (1/rdq)*np.abs(np_psi5q), label='|ψ(q)|')
        ax.plot(np_qv, (1/rdq)*np.real(np_psi5q), label='Re(ψ(q))')
        ax.plot(np_qv, (1/rdq)*np.imag(np_psi5q), label='Im(ψ(q))')
        ax.set_xlabel('q')
        ax.set_ylabel('amplitude')
        ax.legend()

    def _produce_psip_frame(self, t: float, T, ax: plt.Axes, p_data: npt.NDArray[np.complex128]):
        # 離散波数が -WM ~ WM の範囲の成分を表示する
        ax.set_ylim([-1.0, 1.0])

        M = self._M
        WM = self._WM

        pv1 = self._pv[0:WM]
        pv2 = self._pv[M-WM:M]
        np_pv = np.concatenate([pv2, pv1])

        ps4p1 = p_data[0:WM]
        ps4p2 = p_data[M-WM:M]
        np_psi4p = np.concatenate([ps4p2, ps4p1])

        rdp = math.sqrt(self._dp)
        psi_label = "ψ_p"
        ax.set_title(psi_label)
        ax.plot(np_pv, (1/rdp)*np.abs(np_psi4p), label='|ψ\u0303(p)|')
        ax.plot(np_pv, (1/rdp)*np.real(np_psi4p), label='Re(ψ\u0303(p))')
        ax.plot(np_pv, (1/rdp)*np.imag(np_psi4p), label='Im(ψ\u0303(p))')
        # the xlabel will clash with the title of the third graph.
        # ax.set_xlabel('p')
        ax.set_ylabel('amplitude')
        ax.legend()

    def _produce_logpsip_frame(self, t: float, T, ax: plt.Axes, p_data: npt.NDArray[np.complex128]):
        ax.set_ylim([-30, 0])  # log range -30 to 0
        ax.grid(True)
        M = self._M
        WM = self._WM

        pv1 = self._pv[0:WM]
        pv2 = self._pv[M-WM:M]
        np_pv = np.concatenate([pv2, pv1])

        logpsi = np.log2(np.abs(p_data))
        logp1 = logpsi[0:WM]
        logp2 = logpsi[M-WM:M]
        np_logp = np.concatenate([logp2, logp1])
        ax.plot(np_pv, np_logp, label='log2|ψ\u0303(p)|\u221adp')

        # logt = np.log2(self._tarray)
        # logt1 = logt[0:WM]
        # logt2 = logt[M-WM:M]
        # np_logt = np.asnumpy(np.concatenate([logt2, logt1]))
        # ax.plot(np_pv, np_logt, label='log2(T)')

        ax.set_title(f'log2|ψ\u0303(p)|\u221adp')

        ax.set_xlabel('p [rad/bohr]')
        ax.set_ylabel('log2(|ψ|)')
        ax.legend()

    @property
    def tagname(self):
        return f"ex0_{self._n1}b.{self._num_nucl_iters}n.{self._num_elec_iters}e"
    
    @property
    def moviefile(self):
        return f"{self._outdir}/{self.tagname}.mp4"

    def generate_report(self):
        self._axs[0].legend()
        # axs[1].legend()
        # axs[2].legend()
        self._fig.savefig(
            self._outdir + "/" + self.tagname + ".dist.png"
        )
        plt.close(self._fig)

        print("producing video : ", self.moviefile)
        stream = ffmpeg.input(f'{self._framesdir}/t_*.png', pattern_type='glob', framerate=8)
        ffmpeg.output(stream, self.moviefile, pix_fmt='yuv420p').run()
