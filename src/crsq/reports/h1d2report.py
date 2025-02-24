from qiskit import QuantumCircuit
import numpy as np
import numpy.typing as npt
import ffmpeg
import os
import glob

from typing import Tuple, Callable

import math

import matplotlib.pyplot as plt
from matplotlib import colormaps
import cmasher as cmr

from crsq.blocks import wave_function, time_evolution
import crsq.utils.statevector as svec

import logging

logger = logging.getLogger("crsq.reports")


class H1D2Report:
    """Report generator for 1 H atom, 1 dimension"""

    def __init__(
        self,
        outdir: str,
        plot_type: str,
        title: str,
        psifunc_label: str,
        num_coordinate_bits: int,
        zmin: float,
        zmax: float,
        vmin: float,
        vmax: float,
        space_length: float,
        delta_t: float,
        num_elec_iters: int,
        num_nucl_iters: int,
    ):
        self._outdir = outdir
        self._plot_type = plot_type
        self._colormap_name = "cmr.guppy"
        self._frames_dir = outdir + "/frames"
        self._title = title
        self._psifunc_label = psifunc_label
        self._n1 = num_coordinate_bits
        M = 1 << self._n1
        self._M = M
        self._WM = self._M // 2
        self._L = space_length
        self._dq = self._L / self._M
        self._dq = space_length / self._M
        self._zmin = zmin
        self._zmax = zmax
        self._vmin = vmin
        self._vmax = vmax
        self._delta_t = delta_t
        self._total_time = delta_t*num_elec_iters*num_nucl_iters
        self._T = delta_t * num_elec_iters * num_nucl_iters
        self._num_elec_iters = num_elec_iters
        self._num_nucl_iters = num_nucl_iters

        self._xv = np.zeros((M, M))
        self._yv = np.zeros((M, M))
        for i in range(M):
            self._yv[:, i] = np.linspace(0, M - 1, M)
            self._xv[i, :] = np.linspace(0, M - 1, M)
        # discretized wave number values
        kv = np.mod(np.linspace(-M // 2, M // 2 - 1, M), M) - (M // 2)
        kv2 = np.square(kv)
        self._kxv2 = np.zeros((M, M))
        self._kyv2 = np.zeros((M, M))
        for i in range(M):
            self._kyv2[:, i] = kv2
            self._kxv2[i, :] = kv2
        self._kv2 = self._kxv2 + self._kyv2

        # value of V
        dq = self._dq
        self._qxv = self._xv * dq
        self._qyv = self._yv * dq

        self._prepare_dir()

    def set_plot_type(self, plot_type: str):
        self._plot_type = plot_type

    def set_color_map(self, colormap_name: str):
        self._colormap_name = colormap_name

    def _prepare_dir(self) -> None:
        """"""
        dirname = self._frames_dir
        if not os.path.exists(dirname):
            logger.info("Creating directory : %s", dirname)
            print("Creating directory : ", dirname)
            os.makedirs(dirname)
        for f in glob.glob(f"{dirname}/t_*.png"):
            os.remove(f)
        if os.path.exists(self.moviefile):
            os.remove(self.moviefile)

    def open_report(self) -> None:
        """"""

    def add_data_sample(self, label: str, t: float, q_data: npt.NDArray[np.complex128]) -> None:
        """ save 2d grid data to a text file"""
        file_name = self._frames_dir + f"/{t:06.3f}.{label}.csv"
        self.write_2d_data(file_name, q_data)

    def write_2d_data(self, file_name: str, data: npt.NDArray[np.complex128]) -> None:
        """ save 2d grid data to a text file"""
        if os.path.exists(file_name):
            logger.info("removing old file : %s", file_name)
            os.remove(file_name)
        logger.info("Saving to : %s", file_name)
        shp = data.shape
        sumdata = np.sum(np.abs(data) ** 2)
        maxdata = np.max(np.abs(data))
        logger.info("sum of |ψ|^2 : %f", sumdata)
        logger.info("max of |ψ| : %f", maxdata)
        with open(file_name, "w") as f:
            f.write(f"{shp[0]},{shp[1]}\n")
            for i in range(shp[0]):
                for j in range(shp[1]):
                    f.write(f"{i},{j},{data[i,j].real},{data[i,j].imag}\n")

    def read_2d_data(self, file_name: str) -> npt.NDArray[np.complex128]:
        """ read 2d grid data from a text file"""
        with open(file_name, "r") as f:
            s = f.readline().split(",")
            n1 = int(s[0])
            n2 = int(s[1])
            data = np.zeros((n1, n2), dtype=np.complex128)
            for i in range(n1):
                for j in range(n2):
                    s = f.readline().split(",")
                    data[i, j] = complex(float(s[2]), float(s[3]))
        return data

    def read_data_sample(self, label: str, t: float) -> npt.NDArray[np.complex128]:
        """ read q-space 2d grid data from a text file"""
        file_name = self._frames_dir + f"/{t:06.3f}.{label}.csv"
        return self.read_2d_data(file_name)
    
    def produce_frame(
        self,
        t: float,
        q_data: npt.NDArray[np.complex128],
        p_data: npt.NDArray[np.complex128],
    ) -> None:
        """ """
        if self._plot_type == "2d":
            self.produce_frame2d(t, q_data, p_data)
        elif self._plot_type == "3d":
            self.produce_frame3d(t, q_data)
        elif self._plot_type == "3d-re":
            self.produce_frame3d_re(t, q_data)
        elif self._plot_type == "3d-3":
            self.produce_frame3d3(t, q_data)
        else:
            raise ValueError(f"Unknown plot type : {self._plot_type}")

    def produce_frame2d(
        self,
        t: float,
        q_data: npt.NDArray[np.complex128],
        p_data: npt.NDArray[np.complex128],
    ) -> None:
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        colormap = plt.get_cmap(self._colormap_name)
        ax: plt.Axes = axs[0]
        # ax.imshow(np.abs(self._psi1q))
        ax.imshow(
            np.real(q_data),
            cmap=colormap,
            vmin=self._vmin,
            vmax=self._vmax,
        )
        psi_label = self._psifunc_label
        fig.suptitle(f"t={t:6.3f},dt={self._delta_t},n1={self._n1}," + psi_label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        px: plt.Axes = axs[1]
        px.imshow(
            np.abs(p_data),
            cmap=colormap,
            vmin=self._vmin,
            vmax=self._vmax,
        )
        px.set_xlabel("kx")
        px.set_ylabel("ky")
        # ax.legend()
        filename = f"{self._frames_dir}/t_{t:06.3f}.png"
        print("writing to file : ", filename)
        fig.savefig(filename)
        plt.close(fig)

    def produce_frame3d(self, t, q_data: npt.NDArray[np.complex128]) -> None:
        fig, ax = plt.subplots(
            subplot_kw={"projection": "3d"}, figsize=(6, 5.5), layout="constrained"
        )
        colormap = plt.get_cmap(self._colormap_name)
        dq = self._dq

        ax.set_title("|ψ|")
        ax.set_zlim3d(self._zmin, self._zmax)
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        np_qxv = self._qxv
        np_qyv = self._qyv
        ax.plot_surface(
            np_qyv,
            np_qxv,
            (1 / dq) * np.abs(q_data),
            cmap=colormap,
            vmin=self._vmin,
            vmax=self._vmax,
        )

        psi_label = self._psifunc_label
        fig.suptitle(f"t={t:6.3f},dt={self._delta_t},n1={self._n1}," + psi_label)
        filename = f"{self._frames_dir}/t_{t:06.3f}.png"
        print("writing to file : ", filename)
        fig.savefig(filename)
        plt.close(fig)

    def produce_frame3d_re(self, t: float, q_data: npt.NDArray[np.complex128]) -> None:
        fig, ax = plt.subplots(
            subplot_kw={"projection": "3d"}, figsize=(6, 5.5), layout="constrained"
        )
        colormap = plt.get_cmap(self._colormap_name)
        dq = self._dq

        psi_label = self._psifunc_label
        ax.set_title(f"Re(ψ) [t={t:6.3f},{psi_label}]")
        ax.set_zlim3d(self._zmin, self._zmax)
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        np_qxv = self._qxv
        np_qyv = self._qyv
        ax.plot_surface(
            np_qyv,
            np_qxv,
            (1 / dq) * np.real(q_data),
            cmap=colormap,
            vmin=self._vmin,
            vmax=self._vmax,
        )

        filename = f"{self._frames_dir}/t_{t:06.3f}.png"
        print("writing to file : ", filename)
        fig.savefig(filename)
        plt.close(fig)

    def produce_frame3d3(self, t: float, q_data: npt.NDArray[np.complex128]) -> None:
        fig, axs = plt.subplots(
            1, 3, subplot_kw={"projection": "3d"}, figsize=(15, 6), layout="constrained"
        )
        colormap = plt.get_cmap(self._colormap_name)
        dq = self._dq

        ax: plt.Axes = axs[0]
        ax.set_title("|ψ|")
        ax.set_zlim3d(self._zmin, self._zmax)
        ax.set_xlabel("y")
        ax.set_ylabel("x")

        np_qyv = self._qyv
        np_qxv = self._qxv
        ax.plot_surface(
            np_qyv,
            np_qxv,
            (1 / dq) * np.abs(q_data),
            cmap=colormap,
            vmin=self._vmin,
            vmax=self._vmax,
        )

        ax = axs[1]
        ax.set_title("Re(ψ)")
        ax.set_zlim3d(self._zmin, self._zmax)
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        ax.plot_surface(
            np_qyv,
            np_qxv,
            (1 / dq) * np.real(q_data),
            cmap=colormap,
            vmin=self._vmin,
            vmax=self._vmax,
        )

        ax = axs[2]
        ax.set_title("Im(ψ)")
        ax.set_zlim3d(self._zmin, self._zmax)
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        ax.plot_surface(
            np_qyv,
            np_qxv,
            (1 / dq) * np.imag(q_data),
            cmap=colormap,
            vmin=self._vmin,
            vmax=self._vmax,
        )

        psi_label = self._psifunc_label
        fig.suptitle(f"t={t:6.3f},dt={self._delta_t},n1={self._n1}," + psi_label)
        filename = f"{self._frames_dir}/t_{t:06.3f}.png"
        print("writing to file : ", filename)
        fig.savefig(filename)
        plt.close(fig)

    def generate_report(self) -> None:
        """"""
        moviefile = self.moviefile
        print("producing video : ", moviefile)
        stream = ffmpeg.input(
            f"{self._frames_dir}/t_*.png", pattern_type="glob", framerate=8
        )
        ffmpeg.output(stream, moviefile, pix_fmt="yuv420p").run()

    @property
    def tagname(self):
        return f"exy0_{self._n1}b.{self._num_nucl_iters}n.{self._num_elec_iters}e"

    @property
    def moviefile(self):
        return f"{self._outdir}/{self.tagname}.mp4"
