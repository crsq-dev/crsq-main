"""Report generator for 1 H atom, 1 dimension
This module takes numpy data types.
"""

from qiskit import QuantumCircuit
import numpy
import numpy.typing as npt
import pandas as pd
import ffmpeg
import os
import glob

from typing import Callable, Dict

import math

import matplotlib.pyplot as plt
from matplotlib import colormaps
import cmasher as cmr

# avoid cmr from getting removed from imports
dummy1 = cmr.amber

from crsq.blocks import wave_function, time_evolution
import crsq.utils.statevector as svec

import logging

logger = logging.getLogger("crsq.reports")


class H1D2ShowPsi:
    def __init__(
        self,
        frames_dir: str,
        num_coordinate_bits: int,
        zmin: float,
        zmax: float,
        vmin: float,
        vmax: float,
        kzmin: float,
        kzmax: float,
        kvmin: float,
        kvmax: float,
        space_length: float,
        colormap_name: str = "cmr.guppy",
    ):
        self._frames_dir = frames_dir
        self._n1 = num_coordinate_bits
        self._M = 1 << self._n1
        M = self._M
        self._L = space_length
        self._dq = space_length / (1 << num_coordinate_bits)
        self._zmin = zmin
        self._zmax = zmax
        self._vmin = vmin
        self._vmax = vmax
        self._kzmin = kzmin
        self._kzmax = kzmax
        self._kvmin = kvmin
        self._kvmax = kvmax
        self._colormap_name = colormap_name
        self._signed = True
        # x,y and qx,qy values
        if self._signed:
            iq = numpy.mod(numpy.linspace(-M // 2, M // 2 - 1, M), M) - M // 2
            self._yq, self._xq = numpy.meshgrid(iq, iq)
        else:
            self._yq, self._xq = numpy.meshgrid(numpy.arange(M), numpy.arange(M))
        dq = self._dq
        self._y = self._yq * dq
        self._x = self._xq * dq

    def _shift_p_data(
        self, p_data: npt.NDArray[numpy.complex128]
    ) -> npt.NDArray[numpy.complex128]:
        """shift p_data by (M/2, M/2) using modulo M"""
        M = self._M
        ind_x, ind_y = numpy.meshgrid(
            (numpy.arange(M) + M // 2) % M, (numpy.arange(M) + M // 2) % M
        )
        return p_data[ind_x, ind_y]

    def swap_hl(
        self, data: npt.NDArray[numpy.complex128]
    ) -> npt.NDArray[numpy.complex128]:
        """swap the left and right halves for i, and upper and lower halves for j of the data[i,j]"""
        if self._signed:
            return self._shift_p_data(data)
        else:
            return data

    def plot(self, ax: plt.Axes, t: float, title: str) -> None:
        """plot the wave function"""
        colormap = plt.get_cmap(self._colormap_name)
        q_data = self.read_data_sample("q", t)
        np_qyv = self.swap_hl(self._y)
        np_qxv = self.swap_hl(self._x)
        sw_q_data = self.swap_hl(q_data)
        dq = self._dq

        ax.set_title(title)
        ax.set_zlim3d(self._zmin, self._zmax)
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        ax.plot_surface(
            np_qyv,
            np_qxv,
            (1 / dq) * numpy.real(sw_q_data),
            cmap=colormap,
            vmin=self._vmin,
            vmax=self._vmax,
        )

    def read_data_sample(self, label: str, t: float) -> npt.NDArray[numpy.complex128]:
        """read q-space 2d grid data from a text file"""
        file_name = f"{self._frames_dir}/{t:06.3f}.{label}.csv"
        with open(file_name, "r") as f:
            s = f.readline().split(",")
            n1 = int(s[0])
            n2 = int(s[1])
            data = numpy.zeros((n1, n2), dtype=numpy.complex128)
            for i in range(n1):
                for j in range(n2):
                    s = f.readline().split(",")
                    data[i, j] = complex(float(s[2]), float(s[3]))
        return data


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
        kzmin: float,
        kzmax: float,
        kvmin: float,
        kvmax: float,
        space_length: float,
        hp_func: (
            Callable[[float, float], float] | Dict[str, Callable[[float, float], float]]
        ),
        delta_t: float,
        num_elec_iters: int,
        num_nucl_iters: int,
        signed=False,
        colormap_name: str = "cmr.guppy",
    ):
        self._outdir = outdir
        self._plot_type = plot_type
        self._colormap_name = colormap_name
        self._frames_dir = outdir + "/frames"
        self._title = title
        self._psifunc_label = psifunc_label
        self._n1 = num_coordinate_bits
        M = 1 << self._n1
        self._M = M
        self._WM = self._M // 2
        self._L = space_length
        self._hp_func = hp_func
        self._dq = self._L / self._M
        self._dq = space_length / self._M
        self._dk = 2 * math.pi / self._L
        self._zmin = zmin
        self._zmax = zmax
        self._vmin = vmin
        self._vmax = vmax
        self._kzmin = kzmin
        self._kzmax = kzmax
        self._kvmin = kvmin
        self._kvmax = kvmax
        self._delta_t = delta_t
        self._total_time = delta_t * num_elec_iters * num_nucl_iters
        self._T = delta_t * num_elec_iters * num_nucl_iters
        self._num_elec_iters = num_elec_iters
        self._num_nucl_iters = num_nucl_iters
        self._signed = signed

        # x,y and qx,qy values
        if self._signed:
            iq = numpy.mod(numpy.linspace(-M // 2, M // 2 - 1, M), M) - M // 2
            self._yq, self._xq = numpy.meshgrid(iq, iq)
        else:
            self._yq, self._xq = numpy.meshgrid(numpy.arange(M), numpy.arange(M))
        dq = self._dq
        self._y = self._yq * dq
        self._x = self._xq * dq
        # potential energy
        if isinstance(self._hp_func, dict):
            self._hp = {
                key: self._hp_func[key](self._x, self._y)
                for key in self._hp_func.keys()
            }
        else:
            self._hp = {"Hp": self._hp_func(self._x, self._y)}
        # discretized wave number values

        kq = numpy.mod(numpy.linspace(-M // 2, M // 2 - 1, M), M) - M // 2
        self._kxq, self._kyq = numpy.meshgrid(kq, kq)

        dk = self._dk
        self._ky = self._kyq * dk
        self._kx = self._kxq * dk

        kq2 = numpy.square(kq)
        self._kxq2, self._kyq2 = numpy.meshgrid(kq2, kq2)
        self._kq2 = self._kxq2 + self._kyq2

        self._k2 = self._kq2 * (dk * dk)
        self._hk = self._k2 / 2.0

        self._prepare_dir()

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

    def add_circuit_diagram(self, circuit: QuantumCircuit, block_name: str):
        """Add a circuit diagram to the report"""
        fname = f"{self._outdir}/{block_name}.png"
        logger.info(f"Saving circuit diagram of {block_name} to {fname}")
        circuit.draw(output="mpl", filename=fname, scale=0.6, fold=100)

    def open_report(self) -> None:
        """"""
        self._trace_time = []
        self._hk_trace = []
        self._hp_trace = {}
        for key in self._hp.keys():
            self._hp_trace[key] = []

    def add_data_sample(
        self, label: str, t: float, q_data: npt.NDArray[numpy.complex128]
    ) -> None:
        """save 2d grid data to a text file"""
        file_name = self._frames_dir + f"/{t:06.3f}.{label}.csv"
        print("Saving to : ", file_name)
        self.write_2d_data(file_name, q_data)

    def write_2d_data(
        self, file_name: str, data: npt.NDArray[numpy.complex128]
    ) -> None:
        """save 2d grid data to a text file"""
        if os.path.exists(file_name):
            logger.info("removing old file : %s", file_name)
            os.remove(file_name)
        logger.info("Saving to : %s", file_name)
        shp = data.shape
        sumdata = numpy.sum(numpy.abs(data) ** 2)
        maxdata = numpy.max(numpy.abs(data))
        logger.info("sum of |ψ|^2 : %f", sumdata)
        logger.info("max of |ψ| : %f", maxdata)
        with open(file_name, "w") as f:
            f.write(f"{shp[0]},{shp[1]}\n")
            for i in range(shp[0]):
                for j in range(shp[1]):
                    f.write(f"{i},{j},{data[i,j].real},{data[i,j].imag}\n")

    def read_2d_data(self, file_name: str) -> npt.NDArray[numpy.complex128]:
        """read 2d grid data from a text file"""
        with open(file_name, "r") as f:
            s = f.readline().split(",")
            n1 = int(s[0])
            n2 = int(s[1])
            data = numpy.zeros((n1, n2), dtype=numpy.complex128)
            for i in range(n1):
                for j in range(n2):
                    s = f.readline().split(",")
                    data[i, j] = complex(float(s[2]), float(s[3]))
        return data

    def read_data_sample(self, label: str, t: float) -> npt.NDArray[numpy.complex128]:
        """read q-space 2d grid data from a text file"""
        file_name = self._frames_dir + f"/{t:06.3f}.{label}.csv"
        return self.read_2d_data(file_name)

    def produce_frame(
        self,
        t: float,
        q_data: npt.NDArray[numpy.complex128],
        p_data: npt.NDArray[numpy.complex128],
    ) -> None:
        """ """
        p_data_shifted = self._shift_p_data(p_data)
        if self._plot_type == "2d":
            self.produce_frame2d(t, q_data, p_data_shifted)
        elif self._plot_type == "3d":
            self.produce_frame3d(t, q_data)
        elif self._plot_type == "3d-re":
            self.produce_frame3d_re(t, q_data)
        elif self._plot_type == "3d-3":
            self.produce_frame3d3(t, q_data)
        elif self._plot_type == "3d-3qp":
            self.produce_frame3d3qp(t, q_data, p_data_shifted)
        else:
            raise ValueError(f"Unknown plot type : {self._plot_type}")

    def _shift_p_data(
        self, p_data: npt.NDArray[numpy.complex128]
    ) -> npt.NDArray[numpy.complex128]:
        """shift p_data by (M/2, M/2) using modulo M"""
        M = self._M
        ind_x, ind_y = numpy.meshgrid(
            (numpy.arange(M) + M // 2) % M, (numpy.arange(M) + M // 2) % M
        )
        return p_data[ind_x, ind_y]

    def swap_hl(
        self, data: npt.NDArray[numpy.complex128]
    ) -> npt.NDArray[numpy.complex128]:
        """swap the left and right halves for i, and upper and lower halves for j of the data[i,j]"""
        if self._signed:
            return self._shift_p_data(data)
        else:
            return data

    def produce_frame2d(
        self,
        t: float,
        q_data: npt.NDArray[numpy.complex128],
        p_data: npt.NDArray[numpy.complex128],
    ) -> None:
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        colormap = plt.get_cmap(self._colormap_name)
        ax: plt.Axes = axs[0]
        # ax.imshow(numpy.abs(self._psi1q))
        sw_q_data = self.swap_hl(q_data)
        ax.imshow(
            numpy.real(sw_q_data),
            cmap=colormap,
            vmin=self._vmin,
            vmax=self._vmax,
        )
        psi_label = self._psifunc_label
        fig.suptitle(f"t={t:6.3f},dt={self._delta_t},n1={self._n1}," + psi_label)
        ax.set_xlabel("xq")
        ax.set_ylabel("yq")
        px: plt.Axes = axs[1]
        px.imshow(
            numpy.abs(p_data),
            cmap=colormap,
            vmin=self._kvmin,
            vmax=self._kvmax,
        )
        px.set_xlabel("kxq")
        px.set_ylabel("kyq")
        # ax.legend()
        filename = f"{self._frames_dir}/t_{t:06.3f}.png"
        print("writing to file : ", filename)
        fig.savefig(filename)
        plt.close(fig)

    def produce_frame3d(self, t, q_data: npt.NDArray[numpy.complex128]) -> None:
        fig, ax = plt.subplots(
            subplot_kw={"projection": "3d"}, figsize=(6, 5.5), layout="constrained"
        )
        colormap = plt.get_cmap(self._colormap_name)
        dq = self._dq
        sw_x = self.swap_hl(self._x)
        sw_y = self.swap_hl(self._y)
        sw_q_data = self.swap_hl(q_data)

        ax.set_title("|ψ|")
        ax.set_zlim3d(self._zmin, self._zmax)
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        np_qxv = sw_x
        np_qyv = sw_y
        ax.plot_surface(
            np_qyv,
            np_qxv,
            (1 / dq) * numpy.abs(sw_q_data),
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

    def produce_frame3d_re(
        self, t: float, q_data: npt.NDArray[numpy.complex128]
    ) -> None:
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
        np_qxv = self.swap_hl(self._x)
        np_qyv = self.swap_hl(self._y)
        sw_q_data = self.swap_hl(q_data)
        ax.plot_surface(
            np_qyv,
            np_qxv,
            (1 / dq) * numpy.real(sw_q_data),
            cmap=colormap,
            vmin=self._vmin,
            vmax=self._vmax,
        )

        filename = f"{self._frames_dir}/t_{t:06.3f}.png"
        print("writing to file : ", filename)
        fig.savefig(filename)
        plt.close(fig)

    def produce_frame3d3(self, t: float, q_data: npt.NDArray[numpy.complex128]) -> None:
        fig, axs = plt.subplots(
            1, 3, subplot_kw={"projection": "3d"}, figsize=(15, 6), layout="constrained"
        )
        self._produce_frame3d3q(t, q_data, axs)
        fig.suptitle(self._title + f" t={t:6.3f}")
        filename = f"{self._frames_dir}/t_{t:06.3f}.png"
        print("writing to file : ", filename)
        fig.savefig(filename)
        plt.close(fig)

    def produce_frame3d3qp(
        self,
        t: float,
        q_data: npt.NDArray[numpy.complex128],
        p_data: npt.NDArray[numpy.complex128],
    ) -> None:
        fig, axs = plt.subplots(
            2,
            3,
            subplot_kw={"projection": "3d"},
            figsize=(15, 10),
            layout="constrained",
        )
        self._produce_frame3d3q(t, q_data, axs[0, :])
        self._produce_frame3d3p(t, p_data, axs[1, :])
        fig.suptitle(self._title + f" t={t:.3f}")
        filename = f"{self._frames_dir}/t_{t:06.3f}.png"
        print("writing to file : ", filename)
        fig.savefig(filename)
        plt.close(fig)

    def _produce_frame3d3q(
        self, t: float, q_data: npt.NDArray[numpy.complex128], axs
    ) -> None:
        colormap = plt.get_cmap(self._colormap_name)
        dq = self._dq

        ax: plt.Axes = axs[0]
        ax.set_title("(A) |ψ|")
        ax.set_zlim3d(self._zmin, self._zmax)
        ax.set_xlabel("y")
        ax.set_ylabel("x")

        np_qyv = self.swap_hl(self._y)
        np_qxv = self.swap_hl(self._x)
        sw_q_data = self.swap_hl(q_data)

        ax.plot_surface(
            np_qyv,
            np_qxv,
            (1 / dq) * numpy.abs(sw_q_data),
            cmap=colormap,
            vmin=self._vmin,
            vmax=self._vmax,
        )

        ax = axs[1]
        ax.set_title("(B) Re(ψ)")
        ax.set_zlim3d(self._zmin, self._zmax)
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        ax.plot_surface(
            np_qyv,
            np_qxv,
            (1 / dq) * numpy.real(sw_q_data),
            cmap=colormap,
            vmin=self._vmin,
            vmax=self._vmax,
        )

        ax = axs[2]
        ax.set_title("(C) Im(ψ)")
        ax.set_zlim3d(self._zmin, self._zmax)
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        ax.plot_surface(
            np_qyv,
            np_qxv,
            (1 / dq) * numpy.imag(sw_q_data),
            cmap=colormap,
            vmin=self._vmin,
            vmax=self._vmax,
        )

    def _produce_frame3d3p(
        self, t: float, p_data: npt.NDArray[numpy.complex128], axs
    ) -> None:
        colormap = plt.get_cmap(self._colormap_name)
        dk = self._dk

        ax: plt.Axes = axs[0]
        ax.set_title("(D) |ψ\u0303|")
        ax.set_zlim3d(self._kzmin, self._kzmax)
        ax.set_xlabel("ky")
        ax.set_ylabel("kx")

        np_ky = self._shift_p_data(self._ky)
        np_kx = self._shift_p_data(self._kx)
        shifted_p_data = p_data
        ax.plot_surface(
            np_ky,
            np_kx,
            (1 / dk) * numpy.abs(shifted_p_data),
            cmap=colormap,
            vmin=self._kvmin,
            vmax=self._kvmax,
        )

        ax = axs[1]
        ax.set_title("(E) Re(ψ\u0303)")
        ax.set_zlim3d(self._kzmin, self._kzmax)
        ax.set_xlabel("ky")
        ax.set_ylabel("kx")
        ax.plot_surface(
            np_ky,
            np_kx,
            (1 / dk) * numpy.real(shifted_p_data),
            cmap=colormap,
            vmin=self._kvmin,
            vmax=self._kvmax,
        )

        ax = axs[2]
        ax.set_title("(F) Im(ψ\u0303)")
        ax.set_zlim3d(self._kzmin, self._kzmax)
        ax.set_xlabel("ky")
        ax.set_ylabel("kx")
        ax.plot_surface(
            np_ky,
            np_kx,
            (1 / dk) * numpy.imag(shifted_p_data),
            cmap=colormap,
            vmin=self._kvmin,
            vmax=self._kvmax,
        )

    def record_energy(self, t, q_data, p_data):
        self._trace_time.append(t)

        Hk = numpy.sum(numpy.abs(p_data * numpy.conjugate(p_data)) * self._hk).item()
        self._hk_trace.append(Hk)

        for key in self._hp.keys():
            hpf = self._hp[key]
            Hp = numpy.sum(numpy.abs(q_data * numpy.conjugate(q_data)) * hpf).item()
            Htot = Hk + Hp
            self._hp_trace[key].append(Hp)
            logger.info("t=%f, Hk=%f, %s=%f, Hk+%s=%f", t, Hk, key, Hp, key, Htot)

    def _plot_energy(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        psi_label = self._psifunc_label
        ax.set_title(f"{self._title} {psi_label}")
        npt = numpy.array(self._trace_time)
        nphk = numpy.array(self._hk_trace)
        ax.grid(True)
        ax.plot(npt, nphk, label="Hk(t)")
        csvdata = {}
        csvdata["Hk"] = nphk
        for key in self._hp.keys():
            nphp = numpy.array(self._hp_trace[key])
            csvdata[key] = nphp
            ax.plot(npt, nphp, label=f"{key}(t)")
        for key in self._hp.keys():
            nphp = numpy.array(self._hp_trace[key])
            nphtot = nphk + nphp
            csvdata["Hk+" + key] = nphtot
            ax.plot(npt, nphtot, label=f"Hk(t)+{key}(t)")
        ax.set_xlabel("t (time)")
        ax.set_ylabel("energy")
        ax.legend()
        filename = f"{self._outdir}/energy_trace.png"
        csv_filename = f"{self._outdir}/energy_trace.csv"
        print("writing to file : ", filename, csv_filename)
        fig.savefig(fname=filename)
        plt.close(fig)

        energy_df = pd.DataFrame(
            index=npt,
            data=csvdata,
        )
        energy_df.to_csv(
            csv_filename, index=True, index_label="t", header=True, float_format="%.6f"
        )

    def generate_report(self) -> None:
        """"""
        self._plot_energy()
        self._produce_video()

    def _produce_video(self) -> None:
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
