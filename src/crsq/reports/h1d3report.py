"""Report generator for 1 H atom, 3 dimension
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

class H1D3Report:
    """Report generator for 1 H atom, 3 dimension
       Draw a slice of the data at z = 0.

    """

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
            Callable[[float, float, float], float] | Dict[str, Callable[[float, float, float], float]]
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
        self._images_dir = outdir + "/images"
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
            self._xq, self._yq = numpy.meshgrid(iq, iq)
        else:
            self._xq, self._yq  = numpy.meshgrid(numpy.arange(M), numpy.arange(M))

        dq = self._dq
        self._x = self._xq * dq
        self._y = self._yq * dq

        q3x = iq[:, None, None]*dq
        q3y = iq[None, :, None]*dq
        q3z = iq[None, None, :]*dq
        # potential energy Hp(x,y,0)
        if isinstance(self._hp_func, dict):
            self._hp = {
                key: self._hp_func[key](q3x, q3y, q3z)
                for key in self._hp_func.keys()
            }
        else:
            self._hp = {"Hp": self._hp_func(q3x, q3y, q3z)}
        # discretized wave number values

        kq = numpy.mod(numpy.linspace(-M // 2, M // 2 - 1, M), M) - M // 2
        kq2 = numpy.square(kq)
        self._kxq = kq[:, None]
        self._kyq = kq[None, :]

        dk = self._dk
        self._kx = self._kxq* dk
        self._ky = self._kyq* dk

        self._kxq2 = kq2[:, None]
        self._kyq2 = kq2[None, :]
        self._kq2 = self._kxq2 + self._kyq2

        self._k3xq = kq[:, None, None]
        self._k3yq = kq[None, :, None]
        self._k3zq = kq[None, None, :]
        self._k3x = self._k3xq * dk
        self._k3y = self._k3yq * dk
        self._k3z = self._k3zq * dk
        self._k3xq2 = kq2[:, None, None]
        self._k3yq2 = kq2[None, :, None]
        self._k3zq2 = kq2[None, None, :]
        self._k3q2 = self._k3xq2 + self._k3yq2 + self._k3zq2

        self._k32 = self._k3q2 * (dk * dk)
        self._hk3 = self._k32 / 2.0

        # make the frames directory
        self._prepare_dir(clean=False)

    def set_color_map(self, colormap_name: str):
        self._colormap_name = colormap_name

    def _prepare_dir(self, clean: bool) -> None:
        """"""
        frames_dir = self._frames_dir
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)
        images_dir = self._images_dir
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        if clean:
            for f in glob.glob(f"{images_dir}/t_*.png"):
                os.remove(f)

    def add_circuit_diagram(self, circuit: QuantumCircuit, block_name: str):
        """Add a circuit diagram to the report"""
        fname = f"{self._outdir}/{block_name}.png"
        logger.info(f"Saving circuit diagram of {block_name} to {fname}")
        circuit.draw(output="mpl", filename=fname, scale=0.6, fold=100)

    def open_report(self, clean = True) -> None:
        """"""
        self._prepare_dir(clean)

        self._trace_time = []
        self._autocorr_trace = []
        self._hk_trace = []
        self._hp_trace = {}
        for key in self._hp.keys():
            self._hp_trace[key] = []

    def add_data_sample(
        self, label: str, t: float, q_data3: npt.NDArray[numpy.complex128]
    ) -> None:
        """save 3d grid data to a text file"""
        file_name = self._frames_dir + f"/{t:06.3f}.{label}.csv"
        q_data2 = q_data3[:, :, 0]
        print("Saving to : ", file_name)
        self.write_2d_data(file_name, q_data2)

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
        q_data3: npt.NDArray[numpy.complex128],
        p_data3: npt.NDArray[numpy.complex128],
    ) -> None:
        """ """
        q_data2 = q_data3[:, :, 0]
        p_data2 = p_data3[:, :, 0]
        logger.info("Producing frame for t=%f", t)
        p_data2_shifted = self.swap_hl_ij(p_data2)
        if self._plot_type == "none":
            return
        if self._plot_type == "2d":
            self.produce_frame2d(t, q_data2, p_data2_shifted)
        elif self._plot_type == "3d":
            self.produce_frame3d(t, q_data2)
        elif self._plot_type == "3d-re":
            self.produce_frame3d_re(t, q_data2)
        elif self._plot_type == "3d-qp":
            self.produce_frame3d_qp(t, q_data2, p_data2_shifted)
        elif self._plot_type == "3d-3":
            self.produce_frame3d3(t, q_data2)
        elif self._plot_type == "3d-3qp":
            self.produce_frame3d3qp(t, q_data2, p_data2_shifted)
        else:
            raise ValueError(f"Unknown plot type : {self._plot_type}")

    def swap_hl_ij(self, a_ij):
        """shift signed index iq by M/2 using modulo M"""
        M = self._M
        ar = numpy.roll(a_ij, (M//2, M//2), axis=(0,1))
        return ar

    def swap_hl_i(self, a_i):
        """shift signed index iq by M/2 using modulo M"""
        M = self._M
        ari = numpy.roll(a_i, M//2, axis=0)
        # ind_i = iq[:,None]
        return ari

    def swap_hl_j(self, a_j):
        """shift signed index iq by M/2 using modulo M"""
        M = self._M
        arj = numpy.roll(a_j, M//2, axis=1)
        # ind_j = iq[None,:]
        return arj

    def produce_frame2d(
        self,
        t: float,
        q_data2: npt.NDArray[numpy.complex128],
        p_data2: npt.NDArray[numpy.complex128],
    ) -> None:
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        colormap = plt.get_cmap(self._colormap_name)
        ax: plt.Axes = axs[0]
        # ax.imshow(numpy.abs(self._psi1q))
        sw_q_data = self.swap_hl_ij(q_data2)
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
            numpy.abs(p_data2),
            cmap=colormap,
            vmin=self._kvmin,
            vmax=self._kvmax,
        )
        px.set_xlabel("kxq")
        px.set_ylabel("kyq")
        # ax.legend()
        filename = f"{self._images_dir}/t_{t:06.3f}_2d.png"
        print("writing to file : ", filename)
        fig.savefig(filename)
        plt.close(fig)

    def produce_frame3d(self, t, q_data2: npt.NDArray[numpy.complex128]) -> None:
        fig, ax = plt.subplots(
            subplot_kw={"projection": "3d"}, figsize=(6, 5.5), layout="constrained"
        )
        colormap = plt.get_cmap(self._colormap_name)
        dq = self._dq
        sw_x = self.swap_hl_j(self._x)
        sw_y = self.swap_hl_i(self._y)
        sw_q_data2 = self.swap_hl_ij(q_data2)

        ax.set_title("|ψ|")
        ax.set_zlim3d(self._zmin, self._zmax)
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        np_qxv = sw_x
        np_qyv = sw_y
        scale = math.pow(dq, -3/2)
        ax.plot_surface(
            np_qyv,
            np_qxv,
            scale * numpy.abs(sw_q_data2),
            cmap=colormap,
            vmin=self._vmin,
            vmax=self._vmax,
        )

        psi_label = self._psifunc_label
        fig.suptitle(f"t={t:6.3f},dt={self._delta_t},n1={self._n1}," + psi_label)
        filename = f"{self._images_dir}/t_{t:06.3f}_3d.png"
        print("writing to file : ", filename)
        fig.savefig(filename)
        plt.close(fig)

    def produce_frame3d_re(
        self, t: float, q_data2: npt.NDArray[numpy.complex128]
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
        np_qxv = self.swap_hl_j(self._x)
        np_qyv = self.swap_hl_i(self._y)
        sw_q_data = self.swap_hl_ij(q_data2)
        scale = math.pow(dq, -3/2)
        ax.plot_surface(
            np_qyv,
            np_qxv,
            scale * numpy.real(sw_q_data),
            cmap=colormap,
            vmin=self._vmin,
            vmax=self._vmax,
        )

        filename = f"{self._images_dir}/t_{t:06.3f}_3d-re.png"
        print("writing to file : ", filename)
        fig.savefig(filename)
        plt.close(fig)

    def produce_frame3d_qp(
        self, t: float, q_data2: npt.NDArray[numpy.complex128], shifted_p_data2: npt.NDArray[numpy.complex128]
    ) -> None:
        fig, axs = plt.subplots(
            2, 1, subplot_kw={"projection": "3d"}, figsize=(6, 11), layout="constrained"
        )
        colormap = plt.get_cmap(self._colormap_name)
        dq = self._dq

        ax: plt.Axes = axs[0]
        psi_label = self._psifunc_label
        ax.set_title(f"Re(ψ) [t={t:6.3f},{psi_label}]")
        ax.set_zlim3d(self._zmin, self._zmax)
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        np_qxv = self.swap_hl_j(self._x)
        np_qyv = self.swap_hl_i(self._y)
        sw_q_data2 = self.swap_hl_ij(q_data2)
        ax.plot_surface(
            np_qyv,
            np_qxv,
            math.pow(dq, -3/2) * numpy.real(sw_q_data2),
            cmap=colormap,
            vmin=self._vmin,
            vmax=self._vmax,
        )

        ax: plt.Axes = axs[1]
        ax.set_title("Re(ψ\u0303)")
        dk = self._dk
        np_ky = self._shift_p_data(self._ky)
        np_kx = self._shift_p_data(self._kx)
        ax.set_zlim3d(self._kzmin, self._kzmax)
        ax.set_xlabel("ky")
        ax.set_ylabel("kx")
        ax.plot_surface(
            np_ky,
            np_kx,
            math.pow(dk, -3/2) * numpy.real(shifted_p_data2),
            cmap=colormap,
            vmin=self._kvmin,
            vmax=self._kvmax,
        )

        filename = f"{self._images_dir}/t_{t:06.3f}_3d-qp.png"
        print("writing to file : ", filename)
        fig.savefig(filename)
        plt.close(fig)

    def produce_frame3d3(self, t: float, q_data2: npt.NDArray[numpy.complex128]) -> None:
        fig, axs = plt.subplots(
            1, 3, subplot_kw={"projection": "3d"}, figsize=(15, 6), layout="constrained"
        )
        self._produce_frame3d3q(t, q_data2, axs)
        fig.suptitle(self._title + f" t={t:6.3f}")
        filename = f"{self._images_dir}/t_{t:06.3f}_3d3.png"
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
        filename = f"{self._images_dir}/t_{t:06.3f}.png"
        print("writing to file : ", filename)
        fig.savefig(filename)
        plt.close(fig)

    def _produce_frame3d3q(
        self, t: float, q_data2: npt.NDArray[numpy.complex128], axs
    ) -> None:
        colormap = plt.get_cmap(self._colormap_name)
        dq = self._dq

        ax: plt.Axes = axs[0]
        ax.set_title("(A) |ψ|")
        ax.set_zlim3d(self._zmin, self._zmax)
        ax.set_xlabel("y")
        ax.set_ylabel("x")

        np_qxv = self.swap_hl_j(self._x)
        np_qyv = self.swap_hl_i(self._y)
        sw_q_data2 = self.swap_hl_ij(q_data2)

        ax.plot_surface(
            np_qyv,
            np_qxv,
            math.pow(dq, -3/2) * numpy.abs(sw_q_data2),
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
            math.pow(dq, -3/2) * numpy.real(sw_q_data2),
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
            math.pow(dq, -3/2) * numpy.imag(sw_q_data2),
            cmap=colormap,
            vmin=self._vmin,
            vmax=self._vmax,
        )

    def _produce_frame3d3p(
        self, t: float, p_data2: npt.NDArray[numpy.complex128], axs
    ) -> None:
        colormap = plt.get_cmap(self._colormap_name)
        dk = self._dk

        ax: plt.Axes = axs[0]
        ax.set_title("(D) |ψ\u0303|")
        ax.set_zlim3d(self._kzmin, self._kzmax)
        ax.set_xlabel("ky")
        ax.set_ylabel("kx")

        np_kx = self.swap_hl_j(self._kx)
        np_ky = self.swap_hl_i(self._ky)
        shifted_p_data2 = self.swap_hl_ij(p_data2)
        ax.plot_surface(
            np_ky,
            np_kx,
            math.pow(dk, -3/2) * numpy.abs(shifted_p_data2),
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
            math.pow(dk, -3/2) * numpy.real(shifted_p_data2),
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
            math.pow(dk, -3/2) * numpy.imag(shifted_p_data2),
            cmap=colormap,
            vmin=self._kvmin,
            vmax=self._kvmax,
        )

    def record_energy(self, t, q_data3, p_data3, q0_data=None):
        logger.info("Recording energy at t=%f", t)

        csvdata = {}
        csvdata["t"] = t
        Hk = numpy.sum(numpy.abs(p_data3)**2 * self._hk3).item()
        csvdata["Hk"] = Hk

        for key in self._hp.keys():
            hpf = self._hp[key]
            Hp = numpy.sum(numpy.abs(q_data3)**2 * hpf).item()
            Htot = Hk + Hp
            csvdata[key] = Hp
            logger.info("t=%f, Hk=%f, %s=%f, Hk+%s=%f", t, Hk, key, Hp, key, Htot)
        
        if q0_data is not None:
            prod = numpy.vdot(q0_data, q_data3).item()
            self._autocorr_trace.append(prod)
            csvdata["autocorr.re"] = numpy.real(prod)
            csvdata["autocorr.im"] = numpy.imag(prod)

        csv_filename = self._frames_dir + f"/{t:06.3f}.ene.csv"
        energy_df = pd.DataFrame(
            index=npt,
            data=csvdata,
        )
        energy_df.to_csv(
            csv_filename, index=True, index_label="t", header=True, float_format="%.6f"
        )
        

    def _plot_energy(self):
        logger.info("Plotting energy trace")
        traces = self._read_energy_csv_files()
        fig, axs = plt.subplots(3, 1, figsize=(6, 12), layout="constrained")
        psi_label = self._psifunc_label
        ax = axs[0]
        ax.set_title(f"{self._title} {psi_label}")
        npt = numpy.array(traces["t"])
        nphk = numpy.array(traces["Hk"])
        ax.grid(True)
        ax.plot(npt, nphk, label="Hk(t)")
        for key in self._hp.keys():
            nphp = numpy.array(traces[key])
            ax.plot(npt, nphp, label=f"{key}(t)")
            nphtot = nphk + nphp
            ax.plot(npt, nphtot, label=f"Hk(t)+{key}(t)")
        ax.set_xlabel("t (time)")
        ax.set_ylabel("energy")
        ax.legend()

        ax = axs[1]
        npautocorr_re = numpy.array(traces["autocorr.re"])
        npautocorr_im = numpy.array(traces["autocorr.im"])
        npautocorr = npautocorr_re + 1j * npautocorr_im
        fidelity = numpy.abs(npautocorr) ** 2
        ax.plot(npt, fidelity)
        ax.grid(True)
        ax.set_title("fidelity trace")
        ax.set_xlabel("t (time)")
        ax.set_ylabel("|⟨ψ(0)|ψ(t)⟩|**2")

        ax = axs[2]
        pplus = 1/2*(1+numpy.real(npautocorr))
        ax.plot(npt, pplus, label="P+")
        ax.grid(True)
        ax.set_title("P+ trace")
        ax.set_xlabel("t (time)")
        ax.set_ylabel("P+")

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

    def _read_energy_csv_files(self):
        traces = {}
        # read energy csv files. All files have the same columns and one data row.
        # append the data columns to a list in the 'traces' dict, with the column name as the key.
        for csv_file in glob.glob(f"{self._frames_dir}/*.ene.csv"):
            df = pd.read_csv(csv_file, index_col=0)
            for col in df.columns:
                if col not in traces:
                    traces[col] = []
                traces[col].append(df[col].iloc[0])
        return traces

    def generate_report(self) -> None:
        """"""
        self._plot_energy()
        self._produce_video()

    def _produce_video(self) -> None:
        if self._plot_type == "none":
            logger.info("No video produced for plot type 'none'")
            return
        logger.info("Producing video")
        moviefile = self.moviefile
        if os.path.exists(moviefile):
            logger.info("Removing old video file : %s", moviefile)
            os.remove(moviefile)
        print("producing video : ", moviefile)
        stream = ffmpeg.input(
            f"{self._images_dir}/t_??.???.png", pattern_type="glob", framerate=8
        )
        ffmpeg.output(stream, moviefile, pix_fmt="yuv420p").run()

    @property
    def tagname(self):
        return f"exy0_{self._n1}b.{self._num_nucl_iters}n.{self._num_elec_iters}e"

    @property
    def moviefile(self):
        return f"{self._outdir}/{self.tagname}.mp4"
