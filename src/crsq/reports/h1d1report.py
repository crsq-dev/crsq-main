
from qiskit import QuantumCircuit
import numpy as np
import numpy.typing as npt

import math

import matplotlib.pyplot as plt

from crsq.blocks import wave_function, time_evolution

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
        space_length: float,
        delta_t: float,
        num_elec_iters: int,
        num_nucl_iters: int
    ):
        self.outdir = outdir
        self._title = title
        self._n1 = num_coordinate_bits
        self._M = 1 << self._n1
        self._L = space_length
        self._dq = space_length / self._M
        self._delta_t = delta_t
        self._num_elec_iters = num_elec_iters
        self._num_nucl_iters = num_nucl_iters
        self._x = np.linspace(-self._L / 2, (self._L / 2) - self._dq, self._M)

    def add_circuit_diagram(self, circuit: QuantumCircuit, block_name: str):
        """Add a circuit diagram to the report"""
        fname = f"{self.outdir}/{block_name}.png"
        logger.info(f"Saving circuit diagram of {block_name} to {fname}")
        circuit.draw(output="mpl", filename=fname, scale=0.6, fold=100)

    def open_figure(self):
        """Start a plot"""
        self._fig, self._axs = plt.subplots(3, 1, figsize=(6, 12))
        self._fig.suptitle(self._title)
        self._axs[0].set_title("abs")
        self._axs[1].set_title("real")
        self._axs[2].set_title("imag")

    def  add_wave_function_plot(self, time: float, y: npt.NDArray[np.complex128]):
        """Add a statevector to the report"""
        ab = np.abs(y) / math.sqrt(self._dq)
        re = np.real(y) / math.sqrt(self._dq)
        im = np.imag(y) / math.sqrt(self._dq)
        self._axs[0].plot(self._x, ab, label=f"t={time:4.3f}")
        self._axs[1].plot(self._x, re, label=f"t={time:4.3f}")
        self._axs[2].plot(self._x, im, label=f"t={time:4.3f}")

    def generate_figure(self):
        self._axs[0].legend()
        # axs[1].legend()
        # axs[2].legend()
        self._fig.savefig(
            self.outdir
            + f"/ex0_{self._n1}b.{self._num_nucl_iters}n.{self._num_elec_iters}e.dist.png"
        )
        plt.close(self._fig)
