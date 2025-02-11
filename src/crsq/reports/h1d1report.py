
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
        wfr_spec: wave_function.WaveFunctionRegisterSpec,
        evo_spec: time_evolution.TimeEvolutionSpec
    ):
        if wfr_spec.dimension != 1:
            raise ValueError("This report is for 1D wave functions only")
        self.outdir = outdir
        self._title = title
        self._wfr_spec = wfr_spec
        self._evo_spec = evo_spec
        self._n1 = wfr_spec.num_coordinate_bits
        self._M = 1 << self._n1
        self._L = wfr_spec.space_length
        self._dq = wfr_spec.delta_q
        self._num_elec_iters = evo_spec.num_elec_per_atom_iterations
        self._num_nucl_iters = evo_spec.num_atom_iterations
        self._x = np.linspace(-self._L / 2, (self._L / 2) - self._dq, self._M)

    def add_circuit_diagram(self, circuit: QuantumCircuit, block_name: str):
        """Add a circuit diagram to the report"""
        fname = f"{self.outdir}/{block_name}.png"
        logger.info(f"Saving circuit diagram of {block_name} to {fname}")
        circuit.draw(output="mpl", filename=fname, scale=0.6, fold=100)

    def open_figure(self):
        """Start a plot"""
        self._fig, self._axs = plt.subplots(3, 1, figsize=(6, 12))
        self._axs[0].set_title("abs")
        self._axs[1].set_title("real")
        self._axs[2].set_title("imag")

    def add_wave_function_plot(self, time: float, y: npt.NDArray[np.complex128]):
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
