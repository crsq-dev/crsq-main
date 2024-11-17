""" Hamiltonian calculation using Radial function QROM circuit
"""

from typing import List
import time
import logging
from contextlib import contextmanager

from qiskit import QuantumRegister
from qiskit.circuit.library import XGate
from crsq_heap import heap
import crsq_arithmetic as ari
from crsq_arithmetic import ast
from crsq.blocks import wave_function, discretization, hamiltonian, radial_func_qrom

logger = logging.getLogger(__name__)
LOG_TIME_THRESH = 1


@contextmanager
def check_time(label: str):
    """context manager to check time.
    :param label: label used for logs.
    """
    logger.info("%s start", label)
    t1 = time.time()
    yield
    dt = time.time() - t1
    if dt > LOG_TIME_THRESH:
        logger.info("%s took %d msec", label, round(dt * 1000))
    else:
        logger.info("%s end", label)


class RfqPotentialSpec:
    """Args:
    wfr_spec: WaveFunctionRegisterSpec
    elec_elec_potential_func: function of distance between two electrons
    elec_nucl_potential_func: function of distance between an electron and a nucleus
    """

    def __init__(
        self,
        wfr_spec: wave_function.WaveFunctionRegisterSpec,
        elec_elec_potential_func,
        elec_nucl_potential_func,
    ):
        assert isinstance(wfr_spec, wave_function.WaveFunctionRegisterSpec)
        self._wfr_spec = wfr_spec
        self._elec_elec_potential_func = elec_elec_potential_func
        self._elec_nucl_potential_func = elec_nucl_potential_func

    @property
    def wfr_spec(self):
        return self._wfr_spec

    @property
    def radial_func(self):
        return self._elec_elec_potential_func

    @property
    def elec_elec_potential_func(self):
        return self._elec_elec_potential_func


class RfqElectronPotentialBlock(heap.Frame):
    def __init__(
        self,
        rfq_spec: RfqPotentialSpec,
        ham_spec: hamiltonian.HamiltonianSpec,
        disc_spec: discretization.DiscretizationSpec,
        allocate=True,
        build=True,
    ):
        super().__init__(label="RFQHp")
        self._rfq_spec = rfq_spec
        self._ham_spec = ham_spec
        self._wfr_spec = ham_spec.wfr_spec
        self._disc_spec = disc_spec
        self._eregs: List[List[List[QuantumRegister]]] = []
        self._nregs: List[List[List[QuantumRegister]]] = []
        if allocate:
            self.allocate_registers()
            if build:
                with check_time("RfqElectronPotentialBlock.build_circuit"):
                    self.build_circuits()

    def allocate_registers(self):
        """Allocate registers for the Hamiltonian calculation"""
        self._eregs = self._wfr_spec.allocate_elec_registers()
        self._nregs = self._wfr_spec.allocate_nucl_registers()
        self.add_param(("eregs", self._eregs), ("nregs", self._nregs))

    def build_circuits(self):
        self._build_elec_elec_potential_terms()
        self._build_elec_nucl_potential_terms()

    def _build_elec_elec_potential_terms(self):
        wfr_spec = self._wfr_spec
        rfq_spec = self._rfq_spec
        for ie in range(wfr_spec.num_electrons):
            for ih in range(ie + 1, wfr_spec.num_electrons):
                t1 = time.time()
                # electron-electron potential term
                scope: ast.Scope = ast.new_scope(self)
                x1r = scope.register(self._eregs[ie][0])
                y1r = scope.register(self._eregs[ie][1])
                x2r = scope.register(self._eregs[ih][0])
                y2r = scope.register(self._eregs[ih][1])
                x1r -= x2r
                y1r -= y2r
                scope.build_circuit()

                rfq = radial_func_qrom.RadialFuncQrom(
                    wfr_spec.num_coordinate_bits,
                    wfr_spec.delta_q,
                    rfq_spec.elec_elec_potential_func,
                )
                self.invoke(rfq.bind(x=x1r.register, y=y1r.register))

                scope.build_inverse_circuit()

                scope.close()

                dt = time.time() - t1
                if dt > LOG_TIME_THRESH:
                    logger.info("  Vee(%d,%d) done. %d msec", ie, ih, round(dt * 1000))

    def _build_elec_nucl_potential_terms(
            self):
        wfr_spec = self._wfr_spec
        ham_spec = self._ham_spec
        rfq_spec = self._rfq_spec
        nuclei_data = ham_spec.nuclei_data
        num_moving_nuclei = wfr_spec.num_moving_nuclei
        # TODO : moving atoms are not implemented yet.
        for ie in range(wfr_spec.num_electrons):
            for ia in range(wfr_spec.num_stationary_nuclei):
                t1 = time.time()
                scope: ast.Scope = ast.new_scope(self)
                exr = scope.register(self._eregs[ie][0])
                eyr = scope.register(self._eregs[ie][1])
                ndata = nuclei_data[num_moving_nuclei + ia]
                pos = ndata["pos"]
                if not (pos[0] == 0 and pos[1] == 0):
                    axc = scope.constant(int(pos[0]/wfr_spec.delta_q))
                    ayc = scope.constant(int(pos[1]/wfr_spec.delta_q))
                    exr -= axc
                    eyr -= ayc
                scope.build_circuit()

                rfq = radial_func_qrom.RadialFuncQrom(
                    wfr_spec.num_coordinate_bits,
                    wfr_spec.delta_q,
                    rfq_spec._elec_nucl_potential_func
                )
                self.invoke(rfq.bind(x=exr.register, y=eyr.register))

                scope.build_inverse_circuit()

                scope.close()

                dt = time.time() - t1
                if dt > LOG_TIME_THRESH:
                    logger.info("  Ven(%d,%d) done. %d msec", ie, ia, round(dt * 1000))
    
    def bind(self, eregs, nregs):
        return heap.Binding(self, {"eregs": eregs, "nregs": nregs})
