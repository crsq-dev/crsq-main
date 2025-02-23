""" Hamiltonian calculation using Radial function QROM circuit
"""

from typing import List
from collections.abc import Callable
import time
import logging
from contextlib import contextmanager

from qiskit import QuantumRegister
from crsq_heap import heap
from crsq_arithmetic import ast
from crsq.blocks import (
    wave_function,
    discretization,
    hamiltonian,
    radial_func_qrom,
    radial_func_gray_code_qrom,
)

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
    param:wfr_spec: WaveFunctionRegisterSpec
    param:elec_elec_potential_func: func(r: float, dq: float) -> float: function of distance between two electrons
    param:elec_nucl_potential_func: func(r: float, dq: float) -> float: function of distance between an electron and a nucleus
    """

    def __init__(
        self,
        wfr_spec: wave_function.WaveFunctionRegisterSpec,
        elec_elec_potential_func: Callable[[float], float],
        elec_nucl_potential_func: Callable[[float], float],
        use_symmetry: bool = True,
        use_transpose: bool = True,
        use_gray_code: bool = False,
        save_state_vector_per_qrom: bool = False,
    ):
        assert isinstance(wfr_spec, wave_function.WaveFunctionRegisterSpec)
        self._wfr_spec = wfr_spec
        self._elec_elec_potential_func = elec_elec_potential_func
        self._elec_nucl_potential_func = elec_nucl_potential_func
        self._use_symmetry = use_symmetry
        self._use_transpose = use_transpose
        self._use_gray_code = use_gray_code
        self._should_save_state_vector_per_qrom = save_state_vector_per_qrom
        logger.info(
            "RfqPotentialSpec: use_symmetry=%s, use_transpose=%s, save_state_vector_per_qrom=%s",
            use_symmetry,
            use_transpose,
            save_state_vector_per_qrom,
        )

    @property
    def wfr_spec(self):
        return self._wfr_spec

    @property
    def elec_nucl_potential_func(self) -> Callable[[float], float]:
        return self._elec_nucl_potential_func

    @property
    def elec_elec_potential_func(self) -> Callable[[float], float]:
        return self._elec_elec_potential_func

    @property
    def should_save_state_vector_per_qrom(self) -> bool:
        return self._should_save_state_vector_per_qrom

    @property
    def should_use_symmetry(self) -> bool:
        return self._use_symmetry

    @property
    def should_use_transpose(self) -> bool:
        return self._use_transpose

    @property
    def should_use_gray_code(self) -> bool:
        return self._use_gray_code


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
        if self._rfq_spec.should_use_gray_code:
            self._target = QuantumRegister(1, "target")
            self.add_param(self._target)

    def build_circuits(self):
        self._build_elec_elec_potential_terms()
        self._build_elec_nucl_potential_terms()

    def _elec_elec_phase_shift(self, r: float):
        """-delta_t * Vee(r)"""
        return -self._disc_spec.delta_t * self._rfq_spec.elec_elec_potential_func(r)

    def _elec_nucl_phase_shift(self, r: float):
        """-delta_t * Ven(r)"""
        return -self._disc_spec.delta_t * self._rfq_spec.elec_nucl_potential_func(r)

    def _apply_radial_func_qrom_1d(self, xr: ast.Register, rfunc):
        wfr_spec = self._wfr_spec
        rfq_spec = self._rfq_spec
        if rfq_spec.should_use_gray_code:
            rfgcq1d = radial_func_gray_code_qrom.RadialFuncGrayCodeQrom1d(
                wfr_spec.num_coordinate_bits, wfr_spec.delta_q, rfunc
            )
            self.invoke(rfgcq1d.bind(x=xr.register, target=self._target))
        else:
            raise NotImplementedError(
                "1D radial function QROM without graycode is not implemented yet."
            )

    def _apply_radial_func_qrom_2d(self, xr: ast.Register, yr: ast.Register, rfunc):
        wfr_spec = self._wfr_spec
        rfq_spec = self._rfq_spec
        if rfq_spec.should_use_gray_code:
            rfgcq = radial_func_gray_code_qrom.RadialFuncGrayCodeQrom(
                wfr_spec.num_coordinate_bits, wfr_spec.delta_q, rfunc
            )
            self.invoke(rfgcq.bind(x=xr.register, y=yr.register, target=self._target))
        else:
            rfq = radial_func_qrom.RadialFuncQrom(
                wfr_spec.num_coordinate_bits,
                wfr_spec.delta_q,
                rfunc,
                use_symmetry=rfq_spec.should_use_symmetry,
                use_transpose=rfq_spec.should_use_transpose,
            )
            self.invoke(
                rfq.bind(x=xr.register, y=yr.register), invoke_as_instruction=True
            )

    def _build_elec_elec_potential_terms(self):
        wfr_spec = self._wfr_spec
        if wfr_spec.dimension == 1:
            self._build_elec_elec_potential_terms_1d()
        elif wfr_spec.dimension == 2:
            self._build_elec_elec_potential_terms_2d()
        else:
            raise NotImplementedError("Dimension > 2 is not implemented yet.")

    def _build_elec_elec_potential_terms_1d(self):
        logger.info("_build_elec_elec_potential_terms_1d")
        wfr_spec = self._wfr_spec
        for ie in range(wfr_spec.num_electrons):
            for ih in range(ie + 1, wfr_spec.num_electrons):
                t1 = time.time()
                # electron-electron potential term
                scope: ast.Scope = ast.new_scope(self)
                x1r = scope.register(self._eregs[ie][0], signed=True)
                x2r = scope.register(self._eregs[ih][0], signed=True)
                x1r -= x2r
                scope.build_circuit()

                self._apply_radial_func_qrom_1d(x1r, self._elec_elec_phase_shift)

                scope.build_inverse_circuit()

                scope.close()

                dt = time.time() - t1
                if dt > LOG_TIME_THRESH:
                    logger.info("  Vee(%d,%d) done. %d msec", ie, ih, round(dt * 1000))

    def _build_elec_elec_potential_terms_2d(self):
        logger.info("_build_elec_elec_potential_terms_2d")
        wfr_spec = self._wfr_spec
        for ie in range(wfr_spec.num_electrons):
            for ih in range(ie + 1, wfr_spec.num_electrons):
                t1 = time.time()
                # electron-electron potential term
                scope: ast.Scope = ast.new_scope(self)
                x1r = scope.register(self._eregs[ie][0], signed=True)
                y1r = scope.register(self._eregs[ie][1], signed=True)
                x2r = scope.register(self._eregs[ih][0], signed=True)
                y2r = scope.register(self._eregs[ih][1], signed=True)
                x1r -= x2r
                y1r -= y2r
                scope.build_circuit()

                self._apply_radial_func_qrom_2d(x1r, y1r, self._elec_elec_phase_shift)

                scope.build_inverse_circuit()

                scope.close()

                dt = time.time() - t1
                if dt > LOG_TIME_THRESH:
                    logger.info("  Vee(%d,%d) done. %d msec", ie, ih, round(dt * 1000))

    def _build_elec_nucl_potential_terms(self):
        wfr_spec = self._wfr_spec
        logger.info("_build_elec_nucl_potential_terms dimension=%d", wfr_spec.dimension)
        if wfr_spec.dimension == 1:
            self._build_elec_nucl_potential_terms_1d()
        elif wfr_spec.dimension == 2:
            self._build_elec_nucl_potential_terms_2d()
        else:
            raise NotImplementedError("Dimension > 2 is not implemented yet.")

    def _build_elec_nucl_potential_terms_1d(self):
        logger.info("_build_elec_nucl_potential_terms_1d:")
        wfr_spec = self._wfr_spec
        ham_spec = self._ham_spec
        nuclei_data = ham_spec.nuclei_data
        num_moving_nuclei = wfr_spec.num_moving_nuclei
        # TODO : moving atoms are not implemented yet.
        for ie in range(wfr_spec.num_electrons):
            for ia in range(wfr_spec.num_stationary_nuclei):
                t1 = time.time()
                scope: ast.Scope = ast.new_scope(self)
                exr = scope.register(self._eregs[ie][0], signed=True)
                ndata = nuclei_data[num_moving_nuclei + ia]
                x0 = ndata["pos"]
                if not (x0 == 0):
                    axc = scope.constant(
                        x0,
                        wfr_spec.num_coordinate_bits,
                        signed=True,
                    )
                    exr -= axc
                # we don't need abs here because the qrom has
                # data for inputs in [-M/2, M/2)
                scope.build_circuit()

                self._apply_radial_func_qrom_1d(exr, self._elec_nucl_phase_shift)

                scope.build_inverse_circuit()

                scope.close()

                dt = time.time() - t1
                if dt > LOG_TIME_THRESH:
                    logger.info("  Ven(%d,%d) done. %d msec", ie, ia, round(dt * 1000))

    def _build_elec_nucl_potential_terms_2d(self):
        logger.info("_build_elec_nucl_potential_terms_2d")
        wfr_spec = self._wfr_spec
        ham_spec = self._ham_spec
        nuclei_data = ham_spec.nuclei_data
        num_moving_nuclei = wfr_spec.num_moving_nuclei
        # TODO : moving atoms are not implemented yet.
        for ie in range(wfr_spec.num_electrons):
            for ia in range(wfr_spec.num_stationary_nuclei):
                t1 = time.time()
                scope: ast.Scope = ast.new_scope(self)
                exr = scope.register(self._eregs[ie][0], signed=True)
                eyr = scope.register(self._eregs[ie][1], signed=True)
                ndata = nuclei_data[num_moving_nuclei + ia]
                pos = ndata["pos"]
                if not (pos[0] == 0 and pos[1] == 0):
                    axc = scope.constant(
                        int(pos[0] / wfr_spec.delta_q),
                        wfr_spec.num_coordinate_bits,
                        signed=True,
                    )
                    ayc = scope.constant(
                        int(pos[1] / wfr_spec.delta_q),
                        wfr_spec.num_coordinate_bits,
                        signed=True,
                    )
                    exr -= axc
                    eyr -= ayc
                scope.build_circuit()

                self._apply_radial_func_qrom_2d(exr, eyr, self._elec_nucl_phase_shift)

                scope.build_inverse_circuit()

                scope.close()

                dt = time.time() - t1
                if dt > LOG_TIME_THRESH:
                    logger.info("  Ven(%d,%d) done. %d msec", ie, ia, round(dt * 1000))

    def bind(self, eregs, nregs, target: QuantumRegister = None):
        if self._rfq_spec.should_use_gray_code:
            return heap.Binding(
                self, {"eregs": eregs, "nregs": nregs, "target": target}
            )
        else:
            return heap.Binding(self, {"eregs": eregs, "nregs": nregs})
