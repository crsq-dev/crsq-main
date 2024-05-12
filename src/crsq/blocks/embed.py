""" State embedding gate
"""

from typing import List
import math
import cmath
import time
import logging

from qiskit import QuantumRegister
from crsq.heap import Frame, Binding

logger = logging.getLogger(__name__)
LOG_TIME_THRESH=1

def calc_phase_tree(phases: List[float]):
    """ calc phase tree"""
    n = len(phases)
    half=n//2
    if n == 2:
        phi = phases[0] - phases[half]
        return [phi]
    else:
        p0 = phases[:half]
        p1 = phases[half:]
        t0 = calc_phase_tree(p0)
        t1 = calc_phase_tree(p1)
        phi = phases[0] - phases[half] - (t0[0] - t1[0])/2
        return [phi, [t0, t1]]


class StateEmbedGate(Frame):
    """ State embedding gate
    """
    def __init__(self, data: List[float] | List[complex], build=True):
        super().__init__()
        # logger.info("start: StateEmbedGate()")
        t1 = time.time()
        num_bits = math.ceil(math.log2(len(data)))
        if 2**num_bits != len(data):
            raise ValueError("data length must be a power of 2")
        self._num_bits = num_bits
        self._label = f"emb({num_bits})"
        self._data = data
        self._qreg: QuantumRegister
        self.allocate_registers()
        if build:
            self.build_circuit()
        t2 = time.time()
        dt = t2 - t1
        if dt > LOG_TIME_THRESH:
            logger.info("end  : StateEmbedGate() %f msec", round(dt*1000))

    def allocate_registers(self):
        """ allocate """
        self._qreg = QuantumRegister(self._num_bits, "q")
        self.add_param(self._qreg)

    def build_circuit(self):
        """ build """
        logger.info("call: AmplitudeEmbedGate")
        n = len(self._data)
        amp_gate = AmplitudeEmbedGate(self._data, 0, n)
        logger.info("done: AmplitudeEmbedGate")
        self.invoke(amp_gate.bind(q=self._qreg))

        logger.info("call: PhaseEmbedGate")
        phase_gate = PhaseEmbedGate(self._data, 0, n)
        logger.info("done: PhaseEmbedGate")
        self.invoke(phase_gate.bind(q=self._qreg))

    def bind(self, q: QuantumRegister)-> Binding:
        """ bind """
        return Binding(self, {"q": q})

def _array_equals(data: list[float] | list[complex], start, mid)->bool:
    """ compares content data with argument
    """
    for i in range(mid-start):
        if data[start+i] != data[mid+i]:
            return False
    return True

class AmplitudeEmbedGate(Frame):
    """ Amplitude setting gate
    """
    def __init__(self, data: List[float] | List[complex], start: int=0, end: int=0, build=True):
        super().__init__()
        if start == 0 and end == 0:
            end = len(data)
        num_bits = math.ceil(math.log2(end-start))
        if 2**num_bits != end-start:
            raise ValueError("data length must be a power of 2")
        self._num_bits = num_bits
        self._label = f"embθ({num_bits})"
        self._data = data
        self._start = start
        self._end = end
        self._qreg: QuantumRegister
        self._norm: float
        self._has_effect: bool = False
        self.allocate_registers()
        if build:
            self.build_circuit()

    @property
    def norm(self) -> float:
        """ sum of square(data[i])
        """
        return self._norm

    @property
    def has_effect(self) -> bool:
        """ True when gate has a non-null effect"""
        return self._has_effect


    def allocate_registers(self):
        """ allocate """
        self._qreg = QuantumRegister(self._num_bits, "q")
        self.add_param(self._qreg)

    def build_circuit(self):
        """ build """
        qc = self.circuit
        n = self._num_bits
        target = self._qreg[n-1]
        minor_bits = self._qreg[:n-1]
        minor_reg = QuantumRegister(bits=minor_bits)
        n = self._num_bits

        start = self._start
        end = self._end
        mid = (start + end)//2
        if n == 1:
            logger.info(" AmplitudeEmbedGate: start == %d", start)
            a0 = self._data[start+0]
            a1 = self._data[start+1]
            s0 = abs(a0)
            s1 = abs(a1)
        else:
            data = self._data
            amp0 = AmplitudeEmbedGate(data, start, mid)
            amp1 = AmplitudeEmbedGate(data, mid, end)
            s0 = amp0.norm
            s1 = amp1.norm

        self._norm = math.sqrt(s0*s0 + s1*s1)

        has_effect = False
        if s0 == 0 and s1 == 0:
            pass
        elif s1 == 0:
            pass
        elif s0 == 0:
            qc.x(target)
            has_effect = True
        elif s0 == s1:
            qc.h(target)
            has_effect = True
        else:
            th = 2*math.atan2(s1,s0)
            qc.ry(th, target)
            has_effect = True

        if n > 1:
            # mid
            logger.info("start: invoke: [%d,%d]", start, end)
            if s0 > 0 and s1 == 0 and amp0.has_effect:
                self.invoke(amp0.bind(q=minor_reg))
                has_effect = True
            elif s0 == 0 and s1 > 0 and amp1.has_effect:
                self.invoke(amp1.bind(q=minor_reg))
                has_effect = True
            elif s0 > 0 and s1 > 0:
                if s0 == s1 and _array_equals(data, start, mid):
                    if amp0.has_effect:
                        self.invoke(amp0.bind(q=minor_reg))
                        has_effect = True
                else:
                    if amp0.has_effect:
                        self.invoke_with_control(
                            amp0.bind(q=QuantumRegister(bits=minor_bits)),
                            ctrl_bits=[target], ctrl_str="0")
                        has_effect = True
                    if amp1.has_effect:
                        self.invoke_with_control(
                            amp1.bind(q=QuantumRegister(bits=minor_bits)),
                            ctrl_bits=[target], ctrl_str="1")
                        has_effect = True
            logger.info("end  : invoke: [%d,%d] has_effect=%d", start, end, has_effect)

        self._has_effect = has_effect

    def bind(self, q: QuantumRegister):
        """ make binding """
        return Binding(self, {"q": q})

class PhaseEmbedGate(Frame):
    """ phase embed gate"""
    def __init__(self, data: List[float] | List[complex], start: int = 0, end: int = 0, build=True):
        super().__init__()
        if start == 0 and end == 0:
            end = len(data)
        num_bits = math.ceil(math.log2(end-start))
        if 2**num_bits != end-start:
            raise ValueError("data length must be a power of 2")
        self._num_bits = num_bits
        self._data = data
        self._start = start
        self._end = end
        self._average: float
        self._label = f"embφ({num_bits})"
        self._qreg: QuantumRegister
        self._has_effect: bool = False
        self.allocate_registers()
        if build:
            self.build_circuit()

    @property
    def has_effect(self):
        """ whether the gate has an effect or not"""
        return self._has_effect

    @property
    def average(self):
        """ phase of first entry in range
        """
        return self._average

    def allocate_registers(self):
        """ allocate"""
        self._qreg = QuantumRegister(self._num_bits, "q")
        self.add_param(self._qreg)

    def build_circuit(self):
        """ build """
        qc = self.circuit
        n = self._num_bits
        target = self._qreg[n-1]
        minor_bits = self._qreg[:n-1]
        n = self._num_bits
        start = self._start
        end = self._end
        mid  = (start+end)//2
        data = self._data
        if n == 1:
            avg0 = cmath.phase(data[start+0])
            avg1 = cmath.phase(data[start+1])
        else:
            phg0 = PhaseEmbedGate(data, start, mid)
            phg1 = PhaseEmbedGate(data, mid, end)
            avg0 = phg0.average
            avg1 = phg1.average
        phi = avg1 - avg0
        self._average = (avg0 + avg1)/2

        has_effect = False
        if phi != 0.0:
            qc.rz(phi, target)
            has_effect = True

        if n > 1:
            if phg0.has_effect:
                self.invoke_with_control(
                    phg0.bind(q=QuantumRegister(bits=minor_bits)),
                    ctrl_bits=[target], ctrl_str="0")
                has_effect = True
            if phg1.has_effect:
                self.invoke_with_control(
                    phg1.bind(q=QuantumRegister(bits=minor_bits)),
                    ctrl_bits=[target], ctrl_str="1")
                has_effect = True
        self._has_effect = has_effect


    def bind(self, q: QuantumRegister):
        """ bind """
        return Binding(self, {"q": q})
