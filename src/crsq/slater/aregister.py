""" The A Register
"""

from qiskit import QuantumCircuit, QuantumRegister

from crsq_heap.heap import Frame
import crsq_arithmetic as ari

class ARegisterFrame(Frame):
    """ Frame for ARegister
    """
    def __init__(self, circuit: QuantumCircuit=None):
        super().__init__(circuit=circuit)
        self.aregs: list[list[QuantumRegister]] = [[]]

    def set_aregs(self, aregs: list[list[QuantumRegister]]):
        """ Store the ARegister QuantumRegisters
        """
        self.aregs = aregs
        for dims in aregs:
            self.add_param(*dims)


class ARegister:
    """ The "A" register (electron index register set).

    """
    def __init__(self, num_electron_index_bits: int, num_particles: int, dimension: int, use_spin:bool = True):
        """ num_bits: bits per each coordinate register
            num_regs: number of phi registers
        """
        self.num_electron_index_bits = num_electron_index_bits
        self.num_particles = num_particles
        self.dimension = dimension
        self.use_spin = use_spin

    def add_a_registers(self, frame: ARegisterFrame):
        """ Allocate registers"""
        self.add_a_registers_upto(frame, self.num_particles)

    def add_a_registers_upto(self, frame: ARegisterFrame, aregs_top: int):
        """ Allocate registers on the register set.
        """
        regs = []
        dim_labels = 'xyz'
        for i in range(aregs_top):
            dims = []
            for d in range(self.dimension):
                dims.append(QuantumRegister(self.num_electron_index_bits, f"e{i}{dim_labels[d]}"))
            if self.use_spin:
                dims.append(QuantumRegister(1, f"e{i}s"))
            regs.append(dims)
        frame.set_aregs(regs)

    def set_test_values(self, frame: ARegisterFrame):
        """ set test values to the registers """
        qc = frame.circuit
        aregs = frame.aregs
        for i in range(self.num_particles):
            for d in range(self.dimension):
                ari.set_value(qc, aregs[i][d], i)
            if self.use_spin:
                ari.set_value(qc, aregs[i][d+1], i % 2)  # spin = 0 or 1
