""" QFT helper.

    bit reverser.
"""

class BitReverser:
    """ reverse bits for accessing QFT results
    """

    def __init__(self, n1):
        self._n1 = n1
        self._M = 1 << n1
        self._make_reverse_bit_index()
    
    def _make_reverse_bit_index(self):
        self._reverse_bit_index = []
        for i in range(self._M):
            self._reverse_bit_index.append(BitReverser._reverse_bits(i, self._n1))

    def _reverse_bits(value, num_bits):
        result = 0
        for i in range(num_bits):
            if value & (1 << i):
                result |= 1 << (num_bits - 1 - i)
        return result

    def reverse_bits(self, value):
        return self._reverse_bit_index[value]
