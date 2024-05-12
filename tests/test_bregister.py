""" test for bregister module"""
import math

import crsq.slater.bregister as br
import crsq.slater.aregister as ar
import crsq.arithmetic.test_tools as test_tools


def test__min_bits_to_hold():
    """ min_bits_to_hold """
    assert br._min_bits_to_hold(1) == 1
    assert br._min_bits_to_hold(2) == 2
    assert br._min_bits_to_hold(3) == 2
    assert br._min_bits_to_hold(4) == 3
    for i in range(1, 16):
        n = br._min_bits_to_hold(i)
        t = 2 ** n - 1
        assert t >= i
        assert t < i * 2


def test__make_cond_bits_for_permutation():
    """ _make_cond_bits_for_permutation """
    assert br._make_cond_bits_for_permutation(4, 1) == (3, '001')
    assert br._make_cond_bits_for_permutation(6, 1) == (3, '001')
    assert br._make_cond_bits_for_permutation(8, 1) == (4, '0001')
    assert br._make_cond_bits_for_permutation(8, 7) == (4, '0111')


def _log2(x):
    s = 0
    t = 1
    while x > t:
        s += 1
        t *= 2
    if x == 1:
        nb = 1
    elif t == x:
        nb = s
    else:
        nb = s + 1
    return (s, t, nb)


def _create_permutations(numbers: list[int]) -> list[tuple[list[int], int]]:
    n = len(numbers)
    if n == 1:
        return [(numbers, 1)]
    result = []
    for k in range(n):
        head = numbers[k:k+1]
        if k == 0:
            my_sign = 1
            reduced = numbers[1:]
        else:
            my_sign = -1
            a = numbers[0:1]
            b = numbers[1:k]
            c = numbers[k+1:]
            reduced = b + a + c
        reduced_permutations = _create_permutations(reduced)
        for y, s in reduced_permutations:
            x = head + y
            sign = my_sign * s
            result.append((x, sign))
    return result


def test_build_permutations_without_areg():
    """ Test the permutation part only, without the
        wave function swapping.
    """
    print("test_build_permutations_without_areg")
    MAX_N=5
    # bregs are allocated only when there are more than 2 electrons.
    for N in range(2, MAX_N+1):
        print(N)
        _s, _t, nb = _log2(N)
        # prepare the expected statevector dict
        NN = math.factorial(N)
        available = list(reversed(range(N)))
        permutations = _create_permutations(available)
        assert len(permutations) == NN
        p = 1/math.sqrt(NN)
        expected = []
        ancilla_bit = 0
        for int_list, sign in permutations:
            regs_dict = {}
            for i in range(N):
                regs_dict[f"b{i}"] = int_list[N-1-i]
            regs_dict["shuffle"] = ancilla_bit
            expected.append( { "regs": regs_dict, "amp": p })
        # build the circuit
        for use_gates in [False, True]:
            breg = br.BRegister(nb, N, 1)
            breg.set_use_custom_gates(use_gates)
            breg.set_should_swap_aregs(False)
            breg_frame = breg.allocate_registers()
            qc = breg_frame.circuit
            breg.build_sums(breg_frame)
            breg.build_permutations(breg_frame)
            qc.save_statevector()
            test_tools.run_circuit_and_check(qc, expected)


def test_build_permutations_with_areg():
    """ test permutations and shuffling together"""
    print("test_build_permutations_with_areg")
    MAX_N=4  # 4 is maximum we can run on AerSimulator.
    for N in range(2, MAX_N+1):
        print(N)
        _s, _t, nb = _log2(N)
        # prepare the expected statevector dict
        NN = math.factorial(N)
        available = list(reversed(range(N)))
        permutations = _create_permutations(available)
        assert len(permutations) == NN
        p = 1/math.sqrt(NN)
        expected = []
        ancilla_bit = 0
        for int_list, sign in permutations:
            regs_dict = {}
            for i in range(N):
                regs_dict[f"b{i}"] = int_list[N-1-i]
                regs_dict[f"e{i}x"] = int_list[N-1-i]
                regs_dict[f"e{i}s"] = int_list[N-1-i] % 2
            if N > 2:
                regs_dict["shuffle"] = ancilla_bit
            expected.append( { "regs": regs_dict, "amp": p * sign})
        # print("expected: ", expected)
        # build the circuit
        for use_gates in [False, True]:
            # without custom gates
            m = nb
            areg = ar.ARegister(m, N, 1)
            breg = br.BRegister(nb, N, 1)
            breg.set_areg(areg)
            breg.set_use_custom_gates(use_gates)
            breg_frame = breg.allocate_registers()
            areg_frame = breg_frame.areg_frame
            qc = breg_frame.circuit
            areg.set_test_values(areg_frame)

            breg.build_sums(breg_frame)
            breg.build_permutations(breg_frame)
            qc.save_statevector()
            test_tools.run_circuit_and_check(qc, expected)


if __name__ == '__main__':
    test__min_bits_to_hold()
    test__make_cond_bits_for_permutation()
    test_build_permutations_without_areg()
    test_build_permutations_with_areg()
