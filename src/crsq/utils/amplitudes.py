""" Functions to work with complex number amplitudes
"""
import cmath
import math
import fractions

eps = 1e-6
ieps = 1e+6

def _phase_str(th):
    thp = th/math.pi
    thp = math.floor(thp*ieps + 0.5)*eps
    thp_n, thp_d = fractions.Fraction(thp).as_integer_ratio()
    if thp_n == 0:
        phase = ("", "")
    elif thp_n == 1:
        if thp_d == 1:
            phase = ("-", "")
        else:
            phase = ("", f"exp(jπ/{thp_d})")
    elif thp_n == -1:
        if thp_d == 1:
            phase = ("-", "")
        else:
            phase = ("", f"exp(-jπ/{thp_d})")
    else:
        phase = ("", f"exp({thp_n}jπ/{thp_d})")
    return phase

def _amp_str(r):
    # we round the inverse of v
    q = 1/(r*r)
    q = math.floor(q*(2**6)+0.5)*(2**-6)
    f = fractions.Fraction(q)
    # convert back from the inverse
    denom, num = f.as_integer_ratio()
    if denom == 1:
        if num == 1:
            amp = ("1", "")
        else:
            amp = ("", f"\u221a({num})")
    else:
        if num == 1:
            amp = ("1", f"/\u221a({denom})")
        else:
            amp = ("", f"\u221a({num}/{denom})")
    return amp


def format_in_polar_fractional(z: complex) -> str:
    """ Format a complex number in polar notation

        output examples:

        1
            just for 1.0
        1/\u221a(3)
            real values are treated as a square root of some fraction
        -1/\u221a(2)
            when the phase is \pi, it is treated as a minus.
        exp(jπ/2)/\u221a(2)
            for other phases, it is treated as \pi multiplied by some fraction

        :param z: a complex number
    """
    r = abs(z)
    th = cmath.phase(z)
    sign, phase = _phase_str(th)
    one, frac = _amp_str(r)
    if phase == "":
        return sign + one + frac
    else:
        return sign + phase + frac


def format_in_polar(z: complex) -> str:
    """ Format a complex number in polar notation

        output examples:

        1
            just for 1.0
        1.5*exp(0.5j)
            for anything that has a phase

        :param z: a complex number
    """
    r = abs(z)
    th = cmath.phase(z)
    if abs(th) > eps:
        phase = f"*exp({th:.6f}j)"
    else:
        phase=""
    return f"{r:.6f}{phase}"

