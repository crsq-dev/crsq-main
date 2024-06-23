""" Time evolution block.
    (1) Suzuki-Trotter decomposition
"""
from crsq.blocks.time_evolution.spec import TimeEvolutionSpec, SUZUKI_TROTTER
from crsq.blocks.time_evolution.suzuki_trotter import ElectronMotionBlock, NucleusMotionBlock, SuzukiTrotterMethodBlock
