""" Time evolution block.
    (1) Suzuki-Trotter decomposition
"""
from crsq.blocks.time_evolution.spec import TimeEvolutionSpec
from crsq.blocks.time_evolution.suzuki_trotter import ElectronMotionBlock, NucleusMotionBlock
from crsq.blocks.time_evolution.time_evolution import TimeEvolutionBlock
