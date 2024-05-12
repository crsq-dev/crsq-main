""" Discretization
"""

import logging

logger = logging.getLogger(__name__)

class DiscretizationSpec:
    """ Discretization parameter holder
    """

    def __init__(self, delta_t):
        self._delta_t = delta_t
        logger.info("DiscretizationSpec: delta_t = %f", delta_t)

    @property
    def delta_t(self) -> float:
        """ dt """
        return self._delta_t
