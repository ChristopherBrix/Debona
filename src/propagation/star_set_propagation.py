"""
Propagation using Star Sets.
"""

from typing import List, Optional

import numpy as np

from src.propagation.abstract_domain_propagation import AbstractDomainPropagation


# TODO
class StarSetPropagation(AbstractDomainPropagation):

    """Class that implements the StarSet algorithm"""

    # TODO: which technique do we use?
    def largest_error_split_node(
        self, output_weights: np.ndarray = None
    ) -> Optional[tuple]:
        pass


# TODO
class StarSetForwardPropagation(StarSetPropagation):
    """Class that implements the StarSet forward propagation algorithm"""

    # TODO
    def calc_bounds(
        self,
        input_constraints: np.ndarray,
        forced_input_bounds: List[np.ndarray],
        from_layer: int = 1,
    ) -> bool:
        pass


class StarSetBackwardPropagation(StarSetPropagation):
    """Class that implements the StarSet backward propagation algorithm"""

    # TODO
    def calc_bounds(
        self,
        input_constraints: np.ndarray,
        forced_input_bounds: List[np.ndarray],
        from_layer: int = 1,
    ) -> bool:
        pass
