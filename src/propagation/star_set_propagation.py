"""
Propagation using Star Sets.
"""

from typing import Optional

import numpy as np

from src.propagation.abstract_domain_propagation import AbstractDomainPropagation


# TODO
class StarSetPropagation(AbstractDomainPropagation):

    """Class that implements the StarSet algorithm"""

    def calc_bounds(self, input_constraints: np.ndarray, from_layer: int = 1) -> bool:
        raise NotImplementedError(
            f"calc_bounds(...) not implemented in {self.__class__.__name__}"
        )

    # TODO: for star sets should not be very different from deeppoly
    def merge_current_bounds_into_forced(self):
        pass

    # TODO: which technique do we use?
    def largest_error_split_node(
        self, output_weights: np.ndarray = None
    ) -> Optional[tuple]:
        pass


# TODO
class StarSetForwardPropagation(StarSetPropagation):
    """Class that implements the StarSet forward propagation algorithm"""

    # TODO
    def calc_bounds(self, input_constraints: np.ndarray, from_layer: int = 1) -> bool:
        pass


class StarSetBackwardPropagation(StarSetPropagation):
    """Class that implements the StarSet backward propagation algorithm"""

    # TODO
    def calc_bounds(self, input_constraints: np.ndarray, from_layer: int = 1) -> bool:
        pass
