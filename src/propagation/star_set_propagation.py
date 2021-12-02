from src.propagation.abstract_domain_propagation import AbstractDomainPropagation
import numpy as np
from typing import Optional

# TODO
class StarSetPropagation(AbstractDomainPropagation):
    def calc_bounds(self, input_constraints: np.array, from_layer: int = 1) -> bool:
        raise NotImplementedError(
            f"calc_bounds(...) not implemented in {self.__name__}"
        )

    # TODO: for star sets should not be very different from deeppoly
    def merge_current_bounds_into_forced(self):
        pass

    # TODO: which technique do we use?
    def largest_error_split_node(
        self, output_weights: np.array = None
    ) -> Optional[tuple]:
        pass


# TODO
class StarSetForwardPropagation(StarSetPropagation):
    # TODO
    def calc_bounds(self, input_constraints: np.array, from_layer: int = 1) -> bool:
        pass


class StarSetBackwardPropagation(StarSetPropagation):
    # TODO
    def calc_bounds(self, input_constraints: np.array, from_layer: int = 1) -> bool:
        pass
