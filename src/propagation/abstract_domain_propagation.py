"""
AbstractDomainPropagation is the basis for the concrete propagation techniques.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from src.domains.abstract_domains import AbstractDomain


class AbstractDomainPropagation(ABC):

    """
    Abstract class for the propagation of the domain
    """

    def __init__(self, domain: AbstractDomain):
        """
        Args:

            domain: The domain of the propagation method
        """
        self._domain = domain

    @property
    def domain(self):
        return self._domain

    @abstractmethod
    def calc_bounds(self, input_constraints: np.ndarray, from_layer: int = 1) -> bool:

        """
        Calculate the bounds for all layers in the network starting at from_layer.

        Notice that from_layer is usually larger than 1 after a split. In this case, the
        split constraints are added to the layer before from_layer by adjusting the
        forced bounds. For this reason, we update the concrete_bounds for the layer
        before from_layer.

        Args:
            input_constraints       : The constraints on the input. The first dimensions
                                      should be the same as the input to the neural
                                      network, the last dimension should contain the
                                      lower bound on axis 0 and the upper on axis 1.
            from_layer              : Updates this layer and all later layers

        Returns:
            True if the method succeeds, False if the bounds are invalid. The bounds are
            invalid if the forced bounds make at least one upper bound smaller than a
            lower bound.
        """

    @abstractmethod
    def merge_current_bounds_into_forced(self):
        """
        Sets forced input bounds to the best of current forced bounds and calculated
        bounds.
        """

    @abstractmethod
    def largest_error_split_node(
        self, output_weights: np.ndarray = None
    ) -> Optional[tuple]:
        """
        Returns the node with the largest weighted error effect on the output

        The error from overestimation is calculated for each output node with respect to
        each hidden node. This value is weighted using the given output_weights and the
        index of the node with largest effect on the output is returned.

        Args:
            output_weights  : A Nx2 array with the weights for the lower bounds in
                              column 1 and the upper bounds in column 2. All weights
                              should be >= 0.

        Returns:
              (layer_num, node_num) of the node with largest error effect on the output
        """
