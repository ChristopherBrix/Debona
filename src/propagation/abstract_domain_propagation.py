"""
AbstractDomainPropagation is the basis for the concrete propagation techniques.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Optional

import gurobipy as grb
import numpy as np

from src.algorithm.heuristic import Heuristic
from src.algorithm.task_constants import TaskConstants
from src.domains.abstract_domains import AbstractDomain


class AbstractDomainPropagation(ABC):

    """
    Abstract class for the propagation of the domain
    """

    def __init__(self, task_constants: TaskConstants, domain: AbstractDomain):
        """
        Args:
            task_constants: The constant parameters of this task
            domain: The domain of the propagation method
        """
        self._task_constants = task_constants
        self._domain = domain
        self._overapproximated_neurons: List[Optional[np.ndarray]] = [
            None
        ] * task_constants.num_layers
        self._overapproximated_neurons[0] = np.zeros((0, 2), dtype=int)

    @property
    def domain(self):
        return self._domain

    @property
    def overapproximated_neurons(self) -> List[Optional[np.ndarray]]:
        """
        A list of N'x2 mappings where the first column contains the layer and the
        second column contains the node-number.
        """

        assert self._overapproximated_neurons is not None
        return self._overapproximated_neurons

    @abstractmethod
    def calc_bounds(
        self,
        input_constraints: np.ndarray,
        forced_input_bounds: List[np.ndarray],
        from_layer: int = 1,
    ) -> bool:

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
            forced_input_bounds     : Known input bounds that must be enforced
            from_layer              : Updates this layer and all later layers

        Returns:
            True if the method succeeds, False if the bounds are invalid. The bounds are
            invalid if the forced bounds make at least one upper bound smaller than a
            lower bound.
        """

    @abstractmethod
    def get_heuristic_ranking(self, output_weights: np.ndarray = None) -> List[tuple]:
        """
        Returns the order of nodes for splitting

        The error from overestimation is calculated for each output node with respect to
        each hidden node. This value is weighted using the given output_weights and the
        index of the node with largest effect on the output is returned.

        Args:
            output_weights  : A Nx2 array with the weights for the lower bounds in
                              column 1 and the upper bounds in column 2. All weights
                              should be >= 0.

        Returns:
              List[(layer_num, node_num)] ordered by the largest error effect on the
                output
        """

    def get_next_split_node(self, output_weights: np.ndarray = None) -> Optional[tuple]:
        # This is ugly, but importing the config at the toplevel would lead to a
        # circular import
        from src.util import config  # pylint: disable=import-outside-toplevel

        if config.HEURISTIC.value == Heuristic.DYNAMIC.value:
            heuristic_ranking = self.get_heuristic_ranking(output_weights)
        elif config.HEURISTIC.value == Heuristic.CACHED.value:
            if self._task_constants.cached_heuristic_ranking is None:
                self._task_constants.cached_heuristic_ranking = (
                    self.get_heuristic_ranking(output_weights)
                )
            heuristic_ranking = self._task_constants.cached_heuristic_ranking
        else:
            assert config.HEURISTIC.value == Heuristic.ARCHITECTURE.value
            heuristic_ranking = []
            for layer, layer_size in enumerate(self._task_constants.layer_sizes):
                mapping = self._task_constants.mappings[layer]
                if mapping is None or mapping.is_linear:
                    continue
                for neuron in reversed(range(layer_size)):
                    heuristic_ranking.append((layer, neuron))
            heuristic_ranking = heuristic_ranking[::-1]

        heuristic_ranking = deepcopy(heuristic_ranking)
        assert self.overapproximated_neurons[-1] is not None
        while len(heuristic_ranking) > 0:
            layer, neuron = heuristic_ranking.pop()
            if [layer, neuron] in self.overapproximated_neurons[-1].tolist():
                return (layer, neuron)
        if len(self.overapproximated_neurons[-1]) > 0:
            return self.overapproximated_neurons[-1][-1]
        return None

    @abstractmethod
    def get_final_eq(self, weights: np.ndarray, bias: float = 0.0) -> np.ndarray:
        """
        Computes the final eq used for the LP solver

        Args:
            weights: How each output neuron should be weighted. Correct class < 0
            bias: Additional bias

        Returns:
            The equation putting the potential counter example and the correct class
            into relation.
        """

    @abstractmethod
    def get_grb_constr(
        self, layer_num: int, node: int, split_x: float, upper: bool, input_vars: list
    ) -> grb.TempConstr:
        """
        Computes the constraint that should be added to the LP.

        Args:
            layer_num:  The layer number of the node
            node:       The node index
            split_x:    For what value the node is split
            upper:      Whether this is the upper branch
            input_vars: List of Gurobi input variables
        """

    @abstractmethod
    def _compute_symbolic_bounds(self, layer_num: int):

        """
        Calculates the symbolic bounds.

        This updates all bounds, relaxations and errors for the given layer, by
        propagating from the previous layer.

        Args:
            layer_num: The layer number
        """

    @abstractmethod
    def _compute_concrete_bounds(
        self, layer_num: int, forced_input_bounds: List[np.ndarray]
    ):
        """
        Calculate the concrete bounds

        Args:
            layer_num           : For which layer to compute the concrete bounds
            forced_input_bounds : Known input bounds that must be enforced
        """

    @abstractmethod
    def _relax_relu(
        self,
        lower_bounds_concrete_in: np.ndarray,
        upper_bounds_concrete_in: np.ndarray,
        upper: bool,
    ) -> np.ndarray:

        """
        Calculates the linear relaxation of a ReLU operation

        The linear relaxation is a Nx2 array, where each row represents a and b of the
        linear equation: l(x) = ax + b

        The relaxations are described in detail in the paper.

        Args:
            lower_bounds_concrete_in    : The concrete lower bounds of the input to the
                                          nodes
            upper_bounds_concrete_in    : The concrete upper bounds of the input to the
                                          nodes
            upper                       : If true, the upper relaxation is calculated,
                                          else the lower

        Returns:
            The relaxations as a Nx2 array
        """
