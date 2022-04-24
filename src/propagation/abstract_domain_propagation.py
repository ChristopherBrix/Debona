"""
AbstractDomainPropagation is the basis for the concrete propagation techniques.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import gurobipy as grb
import numpy as np

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
    def get_next_split_node(self, output_weights: np.ndarray = None) -> Optional[tuple]:
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

    @abstractmethod
    def get_final_eq(self, potential_counter, correct_class) -> np.ndarray:
        """
        Computes the final eq used for the LP solver

        Args:
            potential_counter : The potential counter example class
            correct_class     : The correct class

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
