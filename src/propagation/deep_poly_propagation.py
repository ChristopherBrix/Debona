"""
Propagation using DeepPoly.
"""

from typing import List

import gurobipy as grb
import numpy as np

from src.algorithm.esip_util import concretise_symbolic_bounds_jit, sum_error_jit
from src.algorithm.mappings.piecewise_linear import Relu
from src.algorithm.task_constants import TaskConstants
from src.domains.deep_poly import DeepPoly
from src.propagation.abstract_domain_propagation import AbstractDomainPropagation


class DeepPolyPropagation(AbstractDomainPropagation):
    """Class that implements the DeepPoly algorithm"""

    def __init__(self, task_constants: TaskConstants):
        """
        Args:
            task_constants: The constant parameters of this task
        """
        super().__init__(task_constants, DeepPoly(task_constants))
        self.node_impact: List[np.ndarray] = []

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
                                      should be the same as the
                                      input to the neural network, the last dimension
                                      should contain the lower bound on axis 0 and the
                                      upper on axis 1.
            forced_input_bounds     : Known input bounds that must be enforced
            from_layer              : Updates this layer and all later layers

        Returns:
            True if the method succeeds, False if the bounds are invalid. The bounds are
            invalid if the forced bounds make at least one upper bound smaller than a
            lower bound.
        """

        assert from_layer >= 1, "From layer should be >= 1"
        assert isinstance(
            input_constraints, np.ndarray
        ), "input_constraints should be a np array"

        self._domain.bounds_concrete[0] = input_constraints

        # Concrete bounds from previous layer might have to be recalculated due to new
        # split-constraints
        if from_layer > 1:
            self._compute_concrete_bounds(from_layer - 1, forced_input_bounds)

        for i in range(self._task_constants.num_layers):
            self.node_impact.append(np.zeros(self._task_constants.layer_sizes[i]))

        for layer_num in range(from_layer, self._task_constants.num_layers):
            self._compute_symbolic_bounds(layer_num)
            self._compute_concrete_bounds(layer_num, forced_input_bounds)
            success = self._valid_concrete_bounds(
                self._domain.bounds_concrete[layer_num]
            )
            if not success:
                return False
            self._compute_overapproximated_neurons(layer_num)

        return True

    def _compute_overapproximated_neurons(self, layer_num: int):
        """
        Computes the list of overapproximated neurons

        Args:
            layer_num: The layer number
        """

        mapping = self._task_constants.mappings[layer_num]
        if mapping.is_linear:
            self.overapproximated_neurons[layer_num] = self.overapproximated_neurons[
                layer_num - 1
            ].copy()
        else:
            relaxations = self.domain.relaxations[layer_num]
            err_idx = np.argwhere(
                (relaxations[1, :, 0] != 1) * (relaxations[1, :, 0] != 0)
            )[:, 0]
            num_err = err_idx.shape[0]

            overapproximated_neurons = self.overapproximated_neurons[layer_num - 1]
            if num_err > 0:
                overapproximated_neurons_new = np.hstack(
                    (
                        np.zeros(err_idx.shape, dtype=int)[:, np.newaxis] + layer_num,
                        err_idx[:, np.newaxis],
                    )
                )
                overapproximated_neurons_new = np.vstack(
                    (overapproximated_neurons, overapproximated_neurons_new)
                )
            else:
                overapproximated_neurons_new = overapproximated_neurons.copy()

            self.overapproximated_neurons[layer_num] = overapproximated_neurons_new

    @staticmethod
    def _adjust_bounds_from_forced_bounds(
        bounds_concrete: np.ndarray, forced_input_bounds: np.ndarray
    ) -> np.ndarray:

        """
        Adjusts the concrete input bounds using the forced bounds.

        The method chooses the best bound from the stored concrete input bounds and the
        forced bounds as the new concrete input bound.

        Args:
            bounds_concrete     : A Nx2 array with the concrete lower and upper bounds,
                                  where N is the number of Nodes.
            forced_input_bounds : A Nx2 array with the forced input bounds used for
                                  adjustment, where N is the number of Nodes.

        Returns:
            A Nx2 array with the adjusted concrete bounds, where N is the number of
            Nodes.
        """

        bounds_concrete_new = bounds_concrete.copy()

        if forced_input_bounds is None:
            return bounds_concrete_new

        forced_lower = forced_input_bounds[:, 0:1]
        forced_upper = forced_input_bounds[:, 1:2]

        smaller_idx = bounds_concrete <= forced_lower
        bounds_concrete_new[smaller_idx] = np.hstack((forced_lower, forced_lower))[
            smaller_idx
        ]

        larger_idx = bounds_concrete >= forced_upper
        bounds_concrete_new[larger_idx] = np.hstack((forced_upper, forced_upper))[
            larger_idx
        ]

        return bounds_concrete_new

    @staticmethod
    def _valid_concrete_bounds(bounds_concrete: np.ndarray) -> bool:

        """
        Checks that all lower bounds are smaller than their respective upper bounds.

        Args:
            bounds_concrete : A Nx2 array with the concrete lower and upper input bounds

        Returns:
            True if the bounds are valid.
        """

        return not (bounds_concrete[:, 1] < bounds_concrete[:, 0]).sum() > 0

    def _calc_relaxations(self, layer_num: int):

        """
        Calculates the linear relaxations for the given mapping and concrete bounds.

        Args:
            layer_num: Layer number
        """

        mapping = self._task_constants.mappings[layer_num]
        bounds_concrete: np.ndarray = self._domain.bounds_concrete[layer_num - 1]
        assert isinstance(mapping, Relu), (
            "To support other mappings than ReLU, include their relaxations in this"
            "class (or restructure the code to something more modular, but the"
            "relaxation must depend on the chosen propagation technique"
        )
        lower_relaxation = self._relax_relu(
            bounds_concrete[:, 0], bounds_concrete[:, 1], False
        )
        upper_relaxation = self._relax_relu(
            bounds_concrete[:, 0], bounds_concrete[:, 1], True
        )

        self._domain.relaxations[layer_num] = np.stack(
            [lower_relaxation, upper_relaxation], axis=0
        )


class DeepPolyForwardPropagation(DeepPolyPropagation):
    """Class that implements the DeepPoly forward propagation algorithm"""

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
        bounds_symbolic = self.domain.bounds_symbolic[-1]
        eq: np.ndarray = np.sum(bounds_symbolic * weights[:, np.newaxis], axis=0)
        eq[-1] += bias

        error = np.sum(self.domain.error_matrix[-1] * weights[:, np.newaxis], axis=0)

        eq[-1] += np.sum(error[error > 0])

        return eq

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

        if self._domain.error_matrix[-1].shape[1] == 0:
            return []

        output_weights = (
            np.ones((self._domain.layer_sizes[-1], 2))
            if output_weights is None
            else output_weights
        )
        output_weights[output_weights <= 0] = 0.01

        err_matrix_neg = self._domain.error_matrix[-1].copy()
        err_matrix_neg[err_matrix_neg > 0] = 0
        err_matrix_pos = self._domain.error_matrix[-1].copy()
        err_matrix_pos[err_matrix_pos < 0] = 0

        err_matrix_neg = -err_matrix_neg * output_weights[:, 0:1]
        err_matrix_pos = err_matrix_pos * output_weights[:, 1:2]

        weighted_error = (err_matrix_neg + err_matrix_pos).sum(axis=0)
        assert len(weighted_error) == len(self.overapproximated_neurons[-1])

        return [
            x for _, x in sorted(zip(weighted_error, self.overapproximated_neurons[-1]))
        ]

    def _compute_symbolic_bounds(self, layer_num: int):

        """
        Calculates the symbolic bounds.

        This updates all bounds, relaxations and errors for the given layer, by
        propagating from the previous layer.

        Args:
            layer_num: The layer number
        """

        mapping = self._task_constants.mappings[layer_num]

        if mapping.is_linear:
            self._domain.bounds_symbolic[layer_num] = mapping.propagate(
                self._domain.bounds_symbolic[layer_num - 1], add_bias=True
            )
            self._domain.error_matrix[layer_num] = mapping.propagate(
                self._domain.error_matrix[layer_num - 1], add_bias=False
            )

        else:
            self._calc_relaxations(layer_num)

            self._domain.bounds_symbolic[
                layer_num
            ] = self._prop_equation_through_relaxation(
                self._domain.bounds_symbolic[layer_num - 1],
                self._domain.relaxations[layer_num],
            )

            self._prop_error_matrix_through_relaxation(layer_num)

    def _compute_concrete_bounds(
        self, layer_num: int, forced_input_bounds: List[np.ndarray]
    ):
        """
        Calculate the concrete bounds

        Args:
            layer_num           : For which layer to compute the concrete bounds
            forced_input_bounds : Known input bounds that must be enforced
        """
        mapping = self._task_constants.mappings[layer_num]

        if mapping.is_1d_to_1d:
            self._domain.bounds_concrete[layer_num] = mapping.propagate(
                self._domain.bounds_concrete[layer_num - 1]
            )

        else:
            concrete_bounds = concretise_symbolic_bounds_jit(
                self._domain.bounds_concrete[0], self._domain.bounds_symbolic[layer_num]
            )
            errors = sum_error_jit(self._domain.error_matrix[layer_num])
            concrete_bounds += errors

            self._domain.bounds_concrete[layer_num] = concrete_bounds
            self._domain.error[layer_num] = errors

        self._domain.bounds_concrete[
            layer_num
        ] = self._adjust_bounds_from_forced_bounds(
            self._domain.bounds_concrete[layer_num],
            forced_input_bounds[layer_num],
        )

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
        symb_input_bounds = self.domain.bounds_symbolic[layer_num - 1][node]

        if upper:
            return (
                grb.LinExpr(symb_input_bounds[:-1], input_vars)
                + self.domain.error[layer_num - 1][node][1]
                + symb_input_bounds[-1]
                >= split_x
            )

        else:
            return (
                grb.LinExpr(symb_input_bounds[:-1], input_vars)
                + self.domain.error[layer_num - 1][node][0]
                + symb_input_bounds[-1]
                <= split_x
            )

    @staticmethod
    def _prop_equation_through_relaxation(
        bounds_symbolic: np.ndarray, relaxations: np.ndarray
    ) -> np.ndarray:

        """
        Propagates the given symbolic equations through the lower linear relaxations.

        Args:
            bounds_symbolic : A Nx(M+1) array with the symbolic bounds, where N is the
                              number of nodes in the layer and M is the number of input
            relaxations     : A 2xNx2 array where the first dimension indicates the
                              lower and upper relaxation, the second dimension contains
                              the nodes in the current layer and the last dimension
                              contains the parameters [a, b] in l(x) = ax + b.

        Returns:
            A Nx(M+1) with the new symbolic bounds.
        """

        bounds_symbolic_new = bounds_symbolic * relaxations[0, :, 0:1]
        bounds_symbolic_new[:, -1] += relaxations[0, :, 1]

        return bounds_symbolic_new

    def _prop_error_matrix_through_relaxation(self, layer_num: int):
        """
        Updates the error matrix.

        The old errors are propagated through the lower relaxations and the new errors
        due to the relaxations at this layer are concatenated the result.

        Args:
            layer_num: The number of the current layer, used to update the error matrix
        """

        error_matrix: np.ndarray = self._domain.error_matrix[layer_num - 1]
        relaxations: np.ndarray = self._domain.relaxations[layer_num]
        bounds_concrete: np.ndarray = self._domain.bounds_concrete[layer_num - 1]

        # Get the relaxation parameters
        a_low, a_up = relaxations[0, :, 0], relaxations[1, :, 0]
        b_low, b_up = relaxations[0, :, 1], relaxations[1, :, 1]

        # Calculate the error at the lower and upper input bound.
        error_lower = (bounds_concrete[:, 0] * a_up + b_up) - (
            bounds_concrete[:, 0] * a_low + b_low
        )
        error_upper = (bounds_concrete[:, 1] * a_up + b_up) - (
            bounds_concrete[:, 1] * a_low + b_low
        )

        # Add the errors introduced by the linear relaxations
        max_err = np.max((error_lower, error_upper), axis=0)
        err_idx = np.argwhere(max_err != 0)[:, 0]
        num_err = err_idx.shape[0]

        # Create error_matrix and propagate old errors through the relaxations
        layer_size = error_matrix.shape[0]
        num_old_err = error_matrix.shape[1]
        error_matrix_new = np.empty((layer_size, num_old_err + num_err), np.float32)

        if num_old_err > 0:
            error_matrix_new[:, :num_old_err] = a_low[:, np.newaxis] * error_matrix

        # Calculate the new errors.
        if num_err > 0:
            error_matrix_new[:, num_old_err:] = 0
            error_matrix_new[:, num_old_err:][err_idx, np.arange(num_err)] = max_err[
                err_idx
            ]

        self._domain.error_matrix[layer_num] = error_matrix_new

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

        layer_size = lower_bounds_concrete_in.shape[0]
        relaxations = np.zeros((layer_size, 2))

        # Operating in the positive area
        fixed_upper_idx = np.argwhere(lower_bounds_concrete_in >= 0)
        relaxations[fixed_upper_idx, 0] = 1
        relaxations[fixed_upper_idx, 1] = 0

        # Operating in the negative area
        fixed_lower_idx = np.argwhere(upper_bounds_concrete_in <= 0)
        relaxations[fixed_lower_idx, :] = 0

        # Operating in the non-linear area
        mixed_idx = np.argwhere(
            (upper_bounds_concrete_in > 0) * (lower_bounds_concrete_in < 0)
        )

        if len(mixed_idx) == 0:
            return relaxations

        xl = lower_bounds_concrete_in[mixed_idx]
        xu = upper_bounds_concrete_in[mixed_idx]

        if upper:
            a = xu / (xu - xl)
            b = -a * xl
            relaxations[:, 0][mixed_idx] = a
            relaxations[:, 1][mixed_idx] = b
        else:
            relaxations[:, 0][mixed_idx] = xu / (xu - xl)
            relaxations[:, 1][mixed_idx] = 0

        # Outward rounding
        num_ops = 2
        max_edge_dist = np.hstack((np.abs(xl), np.abs(xu))).max(axis=1)[:, np.newaxis]
        max_err = np.spacing(
            np.abs(relaxations[:, 0][mixed_idx])
        ) * max_edge_dist + np.spacing(np.abs(relaxations[:, 1][mixed_idx]))
        outward_round = max_err * num_ops if upper else -max_err * num_ops
        relaxations[:, 1][mixed_idx] += outward_round

        return relaxations


class DeepPolyBackwardPropagation(DeepPolyPropagation):
    """Class that implements the DeepPoly backward propagation algorithm"""

    def _get_symbolic_equation(
        self, layer_num: int, pseudo_last_layer: np.ndarray = None
    ) -> np.ndarray:
        if layer_num == -1:
            layer_num = self._task_constants.num_layers - 1
        assert len(self._task_constants.mappings) == self._task_constants.num_layers
        assert layer_num > 0

        if pseudo_last_layer is not None:
            assert (
                layer_num == self._task_constants.num_layers
            ), f"{layer_num} != {self._task_constants.num_layers}"
            symbolic_bounds = pseudo_last_layer
        else:
            symbolic_bounds = np.tile(
                np.eye(
                    self._task_constants.layer_sizes[layer_num],
                    self._task_constants.layer_sizes[layer_num] + 1,
                ),
                reps=(2, 1, 1),
            )
        for current_layer_num in range(
            min(layer_num, self._task_constants.num_layers - 1), 0, -1
        ):
            if pseudo_last_layer is not None:
                self.node_impact[current_layer_num] += np.maximum(
                    0, symbolic_bounds[1, 0, :-1]
                )
            mapping = self._task_constants.mappings[current_layer_num]
            if mapping.is_linear:
                new_symbolic_bounds = mapping.propagate_back(symbolic_bounds)
            else:
                assert (
                    self._domain.relaxations[current_layer_num] is not None
                ), current_layer_num
                relaxations = self._domain.relaxations[current_layer_num]
                pos_symbolic_bounds = np.where(symbolic_bounds > 0, symbolic_bounds, 0)
                neg_symbolic_bounds = np.where(symbolic_bounds < 0, symbolic_bounds, 0)
                new_symbolic_bounds = np.empty_like(symbolic_bounds)
                new_symbolic_bounds[:, :, :-1] = (
                    pos_symbolic_bounds[:, :, :-1]
                    * relaxations[:, :, 0][:, np.newaxis, :]
                )
                new_symbolic_bounds[0, :, :-1] += (
                    neg_symbolic_bounds[0, :, :-1] * relaxations[1, :, 0][np.newaxis, :]
                )
                new_symbolic_bounds[1, :, :-1] += (
                    neg_symbolic_bounds[1, :, :-1] * relaxations[0, :, 0][np.newaxis, :]
                )
                added_relaxation_bias = np.sum(
                    (
                        pos_symbolic_bounds[:, :, :-1]
                        * relaxations[:, :, 1][:, np.newaxis, :]
                    ),
                    axis=2,
                )
                added_relaxation_bias[0, :] += np.sum(
                    (
                        neg_symbolic_bounds[0, :, :-1]
                        * relaxations[1, :, 1][np.newaxis, :]
                    ),
                    axis=1,
                )
                added_relaxation_bias[1, :] += np.sum(
                    (
                        neg_symbolic_bounds[1, :, :-1]
                        * relaxations[0, :, 1][np.newaxis, :]
                    ),
                    axis=1,
                )
                new_symbolic_bounds[:, :, -1] = (
                    symbolic_bounds[:, :, -1] + added_relaxation_bias
                )
            symbolic_bounds = new_symbolic_bounds

        return symbolic_bounds

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

        pseudo_last_layer = np.zeros((2, 1, self._task_constants.layer_sizes[-1] + 1))
        pseudo_last_layer[:, 0, :-1] = weights
        pseudo_last_layer[:, 0, -1] = bias

        bounds_symbolic = self._get_symbolic_equation(
            self._task_constants.num_layers, pseudo_last_layer
        )
        eq = bounds_symbolic[1, 0]

        return eq

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

        largest_layer_num = np.max(self.overapproximated_neurons[-1][:, 0])
        heuristic_ranking: List[tuple] = []
        for layer_num in reversed(range(largest_layer_num + 1)):
            for i in np.argsort(self.node_impact[layer_num]):
                heuristic_ranking.append((layer_num, i))
        return heuristic_ranking

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
        if upper:
            symb_input_bounds = self.domain.bounds_symbolic[layer_num - 1][1, node]
            return (
                grb.LinExpr(symb_input_bounds[:-1], input_vars) + symb_input_bounds[-1]
                >= split_x
            )

        else:
            symb_input_bounds = self.domain.bounds_symbolic[layer_num - 1][0, node]
            return (
                grb.LinExpr(symb_input_bounds[:-1], input_vars) + symb_input_bounds[-1]
                <= split_x
            )

    def _compute_symbolic_bounds(self, layer_num: int):

        """
        Calculates the symbolic bounds.

        This updates all bounds, relaxations and errors for the given layer, by
        propagating from the previous layer.

        Args:
            layer_num: The layer number
        """

        assert layer_num > 0

        mapping = self._task_constants.mappings[layer_num]

        if mapping.is_linear:
            self.overapproximated_neurons[layer_num] = self.overapproximated_neurons[
                layer_num - 1
            ].copy()
        else:
            self._calc_relaxations(layer_num)

        self._domain.bounds_symbolic[layer_num] = self._get_symbolic_equation(layer_num)
        self._domain.error_matrix[layer_num] = np.zeros((0, 0))

        return True

    def _compute_concrete_bounds(
        self, layer_num: int, forced_input_bounds: List[np.ndarray]
    ):
        """
        Calculate the concrete bounds

        Args:
            layer_num           : For which layer to compute the concrete bounds
            forced_input_bounds : Known input bounds that must be enforced
        """
        mapping = self._task_constants.mappings[layer_num]

        if mapping.is_1d_to_1d:
            self._domain.bounds_concrete[layer_num] = mapping.propagate(
                self._domain.bounds_concrete[layer_num - 1]
            )

        else:
            lower_symbolic_bounds = self._domain.bounds_symbolic[layer_num][0]
            upper_symbolic_bounds = self._domain.bounds_symbolic[layer_num][1]
            lower_concrete_bounds = concretise_symbolic_bounds_jit(
                self._domain.bounds_concrete[0], lower_symbolic_bounds
            )
            upper_concrete_bounds = concretise_symbolic_bounds_jit(
                self._domain.bounds_concrete[0], upper_symbolic_bounds
            )

            self._domain.bounds_concrete[layer_num] = np.stack(
                [
                    lower_concrete_bounds[:, 0],
                    upper_concrete_bounds[:, 1],
                ],
                axis=1,
            )

        self._domain.bounds_concrete[
            layer_num
        ] = self._adjust_bounds_from_forced_bounds(
            self._domain.bounds_concrete[layer_num],
            forced_input_bounds[layer_num],
        )

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

        layer_size = lower_bounds_concrete_in.shape[0]
        relaxations = np.zeros((layer_size, 2))

        # Operating in the positive area
        fixed_upper_idx = np.argwhere(lower_bounds_concrete_in >= 0)
        relaxations[fixed_upper_idx, 0] = 1
        relaxations[fixed_upper_idx, 1] = 0

        # Operating in the negative area
        fixed_lower_idx = np.argwhere(upper_bounds_concrete_in <= 0)
        relaxations[fixed_lower_idx, :] = 0

        # Operating in the non-linear area
        mixed_idx = np.argwhere(
            (upper_bounds_concrete_in > 0) * (lower_bounds_concrete_in < 0)
        )

        if len(mixed_idx) == 0:
            return relaxations

        xl = lower_bounds_concrete_in[mixed_idx]
        xu = upper_bounds_concrete_in[mixed_idx]

        if upper:
            a = xu / (xu - xl)
            b = -a * xl
            relaxations[:, 0][mixed_idx] = a
            relaxations[:, 1][mixed_idx] = b
        else:
            relaxations[:, 0][mixed_idx] = (xl + xu) > 0
            relaxations[:, 1][mixed_idx] = 0

        # Outward rounding
        num_ops = 2
        max_edge_dist = np.hstack((np.abs(xl), np.abs(xu))).max(axis=1)[:, np.newaxis]
        max_err = np.spacing(
            np.abs(relaxations[:, 0][mixed_idx])
        ) * max_edge_dist + np.spacing(np.abs(relaxations[:, 1][mixed_idx]))
        outward_round = max_err * num_ops if upper else -max_err * num_ops
        relaxations[:, 1][mixed_idx] += outward_round

        return relaxations


class BoundsException(Exception):
    pass


class MappingNotImplementedException(BoundsException):
    pass
