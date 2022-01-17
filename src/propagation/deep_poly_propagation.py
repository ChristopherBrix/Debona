"""
Propagation using DeepPoly.
"""

from typing import Optional

import numpy as np

from src.algorithm.esip_util import concretise_symbolic_bounds_jit, sum_error_jit
from src.algorithm.mappings.abstract_mapping import AbstractMapping
from src.domains.deep_poly import DeepPoly
from src.neural_networks.verinet_nn import VeriNetNN
from src.propagation.abstract_domain_propagation import AbstractDomainPropagation


class DeepPolyPropagation(AbstractDomainPropagation):
    """Class that implements the DeepPoly algorithm"""

    def __init__(self, model: VeriNetNN, input_shape):
        """
        Args:

            model                       : The VeriNetNN neural network as defined in
                                          src/neural_networks/verinet_nn.py
            input_shape                 : The shape of the input, (input_size,) for 1D
                                          input or (channels, height, width) for 2D.
        """
        domain = DeepPoly(model, input_shape)
        super().__init__(domain)

    def merge_current_bounds_into_forced(self):

        """
        Sets forced input bounds to the best of current forced bounds and calculated
        bounds.
        """

        for i in range(self._domain.num_layers):
            if self._domain.bounds_concrete[i] is None:
                continue

            elif self._domain.forced_input_bounds[i] is None:
                self._domain.forced_input_bounds[i] = self._domain.bounds_concrete[i]

            else:
                better_lower = (
                    self._domain.forced_input_bounds[i][:, 0]
                    < self._domain.bounds_concrete[i][:, 0]
                )
                self._domain.forced_input_bounds[i][
                    better_lower, 0
                ] = self._domain.bounds_concrete[i][better_lower, 0]

                better_upper = (
                    self._domain.forced_input_bounds[i][:, 1]
                    > self._domain.bounds_concrete[i][:, 1]
                )
                self._domain.forced_input_bounds[i][
                    better_upper, 1
                ] = self._domain.bounds_concrete[i][better_upper, 1]

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

        if self._domain.error_matrix[-1].shape[1] == 0:
            return None

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
        max_err_idx = np.argmax(weighted_error)

        if weighted_error[max_err_idx] <= 0:
            return None
        else:
            return self._domain.error_matrix_to_node_indices[-1][max_err_idx]


class DeepPolyForwardPropagation(DeepPolyPropagation):
    """Class that implements the DeepPoly forward propagation algorithm"""

    def calc_bounds(self, input_constraints: np.ndarray, from_layer: int = 1) -> bool:

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
            (
                self._domain.bounds_concrete[from_layer - 1],
                self._domain.error[from_layer - 1],
            ) = self._calc_bounds_concrete_jit(
                self._domain.bounds_concrete[0],
                self._domain.bounds_symbolic[from_layer - 1],
                self._domain.error_matrix[from_layer - 1],
            )

            self._domain.bounds_concrete[
                from_layer - 1
            ] = self._adjust_bounds_from_forced_bounds(
                self._domain.bounds_concrete[from_layer - 1],
                self._domain.forced_input_bounds[from_layer - 1],
            )

        for layer_num in range(from_layer, self._domain.num_layers):

            success = self._prop_bounds_and_errors(layer_num)
            if not success:
                return False

        return True

    def _prop_bounds_and_errors(self, layer_num: int) -> bool:

        """
        Calculates the symbolic input bounds.

        This updates all bounds, relaxations and errors for the given layer, by
        propagating from the previous layer.

        Args:
            layer_num: The layer number

        Returns:
            True if the resulting concrete bounds are valid, else false.
        """

        mapping = self._domain.mappings[layer_num]

        if mapping.is_linear:
            self._domain.bounds_symbolic[layer_num] = mapping.propagate(
                self._domain.bounds_symbolic[layer_num - 1], add_bias=True
            )
            self._domain.error_matrix[layer_num] = mapping.propagate(
                self._domain.error_matrix[layer_num - 1], add_bias=False
            )
            self._domain.error_matrix_to_node_indices[
                layer_num
            ] = self._domain.error_matrix_to_node_indices[layer_num - 1].copy()

        else:
            self._domain.relaxations[layer_num] = self._calc_relaxations(
                self._domain.mappings[layer_num],
                self._domain.bounds_concrete[layer_num - 1],
            )

            self._domain.bounds_symbolic[
                layer_num
            ] = self._prop_equation_trough_relaxation(
                self._domain.bounds_symbolic[layer_num - 1],
                self._domain.relaxations[layer_num],
            )

            (
                self._domain.error_matrix[layer_num],
                self._domain.error_matrix_to_node_indices[layer_num],
            ) = self._prop_error_matrix_trough_relaxation(
                self._domain.error_matrix[layer_num - 1],
                self._domain.relaxations[layer_num],
                self._domain.bounds_concrete[layer_num - 1],
                self._domain.error_matrix_to_node_indices[layer_num - 1],
                layer_num,
            )

        if mapping.is_1d_to_1d:
            self._domain.bounds_concrete[layer_num] = mapping.propagate(
                self._domain.bounds_concrete[layer_num - 1]
            )

        else:
            (
                self._domain.bounds_concrete[layer_num],
                self._domain.error[layer_num],
            ) = self._calc_bounds_concrete_jit(
                self._domain.bounds_concrete[0],
                self._domain.bounds_symbolic[layer_num],
                self._domain.error_matrix[layer_num],
            )

        self._domain.bounds_concrete[
            layer_num
        ] = self._adjust_bounds_from_forced_bounds(
            self._domain.bounds_concrete[layer_num],
            self._domain.forced_input_bounds[layer_num],
        )

        return self._valid_concrete_bounds(self._domain.bounds_concrete[layer_num])

    @staticmethod
    def _calc_bounds_concrete_jit(
        input_bounds: np.ndarray, symbolic_bounds: np.ndarray, error_matrix: np.ndarray
    ) -> tuple:

        """
        Calculates the concrete input bounds and error from the symbolic input bounds
        and error matrix.

        The concrete bounds are calculated by maximising/ minimising the symbolic bounds
        for each layer and adding the error. The lower/ upper errors are the sum of the
        negative/ positive values respectively of each row in the error-matrix.

        Args:
            input_bounds    : A Mx2 array with the input bounds, where M is the input
                              dimension of the network.
            symbolic_bounds : A Nx(M+1) array with the symbolic bounds, where N is the
                              number of nodes in the layer and M is the input dimension
                              of the network.
            error_matrix    : A NxN' with the errors, where N is the number of nodes in
                              the layer and N' is the
                              total number of nodes in all previous layers. Returns
                              (concrete_bounds, errors), where concrete_bounds and
                              errors are Nx2 arrays.
        """

        concrete_bounds = concretise_symbolic_bounds_jit(input_bounds, symbolic_bounds)
        errors = sum_error_jit(error_matrix)
        concrete_bounds += errors

        return concrete_bounds, errors

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

    @staticmethod
    def _calc_relaxations(
        mapping: AbstractMapping, bounds_concrete: np.ndarray
    ) -> np.ndarray:

        """
        Calculates the linear relaxations for the given mapping and concrete bounds.

        Args:
            mapping         : The mapping for which to calculate the linear relaxations
            bounds_concrete : A Nx2 array with the concrete lower and upper input bounds

        Returns:
            A 2xNx2 array where the first dimension indicates the lower and upper
            relaxation, the second dimension are the nodes in the current layer and the
            last dimension contains the parameters [a, b] in l(x) = ax + b.
        """

        lower_relaxation = mapping.linear_relaxation(
            bounds_concrete[:, 0], bounds_concrete[:, 1], False
        )
        upper_relaxation = mapping.linear_relaxation(
            bounds_concrete[:, 0], bounds_concrete[:, 1], True
        )

        return np.concatenate(
            (lower_relaxation[np.newaxis, :, :], upper_relaxation[np.newaxis, :, :]),
            axis=0,
        )

    @staticmethod
    def _prop_equation_trough_relaxation(
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

    @staticmethod
    def _prop_error_matrix_trough_relaxation(
        error_matrix: np.ndarray,
        relaxations: np.ndarray,
        bounds_concrete: np.ndarray,
        error_matrix_to_node_indices: np.ndarray,
        layer_num: int,
    ) -> tuple:

        """
        Updates the error matrix and the error_matrix_to_node_indices array.

        The old errors are propagated through the lower relaxations and the new errors
        due to the relaxations at this layer are concatenated the result.

        Args:
            error_matrix                : A NxN' with the errors, where N is the number
                                          of nodes in the layer and N' is the total
                                          number of nodes in all previous layers.
            relaxations                 : A 2xNx2 array where the first dimension
                                          indicates the lower and upper relaxation, the
                                          second dimension contains the nodes in the
                                          current layer and the last dimension contains
                                          the parameters [a, b] in l(x) = ax + b.
            bounds_concrete             : A Nx2 array with the concrete lower and upper
                                          input bounds.
            error_matrix_to_node_indices: A N'x2 mapping where the first column contains
                                          the layer and the second column contains the
                                          node-number of the corresponding column in the
                                          error matrix.
            layer_num                   : The number of the current layer, used to
                                          update error_matrix_to_node_indices.

        Returns:
            (error_matrix_new, error_matrix_to_node_indices_new), where both are on the
            same form as explained in the input arguments.
        """

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

            error_matrix_to_node_indices_new = np.hstack(
                (
                    np.zeros(err_idx.shape, dtype=int)[:, np.newaxis] + layer_num,
                    err_idx[:, np.newaxis],
                )
            )
            error_matrix_to_node_indices_new = np.vstack(
                (error_matrix_to_node_indices, error_matrix_to_node_indices_new)
            )
        else:
            error_matrix_to_node_indices_new = error_matrix_to_node_indices.copy()

        return error_matrix_new, error_matrix_to_node_indices_new


# TODO
# class DeepPolyBackwardPropagation(DeepPolyPropagation):
#     """Class that implements the DeepPoly backward propagation algorithm"""

#     pass


class BoundsException(Exception):
    pass


class MappingNotImplementedException(BoundsException):
    pass
