from typing import Optional
import torch
import torch.nn as nn
import numpy as np

from src.algorithm.mappings.abstract_mapping import AbstractMapping
from src.neural_networks.verinet_nn import VeriNetNN
from src.algorithm.esip_util import concretise_symbolic_bounds_jit, sum_error_jit


class AbstractDomainPropagation:

    """
    Abstract class for the propagation of the domain
    """

    def __init__(self, model: VeriNetNN, input_shape):

        """
        Args:

            model                       : The VeriNetNN neural network as defined in src/neural_networks/verinet_nn.py
            input_shape                 : The shape of the input, (input_size,) for 1D input or
                                          (channels, height, width) for 2D.
        """

        self._model = model
        self._input_shape = input_shape

        self._mappings = None
        self._layer_sizes = None
        self._layer_shapes = None

        self._bounds_concrete: Optional[list] = None
        self._bounds_symbolic: Optional[list] = None

        self._error_matrix: Optional[list] = None
        self._error_matrix_to_node_indices: Optional[list] = None
        self._error: Optional[list] = None

        self._relaxations: Optional[list] = None
        self._forced_input_bounds: Optional[list] = None

        self._read_mappings_from_torch_model(model)
        self._init_datastructure()

    @property
    def layer_sizes(self):
        return self._layer_sizes

    @property
    def num_layers(self):
        return len(self.layer_sizes)

    @property
    def mappings(self):
        return self._mappings

    @property
    def bounds_concrete(self):
        return self._bounds_concrete

    @property
    def bounds_symbolic(self):
        return self._bounds_symbolic

    @property
    def relaxations(self):
        return self._relaxations

    @property
    def forced_input_bounds(self):
        return self._forced_input_bounds

    @forced_input_bounds.setter
    def forced_input_bounds(self, val: np.array):
        self._forced_input_bounds = val

    @property
    def error(self):

        """
        The lower and upper errors for the nodes.

        For a network with L layers and Ni nodes in each layer, the error is a list of length L where each element is
        a Ni x 2 array. The first element of the last dimension is the sum of the negative errors, while the second
        element is the sum of the positive errors as represented by the column in the error matrix.
        """

        return self._error

    @property
    def error_matrix(self):
        return self._error_matrix

    def reset_datastruct(self):

        """
        Resets the symbolic datastructure
        """

        self._init_datastructure()

    def _init_datastructure(self):

        """
        Initialises the data-structure.
        """

        num_layers = len(self.layer_sizes)

        self._bounds_concrete: Optional[list] = [None] * num_layers
        self._output_bounds_concrete: Optional[list] = [None] * num_layers

        self._bounds_symbolic: Optional[list] = [None] * num_layers
        self._output_bounds_symbolic: Optional[list] = [None] * num_layers

        self._relaxations: Optional[list] = [None] * num_layers
        self._forced_input_bounds: Optional[list] = [None] * num_layers

        self._error_matrix: Optional[list] = [None] * num_layers
        self._error_matrix_to_node_indices: Optional[list] = [None] * num_layers
        self._error: Optional[list] = [None] * num_layers

        # Set the error matrices of the input layer to zero
        self._error_matrix[0] = np.zeros((self.layer_sizes[0], 0), dtype=np.float32)
        self._error_matrix_to_node_indices[0] = np.zeros((0, 2), dtype=int)

        # Set the correct symbolic equations for input layer
        diagonal_idx = np.arange(self.layer_sizes[0])
        self._bounds_symbolic[0] = np.zeros(
            (self._layer_sizes[0], self._layer_sizes[0] + 1), dtype=np.float32
        )
        self._bounds_symbolic[0][diagonal_idx, diagonal_idx] = 1

    def _read_mappings_from_torch_model(self, torch_model: VeriNetNN):

        """
        Initializes the mappings from the torch model.

        Args:
            torch_model : The Neural Network
        """

        # Initialise with None for input layer
        self._mappings = [None]
        self._layer_shapes = [self._input_shape]

        for layer in torch_model.layers:
            self._process_layer(layer)

        self._layer_sizes = [int(np.prod(shape)) for shape in self._layer_shapes]

    def _process_layer(self, layer: int):

        """
        Processes the mappings (Activation function, FC, Conv, ...) for the given "layer".

        Reads the mappings from the given layer, adds the relevant abstraction to self._mappings and calculates the data
        shape after the mappings.

        Args:
            layer: The layer number
        """

        # Recursively process Sequential layers
        if isinstance(layer, nn.Sequential):
            for child in layer:
                self._process_layer(child)
            return

        # Add the mapping
        try:
            self._mappings.append(
                AbstractMapping.get_activation_mapping_dict()[layer.__class__]()
            )
        except KeyError as e:
            raise MappingNotImplementedException(
                f"Mapping: {layer} not implemented"
            ) from e

        # Add the necessary parameters (Weight, bias....)
        for param in self._mappings[-1].required_params:

            attr = getattr(layer, param)

            if isinstance(attr, torch.Tensor):
                self._mappings[-1].params[param] = attr.detach().numpy()
            else:
                self._mappings[-1].params[param] = attr

        # Calculate the output shape of the layer
        self._mappings[-1].params["in_shape"] = self._layer_shapes[-1]
        self._layer_shapes.append(self._mappings[-1].out_shape(self._layer_shapes[-1]))

    def calc_bounds(self, input_constraints: np.array, from_layer: int = 1) -> bool:

        """
        Calculate the bounds for all layers in the network starting at from_layer.

        Notice that from_layer is usually larger than 1 after a split. In this case, the split constraints are
        added to the layer before from_layer by adjusting the forced bounds. For this reason, we update the
        concrete_bounds for the layer before from_layer.

        Args:
            input_constraints       : The constraints on the input. The first dimensions should be the same as the
                                      input to the neural network, the last dimension should contain the lower bound
                                      on axis 0 and the upper on axis 1.
            from_layer              : Updates this layer and all later layers

        Returns:
            True if the method succeeds, False if the bounds are invalid. The bounds are invalid if the forced bounds
            make at least one upper bound smaller than a lower bound.
        """

        raise NotImplementedError(
            f"calc_bounds(...) not implemented in {self.__name__}"
        )

    def merge_current_bounds_into_forced(self):

        """
        Sets forced input bounds to the best of current forced bounds and calculated bounds.
        """

        raise NotImplementedError(
            f"merge_current_bounds_into_forced(...) not implemented in {self.__name__}"
        )

    def largest_error_split_node(
        self, output_weights: np.array = None
    ) -> Optional[tuple]:

        """
        Returns the node with the largest weighted error effect on the output

        The error from overestimation is calculated for each output node with respect to each hidden node.
        This value is weighted using the given output_weights and the index of the node with largest effect on the
        output is returned.

        Args:
            output_weights  : A Nx2 array with the weights for the lower bounds in column 1 and the upper bounds
                              in column 2. All weights should be >= 0.
        Returns:
              (layer_num, node_num) of the node with largest error effect on the output
        """

        raise NotImplementedError(
            f"largest_error_split_node(...) not implemented in {self.__name__}"
        )


class BoundsException(Exception):
    pass


class MappingNotImplementedException(BoundsException):
    pass
