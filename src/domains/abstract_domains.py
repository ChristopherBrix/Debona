"""
Abstract domains are the basis for the concrete implementations like DeepPoly and Star
Sets.
"""

from typing import Optional

import numpy as np
import torch
from torch import nn

from src.algorithm.mappings.abstract_mapping import AbstractMapping
from src.neural_networks.verinet_nn import VeriNetNN


class AbstractDomain:

    """
    Abstract class that has all relevent data and methods for all domains.
    """

    def __init__(self, model: VeriNetNN, input_shape):

        """
        Args:

            model                       : The VeriNetNN neural network as defined in
                                          src/neural_networks/verinet_nn.py
            input_shape                 : The shape of the input, (input_size,) for 1D
                                          input or (channels, height, width) for 2D.
        """

        self._model = model
        self._input_shape = input_shape

        self._mappings: list = []
        self._layer_sizes: Optional[list] = []
        self._layer_shapes: list = []

        self._bounds_concrete: Optional[list] = None
        self._bounds_symbolic: Optional[list] = None

        self._error_matrix: Optional[list] = None
        self._error_matrix_to_node_indices: Optional[list] = None
        self._error: Optional[list] = None

        self._relaxations: Optional[list] = None
        self._forced_input_bounds: Optional[np.ndarray] = None

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
    def forced_input_bounds(self, val: np.ndarray):
        self._forced_input_bounds = val

    @property
    def error(self):

        """
        The lower and upper errors for the nodes.

        For a network with L layers and Ni nodes in each layer, the error is a list of
        length L where each element is a Ni x 2 array. The first element of the last
        dimension is the sum of the negative errors, while the second element is the sum
        of the positive errors as represented by the column in the error matrix.
        """

        return self._error

    @property
    def error_matrix(self):
        return self._error_matrix

    @property
    def error_matrix_to_node_indices(self):
        return self._error_matrix_to_node_indices

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
        Processes the mappings (Activation function, FC, Conv, ...) for the given
        "layer".

        Reads the mappings from the given layer, adds the relevant abstraction to
        self._mappings and calculates the data shape after the mappings.

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


class BoundsException(Exception):
    pass


class MappingNotImplementedException(BoundsException):
    pass
