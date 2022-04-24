"""
TaskConstants store constant parameters of the task to make them available
"""

from typing import List

import numpy as np
import torch
from torch import nn

from src.algorithm.mappings.abstract_mapping import AbstractMapping
from src.neural_networks.verinet_nn import VeriNetNN


class TaskConstants:
    def __init__(self, model: VeriNetNN, input_shape: np.ndarray):
        self._model = model
        self._input_shape = input_shape
        self._layer_sizes: list = []
        self._layer_shapes: list = []

        self._read_mappings_from_torch_model()

    @property
    def model(self) -> VeriNetNN:
        return self._model

    @property
    def input_shape(self) -> np.ndarray:
        return self._input_shape

    @property
    def layer_sizes(self) -> list:
        return self._layer_sizes

    @property
    def layer_shapes(self) -> list:
        return self._layer_shapes

    @property
    def num_layers(self) -> int:
        return len(self.layer_sizes)

    @property
    def mappings(self) -> List[AbstractMapping]:
        return self._mappings

    def _read_mappings_from_torch_model(self):

        """
        Initializes the mappings from the torch model.

        Args:
            torch_model : The Neural Network
        """

        # Initialise with None for input layer
        self._mappings = [None]
        self._layer_shapes = [self.input_shape]

        for layer in self.model.layers:
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


class MappingNotImplementedException(Exception):
    pass
