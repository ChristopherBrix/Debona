"""
This file contains abstractions for piecewise linear activation functions (ReLU,
Identity ...).

The abstractions are used to calculate linear relaxations, function values, and
derivatives

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import numpy as np
import torch.nn as nn

from src.algorithm.mappings.abstract_mapping import (
    AbstractMapping,
    ActivationFunctionAbstractionException,
)


class Relu(AbstractMapping):
    @property
    def is_linear(self) -> bool:
        return False

    @property
    def is_1d_to_1d(self) -> bool:
        return True

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Returns:
            A list with all torch functions that are abstracted by the current subclass.
        """

        return [nn.modules.activation.ReLU, nn.ReLU]

    def propagate(self, x: np.ndarray, _add_bias: bool = True):

        """
        Propagates trough the mapping by applying the ReLU function element-wise.

        Args:
            x           : The input as a np.array
            add_bias    : Adds bias if relevant, for example for FC and Conv layers.

        Returns:
            The value of the activation function at x
        """

        return (x > 0) * x

    def split_point(self, _xl: float, _xu: float):

        """
        Returns the preferred split point for branching which is 0 for the ReLU.

        Args:
            xl  : The lower bound on the input
            xu  : The upper bound on the input

        Returns:
            The preferred split point
        """

        return 0


class Identity(AbstractMapping):
    @property
    def is_linear(self) -> bool:
        return True

    @property
    def is_1d_to_1d(self) -> bool:
        return True

    def propagate(self, x: np.ndarray, _add_bias: bool = True) -> np.ndarray:

        """
        Propagates trough the mapping by returning the input unchanged.

        Args:
            x           : The input as a np.array
            add_bias    : Adds bias if relevant, for example for FC and Conv layers.

        Returns:
            The value of the activation function at x
        """

        return x

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        This function is used to create a mapping from torch functions to their
        abstractions.

        Returns:
           A list with all torch functions that are abstracted by the current subclass.
        """

        return []

    def split_point(self, _xl: float, _xu: float):

        """
        Not implemented since function is linear.
        """

        msg = (
            f"split_point(...) not implemented for {self.__class__.__name__} "
            + "since it is linear "
        )
        raise ActivationFunctionAbstractionException(msg)
