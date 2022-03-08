"""
Abstract domains are the basis for the concrete implementations like DeepPoly and Star
Sets.
"""

from typing import Optional

import numpy as np

from src.algorithm.task_constants import TaskConstants


class AbstractDomain:

    """
    Abstract class that has all relevent data and methods for all domains.
    """

    def __init__(self, task_constants: TaskConstants):

        """
        Args:
            task_constants: Constant parameters of this task
        """

        self._bounds_concrete: Optional[list] = None
        self._bounds_symbolic: Optional[list] = None

        self._error_matrix: Optional[list] = None
        self._error_matrix_to_node_indices: Optional[list] = None
        self._error: Optional[list] = None

        self._relaxations: Optional[list] = None
        self._task_constants = task_constants

        self._init_datastructure()

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

        num_layers = self._task_constants.num_layers

        self._bounds_concrete: Optional[list] = [None] * num_layers
        self._output_bounds_concrete: Optional[list] = [None] * num_layers

        self._bounds_symbolic: Optional[list] = [None] * num_layers
        self._output_bounds_symbolic: Optional[list] = [None] * num_layers

        self._relaxations: Optional[list] = [None] * num_layers

        self._error_matrix: Optional[list] = [None] * num_layers
        self._error_matrix_to_node_indices: Optional[list] = [None] * num_layers
        self._error: Optional[list] = [None] * num_layers

        # Set the error matrices of the input layer to zero
        self._error_matrix[0] = np.zeros(
            (self._task_constants.layer_sizes[0], 0), dtype=np.float32
        )
        self._error_matrix_to_node_indices[0] = np.zeros((0, 2), dtype=int)

        # Set the correct symbolic equations for input layer
        diagonal_idx = np.arange(self._task_constants.layer_sizes[0])
        self._bounds_symbolic[0] = np.zeros(
            (
                self._task_constants.layer_sizes[0],
                self._task_constants.layer_sizes[0] + 1,
            ),
            dtype=np.float32,
        )
        self._bounds_symbolic[0][diagonal_idx, diagonal_idx] = 1
