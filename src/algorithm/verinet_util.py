"""
Util classes for nn_bounds

Author: Patrick Henriksen <patrick@henriksen.as>
"""

from enum import Enum
from typing import List

import gurobipy as grb
import numpy as np
from gurobipy.gurobipy import Constr

from src.algorithm.lp_solver import LPSolver
from src.propagation.bound_propagation import bound_propagation


class Status(Enum):
    """
    For keeping track of the verification _status.
    """

    SAFE = 1
    UNSAFE = 2
    UNDECIDED = 3
    UNDERFLOW = 4


class Branch:
    """
    A class to keep track of the data needed when branching
    """

    def __init__(self, depth: int, forced_input_bounds: np.ndarray, split_list: list):

        """
        Args:
            depth   : The current split depth
            forced_input_bounds   : The forced input bounds used in NNBounds
            split_list                  : A list of dictionaries on the form
                                          {"layer": int, "node": int, "split_x": float,
                                          "upper": bool}
        """

        self._depth = depth

        self._forced_input_bounds = forced_input_bounds
        self._split_list = split_list

        self._lp_solver_constraints: List[Constr] = []
        self.safe_classes: list = []

    @property
    def depth(self):
        return self._depth

    @property
    def forced_input_bounds(self):
        return self._forced_input_bounds

    @forced_input_bounds.setter
    def forced_input_bounds(self, bounds):
        self._forced_input_bounds = bounds

    @property
    def split_list(self):
        return self._split_list

    @property
    def lp_solver_constraints(self) -> List[Constr]:
        return self._lp_solver_constraints

    @lp_solver_constraints.setter
    def lp_solver_constraints(self, constraints: List[Constr]):
        self._lp_solver_constraints = constraints

    @staticmethod
    def add_constr_to_solver(
        bounds: bound_propagation, lp_solver: LPSolver, split: np.ndarray
    ) -> grb.Constr:

        """
        Creates a grb constraint from the given split.

        Args:
            bounds      : The NNBounds object
            lp_solver   : The LPSolver object
            split       : The split in format (layer_num, node_num, split_x,
                          upper_split)

        Returns:
              The gurobi constraint
        """

        input_vars = lp_solver.input_variables.select()
        layer, node, split_x, upper = (
            split["layer"],
            split["node"],
            split["split_x"],
            split["upper"],
        )
        symb_input_bounds = bounds.domain.bounds_symbolic[layer - 1][node]

        if upper:
            constr = lp_solver.grb_solver.addConstr(
                grb.LinExpr(symb_input_bounds[:-1], input_vars)
                + bounds.domain.error[layer - 1][node][1]
                + symb_input_bounds[-1]
                >= split_x
            )
        else:
            constr = lp_solver.grb_solver.addConstr(
                grb.LinExpr(symb_input_bounds[:-1], input_vars)
                + bounds.domain.error[layer - 1][node][0]
                + symb_input_bounds[-1]
                <= split_x
            )

        lp_solver.grb_solver.update()

        return constr

    def remove_all_constrs_from_solver(self, solver: LPSolver):

        """
        Removes the last num constraints in self.lp_solver_constraints to grb_solver
        """

        if len(self.lp_solver_constraints) != 0:
            for constr in self.lp_solver_constraints:
                solver.grb_solver.remove(constr)
        self.lp_solver_constraints = []
        solver.grb_solver.update()

    def add_all_constrains(
        self, bounds: bound_propagation, solver: LPSolver, split_list: list
    ):

        """
        Adds all constrains in the split list to the lp-solver.

        This method assumes that the lp-solver does not have any invalid constraints
        from previous branches.


        Args:
            bounds          : The bound_propagation object
            solver          : The LPSolver
            split_list  : The list with splits from the old branch
        """

        assert (
            len(self.lp_solver_constraints) == 0
        ), "Tried adding new constraints before removing old"
        self.lp_solver_constraints = []

        for split in split_list:
            self.lp_solver_constraints.append(
                Branch.add_constr_to_solver(bounds, solver, split)
            )

    def update_constrs(
        self,
        bounds: bound_propagation,
        solver: LPSolver,
        old_split_list: list,
        old_constr_list: List[Constr],
    ):

        """
        Updates the constraints from the constraints of the last branch to the
        constraints of this branch.

        All constraints due to splits that are not in this branch are removed. We also
        re-add all constraints from splits in layers after the layer of the new split.
        This is done since the equations in bound_propagation may have changed. All
        other constraints are kept as is.

        Args:
            bounds          : The bound_propagation object
            solver          : The LPSolver
            old_split_list  : The list with splits from the old branch
            old_constr_list : The list with constraints from the old branch
        """

        assert (
            len(self.lp_solver_constraints) == 0
        ), "Tried adding new constraints before removing old"
        self.lp_solver_constraints = []

        min_layer = bounds.domain.num_layers

        for i in range(self.depth - 1, len(old_split_list)):
            # On backtrack we have to update all nodes after the minimum layer
            # constraint that was changed
            min_layer = min(min_layer, old_split_list[i]["layer"])
            solver.grb_solver.remove(old_constr_list[i])

        min_layer = min(min_layer, self.split_list[self.depth - 1]["layer"])

        # Re-add constraints where the symbolic bounds might change due to the new
        # constraint
        re_add_idx = [
            i for i in range(self.depth - 1) if old_split_list[i]["layer"] > min_layer
        ]

        for i in range(self.depth - 1):

            if i in re_add_idx:

                solver.grb_solver.remove(old_constr_list[i])
                old_constr_list[i] = Branch.add_constr_to_solver(
                    bounds, solver, self.split_list[i]
                )

            self._lp_solver_constraints.append(old_constr_list[i])

        self.lp_solver_constraints.append(
            Branch.add_constr_to_solver(bounds, solver, self.split_list[-1])
        )
        solver.grb_solver.update()
