"""
Scripts used for benchmarking

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import os
import random
import sys
import time
from shutil import copyfile

import gurobipy as grb
import numpy as np
import torch

from src.algorithm.status import Status
from src.algorithm.verification_objectives import ArbitraryObjective
from src.algorithm.verinet import VeriNet
from src.data_loader.onnx_parser import ONNXParser

from src.propagation.deep_poly_propagation import (
    DeepPolyBackwardPropagation,
    DeepPolyForwardPropagation,
)
from src.scripts.vnncomp import vnnlib

from src.util import config

RANDOM_SEED: int = 0
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if __name__ == "__main__":
    assert len(sys.argv) == 6
    model_path = sys.argv[1]
    vnnlib_path = sys.argv[2]
    result_path = sys.argv[3]
    timeout: int = int(float(sys.argv[4]))
    forward_prop: bool = bool(sys.argv[5])

    if forward_prop:
        print("Using forward propagation")
        config.DOMAIN_PROPAGATION = DeepPolyForwardPropagation
    else:
        print("Using backward propagation")
        config.DOMAIN_PROPAGATION = DeepPolyBackwardPropagation

    # Get the "Academic license" print from gurobi at the beginning
    grb.Model()

    onnx_parser = ONNXParser(model_path)
    if os.path.isfile(result_path):
        copyfile(result_path, result_path + ".bak")

    input_nodes, output_nodes = vnnlib.get_num_inputs_outputs(model_path)
    a = vnnlib.read_vnnlib_simple(vnnlib_path, input_nodes, output_nodes)

    assert len(a) == 1
    input_bounds, constraints = a[0]
    input_bounds = np.array(input_bounds)
    objectiveMatrix = []
    objectiveBias = []
    for weights, bias in constraints:
        objectiveMatrix.append(weights)
        objectiveBias.append(bias)
    objectives = np.concatenate(
        [np.array(objectiveMatrix), -np.array(objectiveBias)[:, :, np.newaxis]], axis=2
    )

    with open(result_path, "w", buffering=1, encoding="UTF-8") as f:

        solver = VeriNet(
            gradient_descent_max_iters=5,
            gradient_descent_step=1e-1,
            gradient_descent_min_loss_change=1e-2,
            max_procs=None,
        )
        start = time.time()
        objective = ArbitraryObjective(
            objectives, input_bounds, output_size=output_nodes
        )
        status = solver.verify(
            model=onnx_parser.to_pytorch(),
            verification_objective=objective,
            timeout=timeout,
            no_split=False,
            gradient_descent_intervals=5,
            verbose=False,
        )
        if status == Status.SAFE:
            f.write("holds")
        elif status == Status.UNSAFE:
            f.write("violated")
        else:
            f.write("run_instance_timeout")

        print(
            f"Final result of input: {status}, branches explored:"
            f" {solver.branches_explored}, max depth: {solver.max_depth}, time"
            f" spent: {time.time()-start:.2f} seconds\n"
        )
