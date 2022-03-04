"""
Can run any benchmark passed to it via the command line.

python run_benchmark.py benchmark_name model_path image_dir epsilon timeout output_dir
  max_procs
"""

import os
from ast import literal_eval

from black import sys

from src.data_loader.input_data_loader import *  # pylint: disable=wildcard-import,unused-wildcard-import # noqa: F401,F403
from src.scripts.benchmark import run_benchmark


def start_benchmark():
    assert len(sys.argv) == 11
    # pylint: disable=unbalanced-tuple-unpacking
    (
        _,
        benchmark_name,
        model_path,
        conv,
        image_dir,  # pylint: disable=unused-variable
        num_images,
        load_func,
        epsilon,
        timeout,
        output_dir,
        max_procs,
    ) = sys.argv

    os.makedirs(output_dir, exist_ok=True)

    num_images = int(num_images)

    # The use of eval(load_func) allows us to handle the different formats for mnist and
    # cifar benchmarks.
    run_benchmark(
        images=eval(load_func.strip("'\"")),  # pylint: disable=eval-used
        epsilons=[float(epsilon)],
        timeout=float(timeout),
        conv=literal_eval(conv),
        model_path=model_path,
        result_path=f"{output_dir}/{benchmark_name}.txt",
        max_procs=int(max_procs),
    )


if __name__ == "__main__":
    start_benchmark()
