# VeriNet Open Source

This repository contains the open source version of VeriNet VeriNet toolkit for local robustness verification of feed-forward neural networks.  

# Important Notice

This version of VeriNet is outdated and should **not be used for benchmarking**; 
for benchmarking purposes see https://github.com/vas-group-imperial/VeriNet. 

## Installation

### Pipenv

Most dependencies can be installed via pipenv:

$ cd <your_verinet_path>/VeriNet/src
$ pipenv install --dev

(If pipenv install fails, try pipenv install torch==1.1.0 and rerun pipenv install)

### Gurobi

VeriNet uses the Gurobi LP-solver which has a free academic license.  

1) Go to https://www.gurobi.com, download Gurobi and get the license.  
2) Follow the install instructions from http://abelsiqueira.github.io/blog/installing-gurobi-7-on-linux/  
3) Activate pipenv by cd'ing into your VeriNet/src and typing $pipenv shell
4) Find your python path by typing $which python
5) cd into your Gurobi installation and run $<your python path> setup.py install

### Numpy with OpenBLAS

For optimal performance, we recommend compiling Numpy from source with OpenBLAS.

Install instruction can be found at: 
https://hunseblog.wordpress.com/2014/09/15/installing-numpy-and-openblas/  

### Development: Git hooks

To run all configured git hooks for each commit, please run
`pipenv run pre-commit install`. For intermediate commits, feel free to skip the checks
by using `git commit --no-verify`, but all tests should pass before creating a merge
request.

If you are using VSCode, you may want to copy the settings in 
`.vscode/settings.json.default` to `.vscode/settings.json`. This enables all tests after
each save.

## IMPORTANT NOTES:

### Cuda devices

Pythons multiprocessing does not work well with Cuda. To avoid any problems 
we hide all Cuda devices using environment variables. This is done in 
VeriNet/src/.env which is run every time you enter pipenv shell. 
If you have a Cuda device and do not use the pipenv environment, you have to 
manually enter:

$export CUDA_DEVICE_ORDER="PCI_BUS_ID"  
$export CUDA_VISIBLE_DEVICES=""

### OpenBLAS threads

Since our algorithm has a highly efficient parallel implementation, OpenBLAS 
should be limited to 1 thread. This can not be done in runtime after Numpy is 
loaded, so we use an environment variable instead. 
The variable is automatically set when using pipenv and can be found in 
VeriNet/src/.env. If you do not use the pipenv environment, this has to be done 
manually with the command:

$export OMP_NUM_THREADS=1

## Usage

All of the experiments used in the paper can be run with the scripts in
VeriNet/src/scripts. The file VeriNet/examples/examples.py contains several
examples of how to run the algorithm using networks loaded from the nnet
format and custom networks.  More information about the nnet format can be found
in VeriNet/data/models_nnet.

## Authors

Patrick Henriksen: ph818@ic.ac.uk  
Alessio Lomuscio
