# Basic setup
apt-get update
apt-get install -y git python3.8 python3.8-dev python3.8-distutils curl wget
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.8 get-pip.py
python3.8 -m pip install --upgrade pip

# Gurobi
cd /opt && \
    wget https://packages.gurobi.com/9.5/gurobi9.5.0_linux64.tar.gz && \
    tar -zxf gurobi9.5.0_linux64.tar.gz && \
    rm gurobi9.5.0_linux64.tar.gz && \
    cd -

# pipenv
python3.8 -m pip install pipenv==11.9.0
pipenv_dir=`dirname $(dirname $(pwd))`
ed -s ~/.bashrc <<EOF
0 i
export GUROBI_HOME="/opt/gurobi950/linux64"
export PATH="\${PATH}:\${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="\${LD_LIBRARY_PATH}:\${GUROBI_HOME}/lib"
export GRB_LICENSE_FILE="\${GUROBI_HOME}/gurobi.lic"

export PIPENV_VENV_IN_PROJECT="enabled"
export PIPENV_CACHE_DIR=".cache/pipenv"
export PIP_CACHE_DIR=".cache/pip"
export PIPENV_PIPFILE="$pipenv_dir/Pipfile"
export SHELL="/bin/bash"
export LANG="en_US.UTF-8"
.
w
EOF
. ~/.bashrc
cd $pipenv_dir
su ubuntu -c 'pipenv install --dev'

cd ~
git clone https://github.com/xianyi/OpenBLAS
cd OpenBLAS
make FC=gfortran
sudo make PREFIX=/opt/openblas install
cd ..
git clone https://github.com/numpy/numpy
cd numpy/
git checkout v1.22.4 
cat << EOF > site.cfg
[default]
include_dirs = /opt/openblas/include
library_dirs = /opt/openblas/lib
 
[openblas]
openblas_libs = openblas
library_dirs = /opt/openblas/lib
 
[lapack]
lapack_libs = openblas
library_dirs = /opt/openblas/lib
EOF

pipenv run pip install cython==0.29.30
git submodule update --init
pipenv run python setup.py build
pipenv run python setup.py install


cd $GUROBI_HOME && su ubuntu -c 'pipenv run sudo python setup.py install'
grbprobe