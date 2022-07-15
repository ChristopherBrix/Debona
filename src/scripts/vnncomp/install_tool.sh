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
echo 'export GUROBI_HOME="/opt/gurobi950/linux64"' >> ~/.bashrc
echo 'export PATH="${PATH}:${GUROBI_HOME}/bin"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"' >> ~/.bashrc
echo 'export GRB_LICENSE_FILE="$GUROBI_HOME/gurobi.lic"' >> ~/.bashrc

# pipenv
python3.8 -m pip install pipenv==11.9.0
echo 'export PIPENV_VENV_IN_PROJECT="enabled"' >> ~/.bashrc
echo 'export PIPENV_CACHE_DIR=".cache/pipenv"' >> ~/.bashrc
echo 'export PIP_CACHE_DIR=".cache/pip"' >> ~/.bashrc
echo 'export PIPENV_PIPFILE="/nnvt/src/Pipfile"' >> ~/.bashrc
echo 'export SHELL="/bin/bash"' >> ~/.bashrc
echo 'export LANG="en_US.UTF-8"' >> ~/.bashrc
. ~/.bashrc
cd ../..
pipenv install --dev
cd $GUROBI_HOME && pipenv run python setup.py install