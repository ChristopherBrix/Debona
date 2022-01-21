# syntax=docker/dockerfile:1
FROM ubuntu:20.04 AS base

# Basic setup
RUN apt-get update
RUN apt-get install -y git python3.8 python3.8-dev python3.8-distutils curl wget
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.8 get-pip.py
RUN python3.8 -m pip install --upgrade pip

# Gurobi
RUN cd /opt && \
    wget https://packages.gurobi.com/9.5/gurobi9.5.0_linux64.tar.gz && \
    tar -zxf gurobi9.5.0_linux64.tar.gz && \
    rm gurobi9.5.0_linux64.tar.gz
ENV GUROBI_HOME="/opt/gurobi950/linux64"
ENV PATH="${PATH}:${GUROBI_HOME}/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
ENV GRB_LICENSE_FILE="$GUROBI_HOME/gurobi.lic"
RUN printf "TOKENSERVER=license6.itc.rwth-aachen.de\nPORT=50039" > $GRB_LICENSE_FILE

# Files needed for pipenv
WORKDIR /nnvt
RUN mkdir src
COPY src/Pipfile src/Pipfile
COPY src/Pipfile.lock src/Pipfile.lock
COPY src/.env src/.env

# pipenv
RUN python3.8 -m pip install pipenv==11.9.0
ENV PIPENV_VENV_IN_PROJECT="enabled"
ENV PIPENV_CACHE_DIR=".cache/pipenv"
ENV PIP_CACHE_DIR=".cache/pip"
ENV PIPENV_PIPFILE="/nnvt/src/Pipfile"
ENV SHELL="/bin/bash"
ENV LANG="en_US.UTF-8"
RUN pipenv install --dev
RUN cd $GUROBI_HOME && pipenv run python setup.py install

FROM base as complete
COPY . .