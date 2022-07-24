#!/bin/bash

VERSION_STRING=v1

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

CATEGORY=$2
ONNX_FILE=$3
VNNLIB_FILE=$4
RESULTS_FILE=$5
TIMEOUT=$6

echo "Running benchmark instance in category '$CATEGORY' with onnx file '$ONNX_FILE', vnnlib file '$VNNLIB_FILE', results file $RESULTS_FILE, and timeout $TIMEOUT"

current_dir=`pwd`
cd $(dirname $(dirname $(dirname $0)))
cat /dev/null > "$current_dir/$RESULTS_FILE"
pwd
pipenv run python scripts/vnncomp/benchmark_vnncomp.py "$current_dir/$ONNX_FILE" "$current_dir/$VNNLIB_FILE" "$current_dir/$RESULTS_FILE" 10 1 0
if grep -q "run_instance_timeout" "$current_dir/$RESULTS_FILE"; then
	pipenv run python scripts/vnncomp/benchmark_vnncomp.py "$current_dir/$ONNX_FILE" "$current_dir/$VNNLIB_FILE" "$current_dir/$RESULTS_FILE" $(($TIMEOUT-10)) 0 0
fi

sed -i "s/violated/sat/g" "$current_dir/$RESULTS_FILE"
sed -i "s/holds/unsat/g" "$current_dir/$RESULTS_FILE"
