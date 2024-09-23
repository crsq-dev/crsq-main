#!/bin/bash -e

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <device> <cuStateVec_enable> <dim> <precision> <bits>"
    exit 1
fi

DEVICE=$1
CUSTATEVEC=$2
DIM=$3
PRECISION=$4
BITS=$5

source ../../.venv/bin/activate
source ../../.env
export PYTHONPATH

python3 ./time_evo_h1.py --device $1 --cuStateVec_enable $2 --dim $3 --precision $4 --bits $5
