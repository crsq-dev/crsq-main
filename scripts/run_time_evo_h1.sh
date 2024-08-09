#!/bin/bash -e

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <device> <cuStateVec_enable> <precision> <bits>"
    exit 1
fi

DEVICE=$1
CUSTATEVEC=$2
PRECISION=$3
BITS=$4

source ../../.venv/bin/activate
source ../../.env
export PYTHONPATH

python3 ./time_evo_h1.py --device $1 --cuStateVec_enable $2 --precision $3 --bits $4
