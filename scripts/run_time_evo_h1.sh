#!/bin/bash -e

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <device> <cuStateVec_enable> <dim> <precision> <bits> <useSavedFile>"
    exit 1
fi

source ../../.venv/bin/activate
source ../../.env
export PYTHONPATH

python3 ./time_evo_h1.py --device $1 --enable-cuStateVec $2 --dim $3 --precision $4 --bits $5 --use-saved-data $6
