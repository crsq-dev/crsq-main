#!/bin/bash -e

if [ "$#" -ne 9 ]; then
    echo "Usage: $0 <device> <cuStateVec_enable> <precision> <bits> <useSavedFile> <qn> <qm> <elec-iters> <nucl-iters>"
    exit 1
fi

source ../../.venv/bin/activate
source ../../.env
export PYTHONPATH

python3 ./time_evo_2D_h1.py --device $1 --enable-cuStateVec $2 --precision $3 --bits $4 --use-saved-data $5 --qnum-n $6 --qnum-m $7 --num-elec-iters $8 --num-nucl-iters $9
