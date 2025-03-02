#!/bin/bash -e

if [ "$#" -ne 10 ]; then
    echo "Usage: $0 <device> <cuStateVec_enable> <precision> <bits> <st-method> <useSavedFile> <signed> <delta-t> <elec-iters> <nucl-iters>"
    exit 1
fi

#source ../../.venv/bin/activate
#source ../../.env
#export PYTHONPATH

python3 ./time_evo_h1.py --device $1 --enable-cuStateVec $2 --precision $3 --bits $4 --st-method $5 \
 --use-saved-data $6 --signed $7 --delta-t $8 --num-elec-iters $9 --num-nucl-iters $10
