#!/bin/sh -e

METHOD=STAR

for B in 5 6 7 ; do
    echo "Running with $B bits"
    python time_evo_h1.py --device GPU --enable-cuStateVec True --precision double --bits $B --num-nucl-iters 10 --num-elec-iters 20 --st-method $METHOD --use-saved-data False --delta-t 0.001
done
