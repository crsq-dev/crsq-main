#!/bin/sh -e

source ../../.venv/bin/activate
source ../../.env
export PYTHONPATH

python3 ./time_evo_h1.py
