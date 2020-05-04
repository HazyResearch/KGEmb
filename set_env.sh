#!/bin/bash
KGHOME=$(pwd)
export PYTHONPATH="$KGHOME:$PYTHONPATH"
export LOG_DIR="$KGHOME/logs"
export DATA_PATH="$KGHOME/data"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
source hyp_kg_env/bin/activate
