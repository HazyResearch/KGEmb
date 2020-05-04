#!/bin/bash
cd .. 
source set_env.sh
python run.py \
            --dataset WN18RR \
            --model ComplEx \
            --rank 32 \
            --regularizer N3 \
            --reg 0.05 \
            --optimizer Adagrad \
            --max_epochs 100 \
            --patience 15 \
            --valid 5 \
            --batch_size 100 \
            --neg_sample_size -1 \
            --init_size 0.001 \
            --learning_rate 0.1 \
            --gamma 0.0 \
            --bias none \
            --dtype single 
cd examples/
