cd .. 
source set_env.sh
python run.py \
            --dataset YAGO3-10 \
            --model RotH \
            --rank 32 \
            --regularizer N3 \
            --reg 0.0 \
            --optimizer Adam \
            --max_epochs 500 \
            --patience 20 \
            --valid 5 \
            --batch_size 1000 \
            --neg_sample_size -1 \
            --init_size 0.001 \
            --learning_rate 0.0005 \
            --gamma 0.0 \
            --bias learn \
            --dtype double \
            --multi_c
cd examples/
