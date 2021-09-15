#!/bin/bash

source activate py37

batch=1
dataset=mmwhs

CUDA_VISIBLE_DEVICES=1,3 python main_supcon.py  --batch_size ${batch} --dataset ${dataset} \
    --data_folder ./data \
    --learning_rate 0.01 \
    --epochs 10 \
    --save_freq 5 \
    --cosine \
