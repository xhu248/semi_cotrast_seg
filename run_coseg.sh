#!/bin/bash


source activate py37

batch=4
dataset=Hippocampus
fold=3
head=mlp
mode=stride
temp=0.1
train_sample=1
# dataset=mmwhs


python main_coseg.py  --batch_size ${batch} --dataset ${dataset} \
    --data_folder ./data \
    --learning_rate 0.0001 \
    --epochs 60 \
    --head ${head} \
    --mode ${mode} \
    --fold ${fold} \
    --save_freq 1 \
    --print_freq 10 \
    --temp ${temp} \
    --train_sample ${train_sample} \
    --pretrained_model_path save/simclr/Hippocampus/b_80_model.pth \
    # --pretrained_model_path save/simclr/Hippocampus/b_80_model.pth \

